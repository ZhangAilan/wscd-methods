import argparse
import datetime
import logging
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets import weaklyCD
from utils import evaluate_CD, imutils
from utils.AverageMeter import AverageMeter
from utils.camutils_CD import cam_to_label,multi_scale_cam
from utils.optimizer import PolyWarmupAdamW
from models.model_transwcd import TransWCD_dual, TransWCD_single

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
# LEVIR/DSIFN/WHU.yaml
parser.add_argument("--config",
                    default='configs/WHU.yaml',
                    type=str,
                    help="config")
parser.add_argument("--pooling", default="gmp", type=str, help="pooling method")
parser.add_argument("--crop_size", default=256, type=int, help="crop_size")
parser.add_argument("--scheme", default='transwcd_dual', type=str, help="transwcd_dual or transwcd_single")
parser.add_argument('--pretrained', default= True, type=bool, help="pretrained")
parser.add_argument('--checkpoint_path', default= False, type=str, help="checkpoint_path" )
parser.add_argument("--root_dir", default=None, type=str, help="override dataset root_dir in config")
parser.add_argument("--pretrained_backbone", default=None, type=str, help="override backbone config (e.g., mit_b0, mit_b1, mit_b2)")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def setup_logger(filename='test.log'):
    logFormatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s: %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fHandler = logging.FileHandler(filename, mode='w')
    fHandler.setFormatter(logFormatter)
    logger.addHandler(fHandler)

    cHandler = logging.StreamHandler()
    cHandler.setFormatter(logFormatter)
    logger.addHandler(cHandler)

def cal_eta(time0, cur_iter, total_iter):
    time_now = datetime.datetime.now()
    time_now = time_now.replace(microsecond=0)

    scale = (total_iter - cur_iter) / float(cur_iter)
    delta = (time_now - time0)
    eta = (delta * scale)
    time_fin = time_now + eta
    eta = time_fin.replace(microsecond=0) - time_now
    return str(delta), str(eta)

def get_down_size(ori_shape=(512, 512), stride=16):
    h, w = ori_shape
    _h = h // stride + 1 - ((h % stride) == 0)
    _w = w // stride + 1 - ((w % stride) == 0)
    return _h, _w

def validate(model=None, data_loader=None, cfg=None):
    preds, gts, cams = [], [], []
    model.eval()
    with torch.no_grad():
        for _, data in tqdm(enumerate(data_loader),
                            total=len(data_loader), ncols=100, ascii=" >="):
            name, inputs_A, inputs_B, labels, cls_label = data

            inputs_A = inputs_A.cuda()
            inputs_B = inputs_B.cuda()
            labels = labels.cuda()
            cls_label = cls_label.cuda()


            _cams = multi_scale_cam(model, inputs_A, inputs_B, cfg.cam.scales)
            resized_cam = F.interpolate(_cams, size=labels.shape[1:], mode='bilinear', align_corners=False)
            cam_label = cam_to_label(resized_cam, cls_label=cls_label, cfg=cfg)

            cams += list(cam_label.cpu().numpy().astype(np.int16))
            gts += list(labels.cpu().numpy().astype(np.int16))

    cam_score = evaluate_CD.scores(gts, cams)
    model.train()
    return cam_score, labels

def train(cfg):
    num_workers = 10

    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)

    # ========== 数据集加载信息 ==========
    logging.info("=" * 60)
    logging.info("Loading datasets...")
    
    train_dataset = weaklyCD.ClsDataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=cfg.dataset.name_list_dir,
        split=cfg.train.split,
        stage='train',
        aug=True,
        rescale_range=cfg.dataset.rescale_range,
        crop_size=cfg.dataset.crop_size,
        img_fliplr=True,
        num_classes=cfg.dataset.num_classes,
    )
    
    logging.info(f"  Train dataset: {len(train_dataset)} samples")
    logging.info(f"    - root_dir: {cfg.dataset.root_dir}")
    logging.info(f"    - name_list_dir: {cfg.dataset.name_list_dir}")
    logging.info(f"    - split: {cfg.train.split}")
    logging.info(f"    - crop_size: {cfg.dataset.crop_size}")
    logging.info(f"    - batch_size: {cfg.train.batch_size}")

    val_dataset = weaklyCD.CDDataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=cfg.dataset.name_list_dir,
        split=cfg.val.split,
        stage='val',
        aug=False,
        num_classes=cfg.dataset.num_classes,
    )
    
    logging.info(f"  Val dataset: {len(val_dataset)} samples")
    logging.info(f"    - split: {cfg.val.split}")
    logging.info("=" * 60)

    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.train.batch_size,
                              # shuffle=True,
                              num_workers=num_workers,
                              pin_memory=False,
                              drop_last=True,
                              prefetch_factor=4)

    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=False,
                            drop_last=False)
    device = torch.device('cuda')

    # ========== 模型信息 ==========
    print("=" * 60)
    print("Building model...")
    print(f"  Scheme: {cfg.scheme}")
    print(f"  Backbone: {cfg.backbone.config}")
    print(f"  Stride: {cfg.backbone.stride}")
    print(f"  Pretrained: {args.pretrained}")
    print(f"  Pooling: {args.pooling}")
    
    if cfg.scheme == "transwcd_dual":
        transwcd = TransWCD_dual(backbone=cfg.backbone.config,
                                 stride=cfg.backbone.stride,
                                 num_classes=cfg.dataset.num_classes,
                                 embedding_dim=256,
                                 pretrained=args.pretrained,
                                 pooling=args.pooling,
                                 pretrained_backbone=cfg.backbone.pretrained_backbone,
                                  )
    elif cfg.scheme == "transwcd_single":
        transwcd = TransWCD_single(backbone=cfg.backbone.config,
                                 stride=cfg.backbone.stride,
                                 num_classes=cfg.dataset.num_classes,
                                 embedding_dim=256,
                                 pretrained=args.pretrained,
                                 pooling=args.pooling,
                                 pretrained_backbone=cfg.backbone.pretrained_backbone,
                                  )
    else:
        print("Please choose a baseline structure in /configs/...yaml")

    # 打印模型参数量
    total_params = sum(p.numel() for p in transwcd.parameters())
    trainable_params = sum(p.numel() for p in transwcd.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print("=" * 60)
    
    param_groups = transwcd.get_param_groups()
    transwcd.to(device)

    writer = SummaryWriter(cfg.work_dir.logger_dir)
    print('writer:',writer)

    optimizer = PolyWarmupAdamW(
        params=[
            {
                "params": param_groups[0],
                "lr": cfg.optimizer.learning_rate,
                "weight_decay": cfg.optimizer.weight_decay,
            },
            {
                "params": param_groups[1],
                "lr": 0.0,  ## freeze norm layers
                "weight_decay": 0.0,
            },
            {
                "params": param_groups[2],
                "lr": cfg.optimizer.learning_rate * 10,
                "weight_decay": cfg.optimizer.weight_decay,
            },
        ],
        lr=cfg.optimizer.learning_rate,
        weight_decay=cfg.optimizer.weight_decay,
        betas=cfg.optimizer.betas,
        warmup_iter=cfg.scheduler.warmup_iter,
        max_iter=cfg.train.max_iters,
        warmup_ratio=cfg.scheduler.warmup_ratio,
        power=cfg.scheduler.power
    )

    train_loader_iter = iter(train_loader)
    avg_meter = AverageMeter()

    bkg_cls = torch.ones(size=(cfg.train.batch_size, 1))

    best_F1 = 0.0
    best_iter = 0
    
    # ========== 训练开始信息 ==========
    print("=" * 60)
    print("Start training...")
    print(f"  Max iterations: {cfg.train.max_iters}")
    print(f"  CAM iters: {cfg.train.cam_iters}")
    print(f"  Eval iters: {cfg.train.eval_iters}")
    print(f"  Log iters: {cfg.train.log_iters}")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print("=" * 60)
    
    # 使用 tqdm 包装训练循环，每次迭代都显示进度
    pbar = tqdm(range(cfg.train.max_iters), ncols=100, ascii=" >=")
    
    for n_iter in pbar:

        try:
            img_name, inputs_A, inputs_B, cls_labels, img_box = next(train_loader_iter)
        except:
            train_loader_iter = iter(train_loader)
            img_name, inputs_A, inputs_A, cls_labels, img_box = next(train_loader_iter)

        inputs_A = inputs_A.to(device, non_blocking=True)
        inputs_B = inputs_B.to(device, non_blocking=True)

        cls_labels = cls_labels.to(device, non_blocking=True)

        cls = transwcd(inputs_A, inputs_B)

        cams = multi_scale_cam(transwcd, inputs_A=inputs_A, inputs_B=inputs_B, scales=cfg.cam.scales)

        valid_cam, pred_cam = cam_to_label(cams.detach(), cls_label=cls_labels, img_box=img_box, ignore_mid=True,
                                           cfg=cfg)

        bkg_cls = bkg_cls.to(cams.device)
        _cls_labels = torch.cat((bkg_cls, cls_labels), dim=1)

        # change classification loss
        cc_loss = F.binary_cross_entropy_with_logits(cls, cls_labels)
        
        # ========== 打印正负样本比例 ==========
        if n_iter == 0:
            pos_ratio = cls_labels.sum().item() / cls_labels.numel() * 100
            print(f"\n  First batch cls_labels positive ratio: {pos_ratio:.2f}%")
            print(f"  CAM shape: {cams.shape}")
            print(f"  CAM value range: [{cams.min().item():.4f}, {cams.max().item():.4f}]")

        if n_iter <= cfg.train.cam_iters:
            loss = 1.0 * cc_loss
        else:
            loss = 1.0 * cc_loss

        avg_meter.add({'cc_loss': cc_loss.item()})

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # ========== 每次迭代更新进度条 ==========
        pbar.set_description(f"Loss: {cc_loss.item():.4f}")

        if (n_iter + 1) % cfg.train.log_iters == 0:

            delta, eta = cal_eta(time0, n_iter + 1, cfg.train.max_iters)
            cur_lr = optimizer.param_groups[0]['lr']

            pred_cam = pred_cam.cpu().numpy().astype(np.int16)
            cur_loss = avg_meter.pop('cc_loss')

            # ========== 详细训练日志（每 log_iters 次迭代打印一次） ==========
            print(f"\n[Iter {n_iter+1}/{cfg.train.max_iters}]")
            print(f"  Elapsed: {delta}, ETA: {eta}")
            print(f"  LR: {cur_lr:.3e}")
            print(f"  cc_loss: {cur_loss:.4f}")
            print(f"  CAM stats: mean={cams.mean().item():.4f}, std={cams.std().item():.4f}")
            print(f"  Pred CAM positive ratio: {pred_cam.mean() / 255 * 100:.2f}%")

            grid_imgs_A, grid_cam_A = imutils.tensorboard_image(imgs=inputs_A.clone(), cam=valid_cam)
            grid_imgs_B, grid_cam_B = imutils.tensorboard_image(imgs=inputs_B.clone(), cam=valid_cam)

            grid_pred_cam = imutils.tensorboard_label(labels=pred_cam)

            writer.add_image("train/images_A"+str(img_name), grid_imgs_A, global_step=n_iter)
            writer.add_image("train/images_B"+str(img_name), grid_imgs_B, global_step=n_iter)
            writer.add_image("cam/valid_cams_A", grid_cam_A, global_step=n_iter)
            writer.add_image("cam/valid_cams_B", grid_cam_B, global_step=n_iter)
            writer.add_image("train/preds_cam", grid_pred_cam, global_step=n_iter)

            writer.add_scalars('train/loss', {"cc_loss": cc_loss.item()},
                               global_step=n_iter)

        if (n_iter + 1) % cfg.train.eval_iters == 0:
            ckpt_name = os.path.join(cfg.work_dir.ckpt_dir, "transwcd_iter_%d.pth" % (n_iter + 1))
            print('\n' + "=" * 40)
            print('CD Validating...')
            print("=" * 40)
            torch.save(transwcd.state_dict(), ckpt_name)
            cam_score, labels = validate(model=transwcd, data_loader=val_loader, cfg=cfg)  # _ 为 labels

            if cam_score['f1'][1] > best_F1:
                best_F1 = cam_score['f1'][1]
                best_iter = n_iter + 1
            
            # ========== 打印验证结果 ==========
            print(f"\n  Validation Results:")
            print(f"    Precision (change): {cam_score['precision'][1]:.4f}")
            print(f"    Recall (change):    {cam_score['recall'][1]:.4f}")
            print(f"    F1 (change):        {cam_score['f1'][1]:.4f}")
            print(f"    IoU (change):       {cam_score['iou'][1]:.4f}")
            print(f"    Best F1:            {best_F1:.4f} [iter {best_iter}]")
            print("=" * 60)
              
    return True

if __name__ == "__main__":
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    # 命令行覆盖 root_dir
    if args.root_dir is not None:
        cfg.dataset.root_dir = args.root_dir
        print(f"Override dataset root_dir with: {args.root_dir}")

    # 命令行覆盖 backbone config
    if args.pretrained_backbone is not None:
        cfg.backbone.pretrained_backbone = args.pretrained_backbone
        print(f"Override backbone pretrained_backbone with: {args.pretrained_backbone}")


    timestamp = "{0:%Y-%m-%d-%H-%M}".format(datetime.datetime.now())

    cfg.work_dir.ckpt_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.ckpt_dir, timestamp)
    cfg.work_dir.pred_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.pred_dir)
    cfg.work_dir.logger_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.logger_dir, timestamp)

    os.makedirs(cfg.work_dir.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.pred_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.logger_dir, exist_ok=True)

    setup_logger(filename=os.path.join(cfg.work_dir.dir, timestamp + '.log'))
    logging.info('\nargs: %s' % args)
    logging.info('\nconfigs: %s' % cfg)
    
    # ========== 打印配置信息 ==========
    print("\n" + "=" * 60)
    print("Configuration:")
    print("=" * 60)
    print(f"  Config file: {args.config}")
    print(f"  Dataset root: {cfg.dataset.root_dir}")
    print(f"  Backbone: {cfg.backbone.config}")
    print(f"  Scheme: {cfg.scheme}")
    print(f"  Work dir: {cfg.work_dir.dir}")
    print(f"  Timestamp: {timestamp}")
    print("=" * 60)

    setup_seed(1)
    train(cfg=cfg)

