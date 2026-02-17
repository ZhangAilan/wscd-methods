# 弱监督变化检测方法复现

## 环境配置
```
conda create --name wscd python=3.8 -y
conda activate wscd
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130   ##根据自己的cuda版本选择
pip install -r requirements.txt
pip install git+https://github.com/lucasb-eyer/pydensecrf.git
```

## 数据集
[WHU, LEVIR](https://huggingface.co/datasets/hygge10111/RS-CDC/tree/main)
```
Levir_CDC_dataset\
├───LevirCCcaptions.json
└───images\
    ├───test\
    │   ├───A\
    │   ├───B\
    │   └───label\
    ├───train\
    │   ├───A\
    │   ├───B\
    │   └───label\
    └───val\
        ├───A\
        ├───B\
        └───label\
```
### 调整数据集格式
```python
python adapt_dataset.py --input_root E:\weakly_CD_dataset\dataset\whu_CDC_dataset\whu_CDC_dataset

python adapt_dataset.py --input_root E:\weakly_CD_dataset\dataset\Levir_CDC_dataset\Levir_CDC_dataset
```

```
LEVIR-MCI-dataset_converted/
├── A/
│   ├── train_000001.png
│   └── ...
├── B/
├── label/
├── list/
│   ├── train.txt        # 每行：train_000001.png
│   ├── train_label.txt
│   ├── val.txt
│   ├── val_label.txt
│   ├── test.txt
│   └── test_label.txt
└── LevirCCcaptions.json
```


## [TransWCD](https://github.com/zhenghuizhao/TransWCD)
```
WSCD dataset with image-level labels:
├─A/                          # 第一时相图像（原始图像）
├─B/                          # 第二时相图像（变化后图像）
└─label/                      # 标注图像（像素级标签）
```

预训练权重```TransWCD/transwcd/pretrained```  [mit_b1.pth](https://drive.google.com/file/d/11Tf0NCm1ry_vmqqoCcKTdtmDEAA_PhTQ/view?usp=sharing)

修改```TransWCD/transwcd/configs/WHU.yaml```中```root_dir```为数据集路径

结果保存在```TransWCD/transwcd/work_dir_WHU/```


## [ACWCD](https://github.com/WenhaoLiu03/ACWCD)
```
Dataset Name (e.g., BCD, LEVIR, DSIFN, CLCD):
├─A/                          # 第一时相图像（原始图像）
├─B/                          # 第二时相图像（变化后图像）
└─label/                      # 标注图像（像素级标签）
```

预训练权重```ACWCD/pretrained```  [mit_b1.pth](https://drive.google.com/file/d/11Tf0NCm1ry_vmqqoCcKTdtmDEAA_PhTQ/view?usp=sharing)

WHU-CD数据：```ACWCD\configs\BCD.yaml```，修改```root_dir```为数据集路径


## [bgmix](https://github.com/tsingqguo/bgmix)
```
python bgmix_dataset.py --input_path E:\weakly_CD_dataset\dataset\whu_CDC_dataset\whu_CDC_dataset
```

```
./my_dataset/
├── train/
│   ├── train_C/           # 用于训练变化检测器的变化图像对
│   │   ├── image.txt      # A时相图像路径 (如: A/img1.png)
│   │   ├── image2.txt     # B时相图像路径 (如: B/img1.png) 
│   │   └── label.txt      # 标签图像路径 (如: label/img1.png)
│   └── train_UC/          # 用于训练背景模型的未变化图像对
│       ├── image.txt      # A时相图像路径
│       ├── image2.txt     # B时相图像路径
│       └── label.txt      # 标签图像路径
├── test/                  # 测试图像对
│   ├── image.txt          # A时相测试图像路径
│   ├── image2.txt         # B时相测试图像路径
│   └── label.txt          # 测试标签路径
└── test_labels/           # 标准答案标签
    ├── img1.png           # 标准标签图像
    ├── img2.png           # 标准标签图像
    └── ...
```

## [WSLCD](https://github.com/mfzhao1998/WSLCD)
```
dataset
├── train/
│   ├── A/          # 时相 1 图像
│   ├── B/          # 时相 2 图像
│   └── label/      # 像素级标签 (训练时用于验证，弱监督实际只用图像级标签)
└── val/
    ├── A/          # 时相 1 图像
    ├── B/          # 时相 2 图像
    └── label/    
```