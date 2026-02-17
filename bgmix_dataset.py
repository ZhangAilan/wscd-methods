import os
import argparse
from pathlib import Path
import cv2
import numpy as np
import shutil


def classify_by_label(label_path):
    """
    根据标签图像的值判断是变化图像还是未变化图像
    如果标签中有非零像素，则认为是变化图像（返回 1）；否则为未变化图像（返回 0）
    """
    label_img = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
    if label_img is None:
        raise ValueError(f"Could not read label image: {label_path}")
    
    # 检查标签图像中是否有非零像素
    has_changes = np.any(label_img > 0)
    return 1 if has_changes else 0


def create_bgmix_dataset_structure(input_path, output_path):
    """
    将 Levir_CDC_dataset 或 WHU_CDC_dataset 结构转换为 bgmix 所需的数据集结构

    输入结构:
    Dataset/
    ├── images/
        ├── train/
        │   ├── A/
        │   ├── B/
        │   └── label/
        ├── val/
        │   ├── A/
        │   ├── B/
        │   └── label/
        └── test/
            ├── A/
            ├── B/
            └── label/

    输出结构 (同时生成两种格式):
    
    1. 用于主模型训练 (model="CD"):
    datadir/
    ├── train/
    │   ├── train_C/           # 用于训练变化检测器的变化图像对
    │   │   ├── image.txt      # A 时相图像路径
    │   │   ├── image2.txt     # B 时相图像路径 
    │   │   └── label.txt      # 标签图像路径
    │   └── train_UC/          # 用于训练背景模型的未变化图像对
    │       ├── image.txt      # A 时相图像路径
    │       ├── image2.txt     # B 时相图像路径
    │       └── label.txt      # 标签图像路径
    ├── test/
    │   ├── image.txt          # A 时相测试图像路径
    │   ├── image2.txt         # B 时相测试图像路径
    │   └── label.txt          # 测试标签路径
    └── test_labels/           # 标准答案标签

    2. 用于分类器训练 (model="SC"):
    datadir/
    ├── train/
    │   ├── image.txt          # 格式：image_path label (如：/path/img.png 0)
    │   └── image2.txt         # 格式：image_path label
    └── test/
        ├── image.txt
        └── image2.txt
    """

    input_path = Path(input_path)
    output_path = Path(output_path)

    # 创建输出目录
    train_output_dir = output_path / "train"
    test_output_dir = output_path / "test"
    test_labels_dir = output_path / "test_labels"

    # 为训练数据创建 C 和 UC 子目录
    train_c_dir = train_output_dir / "train_C"
    train_uc_dir = train_output_dir / "train_UC"
    
    train_c_dir.mkdir(parents=True, exist_ok=True)
    train_uc_dir.mkdir(parents=True, exist_ok=True)
    test_output_dir.mkdir(parents=True, exist_ok=True)
    test_labels_dir.mkdir(parents=True, exist_ok=True)

    # ==================== 处理训练数据 ====================
    print("Processing train split...")
    split_input_dir = input_path / "images" / "train"

    # 获取所有图像文件
    image_a_dir = split_input_dir / "A"
    image_b_dir = split_input_dir / "B"
    label_dir = split_input_dir / "label"

    # 获取图像文件列表
    image_a_files = sorted([f for f in os.listdir(image_a_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    image_b_files = sorted([f for f in os.listdir(image_b_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    label_files = sorted([f for f in os.listdir(label_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    # 确保三个列表长度相同
    assert len(image_a_files) == len(image_b_files) == len(label_files), \
        f"Mismatch in number of files for train: A={len(image_a_files)}, B={len(image_b_files)}, label={len(label_files)}"

    print(f"Found {len(image_a_files)} image pairs for train")

    # 分类图像对
    c_image_a_paths, c_image_b_paths, c_label_paths = [], [], []
    uc_image_a_paths, uc_image_b_paths, uc_label_paths = [], [], []
    sc_image_a_labels, sc_image_b_labels = [], []

    for img_a, img_b, lbl in zip(image_a_files, image_b_files, label_files):
        label_path = label_dir / lbl
        label_value = classify_by_label(label_path)
        
        # 用于主模型训练 (CD) 的数据 - 按标签分类
        if label_value == 1:
            c_image_a_paths.append(str(image_a_dir / img_a))
            c_image_b_paths.append(str(image_b_dir / img_b))
            c_label_paths.append(str(label_dir / lbl))
        else:
            uc_image_a_paths.append(str(image_a_dir / img_a))
            uc_image_b_paths.append(str(image_b_dir / img_b))
            uc_label_paths.append(str(label_dir / lbl))
        
        # 用于分类器训练 (SC) 的数据 - 保留所有数据，带图像级标签
        sc_image_a_labels.append(f"{image_a_dir / img_a} {label_value}")
        sc_image_b_labels.append(f"{image_b_dir / img_b} {label_value}")

    print(f"Classified {len(c_image_a_paths)} as C (changed) and {len(uc_image_a_paths)} as UC (unchanged)")

    # ========== 写入主模型训练数据 (model="CD") ==========
    # 写入 C 类别的训练文件
    with open(train_c_dir / "image.txt", 'w', encoding='utf-8') as f1, \
         open(train_c_dir / "image2.txt", 'w', encoding='utf-8') as f2, \
         open(train_c_dir / "label.txt", 'w', encoding='utf-8') as f3:
        for img_a_path, img_b_path, lbl_path in zip(c_image_a_paths, c_image_b_paths, c_label_paths):
            f1.write(img_a_path + '\n')
            f2.write(img_b_path + '\n')
            f3.write(lbl_path + '\n')

    # 写入 UC 类别的训练文件
    with open(train_uc_dir / "image.txt", 'w', encoding='utf-8') as f1, \
         open(train_uc_dir / "image2.txt", 'w', encoding='utf-8') as f2, \
         open(train_uc_dir / "label.txt", 'w', encoding='utf-8') as f3:
        for img_a_path, img_b_path, lbl_path in zip(uc_image_a_paths, uc_image_b_paths, uc_label_paths):
            f1.write(img_a_path + '\n')
            f2.write(img_b_path + '\n')
            f3.write(lbl_path + '\n')

    # ========== 写入分类器训练数据 (model="SC") ==========
    # 写入 train/image.txt (格式：image_path label)
    with open(train_output_dir / "image.txt", 'w', encoding='utf-8') as f1, \
         open(train_output_dir / "image2.txt", 'w', encoding='utf-8') as f2:
        for line_a, line_b in zip(sc_image_a_labels, sc_image_b_labels):
            f1.write(line_a + '\n')
            f2.write(line_b + '\n')

    # ==================== 处理测试数据 ====================
    print("Processing test split...")
    test_split_input_dir = input_path / "images" / "test"

    test_image_a_dir = test_split_input_dir / "A"
    test_image_b_dir = test_split_input_dir / "B"
    test_label_dir = test_split_input_dir / "label"

    # 获取测试图像文件列表
    test_image_a_files = sorted([f for f in os.listdir(test_image_a_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    test_image_b_files = sorted([f for f in os.listdir(test_image_b_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    test_label_files = sorted([f for f in os.listdir(test_label_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    # 确保三个列表长度相同
    assert len(test_image_a_files) == len(test_image_b_files) == len(test_label_files), \
        f"Mismatch in number of files for test: A={len(test_image_a_files)}, B={len(test_image_b_files)}, label={len(test_label_files)}"

    print(f"Found {len(test_image_a_files)} image pairs for test")

    # ========== 写入主模型测试数据 (model="CD") ==========
    with open(test_output_dir / "image.txt", 'w', encoding='utf-8') as f1, \
         open(test_output_dir / "image2.txt", 'w', encoding='utf-8') as f2, \
         open(test_output_dir / "label.txt", 'w', encoding='utf-8') as f3:

        for img_a, img_b, lbl in zip(test_image_a_files, test_image_b_files, test_label_files):
            img_a_path = str(test_image_a_dir / img_a)
            img_b_path = str(test_image_b_dir / img_b)
            lbl_path = str(test_label_dir / lbl)
            
            f1.write(img_a_path + '\n')
            f2.write(img_b_path + '\n')
            f3.write(lbl_path + '\n')
            
            # 同时复制标签到 test_labels 目录
            src_label_path = test_label_dir / lbl
            dst_label_path = test_labels_dir / lbl
            shutil.copy2(src_label_path, dst_label_path)

    # ========== 写入分类器测试数据 (model="SC") ==========
    with open(test_output_dir / "image.txt", 'w', encoding='utf-8') as f1, \
         open(test_output_dir / "image2.txt", 'w', encoding='utf-8') as f2:
        for img_a, img_b, lbl in zip(test_image_a_files, test_image_b_files, test_label_files):
            label_val = classify_by_label(test_label_dir / lbl)
            f1.write(f"{test_image_a_dir / img_a} {label_val}\n")
            f2.write(f"{test_image_b_dir / img_b} {label_val}\n")

    print(f"Dataset conversion completed. Output saved to: {output_path}")
    print("\n生成的数据集结构说明:")
    print("1. 主模型训练 (train.py): 使用 --datadir 指向 train_C 或 train_UC 目录")
    print("2. 分类器训练 (train_Classifier.py): 使用 --datadir 指向 train 目录")


def main():
    parser = argparse.ArgumentParser(description='Convert CDC dataset to bgmix format')
    parser.add_argument('--input_path', type=str, required=True,
                        help='Input path to CDC dataset (Levir_CDC_dataset or whu_CDC_dataset)')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Output path for converted dataset (default: input_path parent directory with _bgmix suffix)')

    args = parser.parse_args()

    # 如果没有指定输出路径，则使用输入路径的父目录下创建输出目录
    if args.output_path is None:
        input_path = Path(args.input_path)
        output_path = input_path.parent / (input_path.name + "_bgmix")
    else:
        output_path = Path(args.output_path)

    print(f"Converting dataset from: {args.input_path}")
    print(f"Output will be saved to: {output_path}")

    create_bgmix_dataset_structure(args.input_path, output_path)


if __name__ == "__main__":
    main()
