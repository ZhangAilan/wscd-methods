import os
import shutil
import argparse
from pathlib import Path

import numpy as np
import matplotlib.image as mpimg


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert WHU-CDC dataset to WHU-CC format with label generation"
    )
    parser.add_argument(
        "--input_root",
        type=str,
        required=True,
        help="Path to input whu_CDC_dataset"
    )
    return parser.parse_args()


def main(args):
    input_root = Path(args.input_root)

    # Automatically set output_root to be at the same level as input_root with "_converted" suffix
    output_root = input_root.parent / f"{input_root.name}_converted"

    splits = ["train", "val", "test"]
    subdirs = ["A", "B", "label"]

    # ---------- create output dirs ----------
    for d in subdirs:
        (output_root / d).mkdir(parents=True, exist_ok=True)
    (output_root / "list").mkdir(parents=True, exist_ok=True)

    # ---------- process splits ----------
    for split in splits:
        split_dir = input_root / "images" / split
        list_file = output_root / "list" / f"{split}.txt"
        label_file = output_root / "list" / f"{split}_label.txt"

        if not split_dir.exists():
            print(f"[Warning] {split_dir} not found, skip.")
            continue

        a_dir = split_dir / "A"
        if not a_dir.exists():
            print(f"[Warning] {a_dir} not found, skip.")
            continue

        image_names = sorted(
            [p.name for p in a_dir.iterdir() if p.is_file()]
        )

        # ---------- write list file ----------
        with open(list_file, "w") as f:
            for name in image_names:
                f.write(name + "\n")

        # ---------- copy files ----------
        for name in image_names:
            for sub in subdirs:
                src = split_dir / sub / name
                dst = output_root / sub / name

                if not src.exists():
                    print(f"[Warning] Missing file: {src}")
                    continue

                shutil.copy2(src, dst)

        # ---------- generate label txt ----------
        with open(label_file, "w") as f:
            for name in image_names:
                label_path = output_root / "label" / name

                if not label_path.exists():
                    print(f"[Warning] Missing label: {label_path}")
                    continue

                label_img = mpimg.imread(label_path)

                if np.all(label_img == 0):
                    f.write(f"{name},0,0,0\n")
                else:
                    f.write(f"{name},0,0,1\n")

        print(f"[OK] {split}: {len(image_names)} samples")

    # ---------- copy json ----------
    import glob
    json_files = list(input_root.glob("*.json"))
    cc_caption_files = [f for f in json_files if "CCcaptions" in f.name]

    if cc_caption_files:
        # Copy the first matching CC captions JSON file
        json_file = cc_caption_files[0]
        shutil.copy2(json_file, output_root / json_file.name)
        print(f"[OK] {json_file.name} copied")
    else:
        print("[Warning] No *CCcaptions*.json file found")

    print("Dataset conversion finished.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
