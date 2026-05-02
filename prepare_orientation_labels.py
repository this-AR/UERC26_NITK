"""
prepare_orientation_labels.py
==============================
OPTIONAL HELPER — only needed if your dataset ALREADY has left/right ears
separated into known subjects but NOT labelled by orientation.

Two strategies are supported:

Strategy A — Manual sampling (recommended for most cases)
    Copy a random sample of images into labelled_ears/left/ and labelled_ears/right/
    so you can label them by hand (or with a quick visual inspection).

Strategy B — Mirror-based auto-labelling (zero annotation required!)
    Assumes your dataset has MIXED orientations. For each original image,
    create a "right" copy and a "left" copy (which is just the mirror).
    The model then learns to distinguish natural from mirrored ears.
    Works surprisingly well when combined with moderate data augmentation.

USAGE
-----
    # Strategy A — sample for manual labelling
    python prepare_orientation_labels.py \
        --input_dir  C:/Users/sansk/Downloads/UERC26_data(1)/data/public \
        --output_dir labelled_ears \
        --strategy   sample \
        --n_samples  300        # 300 per class target

    # Strategy B — auto mirror
    python prepare_orientation_labels.py \
        --input_dir  C:/Users/sansk/Downloads/UERC26_data(1)/data/public/public \
        --output_dir labelled_ears \
        --strategy   mirror \
        --n_samples  2000       # total images to use
"""

import argparse
import os
import random
import shutil
from pathlib import Path
from PIL import Image, ImageOps
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir",  required=True)
    p.add_argument("--output_dir", default="labelled_ears")
    p.add_argument("--strategy",   choices=["sample", "mirror"], default="mirror")
    p.add_argument("--n_samples",  type=int, default=2000)
    p.add_argument("--extensions", nargs="+", default=[".png", ".jpg", ".jpeg"])
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()


def find_images(root, extensions):
    exts = {e.lower() for e in extensions}
    return sorted([
        str(p) for p in Path(root).rglob("*")
        if p.suffix.lower() in exts
    ])


def strategy_sample(all_images, output_dir, n_samples, seed):
    """Copy a random sample to output_dir/unlabelled/ for hand-labelling."""
    random.seed(seed)
    chosen = random.sample(all_images, min(n_samples, len(all_images)))
    out = os.path.join(output_dir, "unlabelled")
    os.makedirs(out, exist_ok=True)
    print(f"Copying {len(chosen)} images to {out} for manual labelling...")
    for src in tqdm(chosen):
        shutil.copy2(src, os.path.join(out, Path(src).name))
    print(f"\nDone. Now manually sort images into:")
    print(f"  {output_dir}/left/")
    print(f"  {output_dir}/right/")
    print("Then delete the unlabelled/ folder and run train_ear_orientation.py.")


def strategy_mirror(all_images, output_dir, n_samples, seed):
    """
    Auto-label: original images → 'right' class,
                mirrored images → 'left' class.
    This works well because a mirrored RIGHT ear looks like a LEFT ear.
    """
    random.seed(seed)
    chosen = random.sample(all_images, min(n_samples, len(all_images)))

    right_dir = os.path.join(output_dir, "right")
    left_dir  = os.path.join(output_dir, "left")
    os.makedirs(right_dir, exist_ok=True)
    os.makedirs(left_dir, exist_ok=True)

    print(f"Creating mirror-based labels from {len(chosen)} images ...")
    for src in tqdm(chosen):
        stem = Path(src).stem
        ext  = Path(src).suffix

        img = Image.open(src).convert("RGB")

        # Original → right
        img.save(os.path.join(right_dir, f"{stem}{ext}"))
        # Mirrored → left
        ImageOps.mirror(img).save(os.path.join(left_dir, f"{stem}_mirror{ext}"))

    print(f"\nDone! Created:")
    print(f"  {len(chosen)} right images in {right_dir}")
    print(f"  {len(chosen)} left  images in {left_dir}")
    print("\nNow run:")
    print(f"  python train_ear_orientation.py --data_dir {output_dir} ...")
    print("\nNOTE: Mirror-based labels work best when the MAJORITY of your")
    print("dataset is already one orientation (e.g. mostly right-facing).")
    print("If your dataset is truly 50/50, you will need manual labels instead.")


def main():
    args = parse_args()
    all_images = find_images(args.input_dir, args.extensions)
    print(f"Found {len(all_images)} images in {args.input_dir}")

    if args.strategy == "sample":
        strategy_sample(all_images, args.output_dir, args.n_samples, args.seed)
    else:
        strategy_mirror(all_images, args.output_dir, args.n_samples, args.seed)


if __name__ == "__main__":
    main()