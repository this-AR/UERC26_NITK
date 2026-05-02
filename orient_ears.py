"""
orient_ears.py
==============
Use a trained orientation classifier to:
  1. Predict whether each ear image is LEFT or RIGHT
  2. Horizontally flip all LEFT ears → unified RIGHT-facing dataset
  3. Save everything as WebP lossless (consistent with the pipeline)

USAGE (after training)
----------------------
    python orient_ears.py \
        --model      ear_orientation_model.pt \
        --input_dir  "C:/Users/sansk/Downloads/UERC26_Modified" \
        --output_dir "C:/Users/sansk/Downloads/UERC26_Oriented" \
        --extensions .webp \
        --copy_rights \
        --confidence 0.85

    # Dry run — preview actions without writing anything:
    python orient_ears.py \
        --model ear_orientation_model.pt \
        --input_dir "C:/Users/sansk/Downloads/UERC26_Modified" \
        --dry_run

OUTPUT
------
    UERC26_Oriented/
        <subject_id>/
            img.webp           ← right-facing images (copied or already correct)
            img.webp           ← left images that were horizontally flipped
    orientation_report.csv     ← per-image log: label, confidence, action taken

PIPELINE POSITION
-----------------
    standalone_precompute.py  →  ear_labeller.py  →  train_ear_orientation.py
    → THIS SCRIPT → UERC26_Oriented (ready for UERC26 training)
"""

import argparse
import csv
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models import MobileNet_V3_Small_Weights
from PIL import Image, ImageOps
from tqdm import tqdm

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Classify ear orientation and flip left ears to match right"
    )
    p.add_argument("--model",       required=True,
                   help="Path to ear_orientation_model.pt from train_ear_orientation.py")
    p.add_argument("--input_dir",   required=True,
                   help="Root of the WebP dataset (UERC26_Modified)")
    p.add_argument("--output_dir",  default="UERC26_Oriented",
                   help="Where to write the orientation-corrected dataset")
    p.add_argument("--extensions",  nargs="+", default=[".webp"],
                   help="Image extensions to process (default: .webp only)")
    p.add_argument("--batch_size",  type=int, default=64,
                   help="Inference batch size")
    p.add_argument("--confidence",  type=float, default=0.85,
                   help="Min softmax confidence to act on. Images below this "
                        "threshold are copied as-is and flagged in the report.")
    p.add_argument("--copy_rights", action="store_true",
                   help="Copy right-facing images to output_dir (recommended). "
                        "If not set, only flipped images are written.")
    p.add_argument("--dry_run",     action="store_true",
                   help="Show what would happen without writing any files")
    return p.parse_args()

# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state        = ckpt["model_state"]
        class_to_idx = ckpt.get("class_to_idx")
        img_size     = ckpt.get("img_size", 224)
    else:
        # Plain state dict (legacy save)
        state        = ckpt
        class_to_idx = None
        img_size     = 224

    model = models.mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
    model.load_state_dict(state)
    model.eval().to(device)

    print(f"Model loaded from: {ckpt_path}")
    print(f"  class_to_idx : {class_to_idx}")
    print(f"  img_size     : {img_size}")
    return model, img_size, class_to_idx

# ── Image discovery ───────────────────────────────────────────────────────────

def find_images(input_dir, extensions):
    exts = {e.lower() for e in extensions}
    paths = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if Path(f).suffix.lower() in exts:
                paths.append(os.path.join(root, f))
    return sorted(paths)

# ── Inference ─────────────────────────────────────────────────────────────────

def build_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

def predict_all(model, image_paths, transform, device, batch_size):
    """
    Run inference in batches.
    Returns list of (pred_class: int | None, confidence: float) per path.
    """
    results = []
    for i in tqdm(range(0, len(image_paths), batch_size),
                  desc="Classifying", unit="batch"):
        batch_paths = image_paths[i : i + batch_size]
        tensors, valid_idx = [], []

        for j, p in enumerate(batch_paths):
            try:
                tensors.append(transform(Image.open(p).convert("RGB")))
                valid_idx.append(j)
            except Exception as e:
                print(f"  [WARN] Skipping unreadable image {p}: {e}")

        # Placeholder row for each path in the batch
        batch_results = [(None, 0.0)] * len(batch_paths)

        if tensors:
            batch = torch.stack(tensors).to(device)
            with torch.no_grad():
                probs = F.softmax(model(batch), dim=1)
                confs, preds = probs.max(dim=1)
            for k, vi in enumerate(valid_idx):
                batch_results[vi] = (preds[k].item(), confs[k].item())

        results.extend(batch_results)
    return results

# ── File helpers ──────────────────────────────────────────────────────────────

def out_path_for(src, input_dir, output_dir):
    """Mirror directory structure; always use .webp extension."""
    rel  = os.path.relpath(src, input_dir)
    rel  = str(Path(rel).with_suffix(".webp"))
    return os.path.join(output_dir, rel)

def save_webp(img, path, lossless=True):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path, format="WEBP", lossless=lossless)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    model, img_size, class_to_idx = load_model(args.model, device)
    transform = build_transform(img_size)

    # Determine left label index
    left_label = 0   # default: alphabetical order → left=0
    if class_to_idx and "left" in class_to_idx:
        left_label = class_to_idx["left"]
        print(f"Auto-detected LEFT class index: {left_label}\n")

    print(f"Scanning {args.input_dir} ...")
    all_paths = find_images(args.input_dir, args.extensions)
    print(f"Found {len(all_paths)} images\n")

    if not all_paths:
        print("No images found. Check --input_dir and --extensions.")
        return

    predictions = predict_all(model, all_paths, transform, device, args.batch_size)

    if not args.dry_run:
        os.makedirs(args.output_dir, exist_ok=True)

    stats       = {"flipped": 0, "kept": 0, "low_conf": 0, "error": 0}
    report_rows = []

    print("\nWriting oriented images ...")
    for src, (pred, conf) in tqdm(zip(all_paths, predictions),
                                  total=len(all_paths), desc="Saving"):
        out = out_path_for(src, args.input_dir, args.output_dir)

        if pred is None:
            stats["error"] += 1
            report_rows.append([src, "ERROR", "0", out, "read_error"])
            continue

        is_left   = (pred == left_label)
        low_conf  = (conf < args.confidence)
        pred_name = "left" if is_left else "right"

        if low_conf:
            # Not confident — copy as-is and flag
            action = "low_conf_copy"
            stats["low_conf"] += 1
            if not args.dry_run:
                save_webp(Image.open(src).convert("RGB"), out)

        elif is_left:
            # Flip horizontally to convert left → right
            action = "flipped"
            stats["flipped"] += 1
            if not args.dry_run:
                save_webp(ImageOps.mirror(Image.open(src).convert("RGB")), out)

        else:
            # Already right-facing
            action = "kept"
            stats["kept"] += 1
            if args.copy_rights and not args.dry_run:
                save_webp(Image.open(src).convert("RGB"), out)

        report_rows.append([src, pred_name, f"{conf:.4f}", out, action])

    # Write report CSV
    report_path = os.path.join(args.output_dir, "orientation_report.csv")
    if not args.dry_run:
        with open(report_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["input_path", "predicted_label", "confidence",
                             "output_path", "action"])
            writer.writerows(report_rows)
        print(f"\nReport saved → {report_path}")

    total = len(all_paths)
    print("\n" + "=" * 58)
    print("SUMMARY")
    print("=" * 58)
    print(f"  Total images processed : {total}")
    print(f"  Flipped  (left → right): {stats['flipped']:>6}  "
          f"({100*stats['flipped']/max(total,1):.1f}%)")
    print(f"  Kept as-is (right)     : {stats['kept']:>6}  "
          f"({100*stats['kept']/max(total,1):.1f}%)")
    print(f"  Low confidence (<{args.confidence:.0%}): {stats['low_conf']:>6}")
    print(f"  Read errors            : {stats['error']:>6}")
    if args.dry_run:
        print("\n  [DRY RUN] No files were written.")
    print("=" * 58)
    print(f"\nOriented dataset saved to: {args.output_dir}")

if __name__ == "__main__":
    main()