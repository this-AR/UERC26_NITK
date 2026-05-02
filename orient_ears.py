"""
orient_ears.py
==============
Use a trained orientation classifier to:
  1. Predict whether each ear image is LEFT or RIGHT
  2. Horizontally flip all LEFT ears → unified RIGHT-facing dataset

USAGE (after training)
----------------------
    python orient_ears.py \
        --model     ear_orientation_model.pt \
        --input_dir UERC26_data/data/public \
        --output_dir UERC26_data/data/public_oriented \
        --left_label 0 \
        --batch_size 128 \
        --copy_rights          # also copy already-right images to output

    # Dry run (no files written) — useful to check predictions first:
    python orient_ears.py --model ear_orientation_model.pt \
        --input_dir UERC26_data/data/public --dry_run

    # Confidence threshold — skip uncertain images:
    python orient_ears.py ... --confidence 0.80

OUTPUT
------
    output_dir/
        <subject_id>/          ← same folder structure as input
            img.png            ← already-right images copied as-is
            img_flipped.png    ← left images that were horizontally flipped

    orientation_report.csv     ← per-image predictions, confidence, action taken
"""

import argparse
import os
import csv
import glob
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models import MobileNet_V3_Small_Weights
from PIL import Image, ImageOps
from tqdm import tqdm

# ── CLI args ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Orient all ear images to face the same direction")
    p.add_argument("--model",       required=True,
                   help="Path to trained .pt checkpoint from train_ear_orientation.py")
    p.add_argument("--input_dir",   required=True,
                   help="Root of the ear dataset (contains subject sub-folders OR flat folder)")
    p.add_argument("--output_dir",  default="oriented_ears",
                   help="Where to write oriented images")
    p.add_argument("--left_label",  type=int, default=0,
                   help="Integer class index that means LEFT ear (check class_to_idx in checkpoint). "
                        "Default 0 assumes alphabetical order: left=0, right=1")
    p.add_argument("--extensions",  nargs="+", default=[".png", ".jpg", ".jpeg"],
                   help="Image extensions to process")
    p.add_argument("--batch_size",  type=int, default=64)
    p.add_argument("--confidence",  type=float, default=0.0,
                   help="Min softmax confidence to act on prediction. "
                        "Images below threshold are SKIPPED (copied as-is with a warning).")
    p.add_argument("--copy_rights", action="store_true",
                   help="Copy right-facing images to output_dir as-is. "
                        "If not set, only flipped images are written.")
    p.add_argument("--dry_run",     action="store_true",
                   help="Print actions without writing any files")
    p.add_argument("--workers",     type=int, default=4)
    return p.parse_args()

# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)

    # Support plain state-dict saves too
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state       = ckpt["model_state"]
        class_to_idx = ckpt.get("class_to_idx")
        img_size    = ckpt.get("img_size", 224)
    else:
        state        = ckpt
        class_to_idx = None
        img_size     = 224

    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 2)
    model.load_state_dict(state)
    model.eval().to(device)

    print(f"Loaded model from {ckpt_path}")
    if class_to_idx:
        print(f"  class_to_idx: {class_to_idx}")
    print(f"  img_size: {img_size}")

    return model, img_size, class_to_idx

# ── Image discovery ───────────────────────────────────────────────────────────

def find_images(input_dir, extensions):
    """Return list of absolute image paths under input_dir."""
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

def predict_batch(model, image_paths, transform, device, batch_size):
    """
    Returns list of (pred_class, confidence) for each path.
    Processes images in batches for speed.
    """
    results = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        tensors = []
        valid_idx = []
        for j, p in enumerate(batch_paths):
            try:
                img = Image.open(p).convert("RGB")
                tensors.append(transform(img))
                valid_idx.append(j)
            except Exception as e:
                print(f"  [WARN] Could not read {p}: {e}")
                results.append((None, 0.0))   # placeholder

        if not tensors:
            continue

        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            logits = model(batch)
            probs  = F.softmax(logits, dim=1)
            confs, preds = probs.max(dim=1)

        pred_list = []
        for pred, conf in zip(preds.cpu().tolist(), confs.cpu().tolist()):
            pred_list.append((pred, conf))

        # Merge back with placeholders for unreadable images
        result_for_batch = [(None, 0.0)] * len(batch_paths)
        for k, vi in enumerate(valid_idx):
            result_for_batch[vi] = pred_list[k]

        results.extend(result_for_batch)

    return results

# ── File I/O helpers ──────────────────────────────────────────────────────────

def compute_out_path(input_path, input_dir, output_dir, suffix=""):
    """Mirror the directory structure of input_dir inside output_dir."""
    rel = os.path.relpath(input_path, input_dir)
    stem, ext = os.path.splitext(rel)
    out_rel = stem + suffix + ext
    return os.path.join(output_dir, out_rel)

def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    model, img_size, class_to_idx = load_model(args.model, device)
    transform = build_transform(img_size)

    # Confirm which label is LEFT
    left_label = args.left_label
    if class_to_idx:
        # Auto-detect if class_to_idx is available
        if "left" in class_to_idx:
            left_label = class_to_idx["left"]
            print(f"Auto-detected LEFT label from checkpoint: {left_label}")
    print(f"LEFT class index = {left_label}  (images with this label will be flipped)\n")

    # Discover images
    print(f"Scanning {args.input_dir} ...")
    all_paths = find_images(args.input_dir, args.extensions)
    print(f"Found {len(all_paths)} images\n")

    if not all_paths:
        print("No images found. Check --input_dir and --extensions.")
        return

    # Run inference
    print("Running orientation classifier ...")
    predictions = predict_batch(
        model, all_paths, transform, device, args.batch_size
    )

    # Process + write
    os.makedirs(args.output_dir, exist_ok=True) if not args.dry_run else None

    stats = {"flipped": 0, "kept": 0, "skipped_low_conf": 0, "error": 0}
    report_rows = []

    print("\nProcessing images ...")
    for img_path, (pred, conf) in tqdm(zip(all_paths, predictions),
                                        total=len(all_paths)):
        if pred is None:
            stats["error"] += 1
            report_rows.append([img_path, "ERROR", "", "", "read_error"])
            continue

        is_left  = (pred == left_label)
        low_conf = (conf < args.confidence)
        action   = "unknown"

        if low_conf:
            # Uncertain → copy as-is (no flip) with warning flag
            action = "skipped_low_conf"
            stats["skipped_low_conf"] += 1
            out_path = compute_out_path(img_path, args.input_dir, args.output_dir)
            if not args.dry_run:
                ensure_dir(out_path)
                Image.open(img_path).convert("RGB").save(out_path)

        elif is_left:
            # Flip horizontally → becomes right
            action = "flipped"
            stats["flipped"] += 1
            out_path = compute_out_path(img_path, args.input_dir,
                                        args.output_dir, suffix="_flipped")
            if not args.dry_run:
                ensure_dir(out_path)
                img = Image.open(img_path).convert("RGB")
                ImageOps.mirror(img).save(out_path)

        else:
            # Already right → copy or skip
            action = "kept"
            stats["kept"] += 1
            out_path = compute_out_path(img_path, args.input_dir, args.output_dir)
            if args.copy_rights and not args.dry_run:
                ensure_dir(out_path)
                Image.open(img_path).convert("RGB").save(out_path)

        label_name = "left" if is_left else "right"
        report_rows.append([img_path, label_name, f"{conf:.4f}", out_path, action])

    # Write CSV report
    report_path = os.path.join(args.output_dir, "orientation_report.csv")
    if not args.dry_run:
        with open(report_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["input_path", "predicted_label", "confidence",
                             "output_path", "action"])
            writer.writerows(report_rows)
        print(f"\nReport saved → {report_path}")

    # Summary
    total = len(all_paths)
    print("\n" + "="*55)
    print("SUMMARY")
    print("="*55)
    print(f"  Total images       : {total}")
    print(f"  Flipped (L→R)      : {stats['flipped']} "
          f"({100*stats['flipped']/total:.1f}%)")
    print(f"  Kept as-is (right) : {stats['kept']} "
          f"({100*stats['kept']/total:.1f}%)")
    print(f"  Low confidence skip: {stats['skipped_low_conf']}")
    print(f"  Read errors        : {stats['error']}")
    if args.dry_run:
        print("\n  [DRY RUN] No files were written.")
    print("="*55)

if __name__ == "__main__":
    main()