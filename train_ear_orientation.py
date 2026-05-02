"""
train_ear_orientation.py
========================
Train a lightweight CNN classifier to detect ear orientation (Left / Right).

USAGE
-----
1. Prepare a small labelled set (a few hundred images is enough):
       labelled_ears/
           left/   ← images of LEFT ears
           right/  ← images of RIGHT ears

2. Run:
       python train_ear_orientation.py \
           --data_dir  labelled_ears \
           --save_path ear_orientation_model.pt \
           --epochs    30 \
           --batch_size 32

The script uses MobileNetV3-Small (≈2 MB, fast on CPU/Pi) fine-tuned for
binary classification. Transfer learning means you only need ~200-500 images
per class to get good accuracy.
"""

import argparse
import os
import copy
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from torchvision.models import MobileNet_V3_Small_Weights

# ── CLI args ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train ear left/right orientation classifier")
    p.add_argument("--data_dir",   required=True,
                   help="Root folder with 'left/' and 'right/' sub-folders")
    p.add_argument("--save_path",  default="ear_orientation_model.pt",
                   help="Where to save the trained weights")
    p.add_argument("--epochs",     type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--val_split",  type=float, default=0.2,
                   help="Fraction of data used for validation")
    p.add_argument("--img_size",   type=int, default=224)
    p.add_argument("--workers",    type=int, default=4)
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()

# ── Data ──────────────────────────────────────────────────────────────────────

def build_loaders(data_dir, img_size, batch_size, val_split, workers, seed):
    train_tf = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),          # helps generalisation
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.05),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    full = datasets.ImageFolder(data_dir)
    print(f"Classes found: {full.class_to_idx}")   # e.g. {'left': 0, 'right': 1}

    n_val  = int(len(full) * val_split)
    n_train = len(full) - n_val
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full, [n_train, n_val], generator=g)

    # Assign different transforms
    train_ds.dataset = copy.deepcopy(full)
    train_ds.dataset.transform = train_tf
    val_ds.dataset  = copy.deepcopy(full)
    val_ds.dataset.transform  = val_tf

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=workers, pin_memory=True)

    return train_loader, val_loader, full.class_to_idx

# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(num_classes=2):
    """MobileNetV3-Small: ~2.5 MB, fast inference even on CPU."""
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    model   = models.mobilenet_v3_small(weights=weights)

    # Replace the final classifier layer
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model

# ── Training loop ─────────────────────────────────────────────────────────────

def train(model, train_loader, val_loader, epochs, lr, device, save_path):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc   = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        # ── train phase ──
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        t0 = time.time()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            _, pred = out.max(1)
            correct += pred.eq(labels).sum().item()
            total   += labels.size(0)

        train_loss = running_loss / total
        train_acc  = 100. * correct / total

        # ── val phase ──
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out  = model(imgs)
                loss = criterion(out, labels)
                v_loss   += loss.item() * imgs.size(0)
                _, pred   = out.max(1)
                v_correct += pred.eq(labels).sum().item()
                v_total   += labels.size(0)

        val_loss = v_loss / v_total
        val_acc  = 100. * v_correct / v_total
        elapsed  = time.time() - t0

        print(f"Epoch {epoch:3d}/{epochs} | "
              f"train loss {train_loss:.4f} acc {train_acc:.1f}% | "
              f"val loss {val_loss:.4f} acc {val_acc:.1f}% | "
              f"{elapsed:.1f}s")

        if val_acc > best_acc:
            best_acc   = val_acc
            best_state = copy.deepcopy(model.state_dict())
            print(f"  ↳ New best val acc: {best_acc:.1f}% — saving checkpoint")
            torch.save({
                "model_state": best_state,
                "class_to_idx": None,   # filled in main()
                "img_size": None,
                "arch": "mobilenet_v3_small",
            }, save_path + ".tmp")

        scheduler.step()

    print(f"\nTraining done. Best val acc: {best_acc:.1f}%")
    return best_state, best_acc

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_loader, val_loader, class_to_idx = build_loaders(
        args.data_dir, args.img_size, args.batch_size,
        args.val_split, args.workers, args.seed
    )
    print(f"Train samples: {len(train_loader.dataset)} | "
          f"Val samples: {len(val_loader.dataset)}")

    model = build_model(num_classes=2).to(device)

    best_state, best_acc = train(
        model, train_loader, val_loader,
        args.epochs, args.lr, device, args.save_path
    )

    # Save final checkpoint with metadata
    torch.save({
        "model_state": best_state,
        "class_to_idx": class_to_idx,   # e.g. {'left': 0, 'right': 1}
        "img_size": args.img_size,
        "arch": "mobilenet_v3_small",
        "best_val_acc": best_acc,
    }, args.save_path)

    # Clean up temp file
    if os.path.exists(args.save_path + ".tmp"):
        os.remove(args.save_path + ".tmp")

    print(f"Model saved → {args.save_path}")
    print(f"\nClass mapping: {class_to_idx}")
    print("Tip: pass this same class_to_idx to orient_ears.py via --left_label")

if __name__ == "__main__":
    main()