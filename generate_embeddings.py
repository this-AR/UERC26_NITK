#!/usr/bin/env python3
"""
UERC26 Embedding Generation & Evaluation
==========================================

Loads a trained DINOv2 model and:
  1. Extracts embeddings for ALL images in the public dataset (train/val/test)
  2. Extracts embeddings for the sequestered_anonymized dataset
  3. Evaluates pairwise verification on public val/test pairs
  4. Saves sequestered features in the format expected by evaluate_features.py

Usage:
    # After training:
    python generate_embeddings.py \\
        --model_path checkpoints/best_model.pt \\
        --public_root UERC26_data/data/public \\
        --sequestered_root UERC26_data/data/sequestered_anonymized \\
        --output_dir SUBMISSIONS/dinov2_solution

    # Then run the official evaluation:
    python evaluate_features.py
"""

import argparse
import os
import sys
import csv
import json
import numpy as np
from PIL import Image
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from ear_preprocessing import normalize_ear_side, apply_clahe


# ============================================================
# MODEL (must match train_pipeline.py)
# ============================================================

class DINOv2EarModel(nn.Module):
    """DINOv2 backbone + projection head — same as in train_pipeline.py"""
    
    # ViT-S/14 (21M params, 384-dim) = RECOMMENDED
    # ViT-B/14 (86M params, 768-dim) = fallback if ViT-S underperforms
    BACKBONE_CONFIG = {
        'vits14': {'hub_name': 'dinov2_vits14', 'dim': 384, 'params': '21M'},
        'vitb14': {'hub_name': 'dinov2_vitb14', 'dim': 768, 'params': '86M'},
    }
    
    def __init__(self, embedding_dim=512, backbone='vits14'):
        super().__init__()
        config = self.BACKBONE_CONFIG[backbone]
        self.backbone_dim = config['dim']
        self.backbone = torch.hub.load(
            'facebookresearch/dinov2', config['hub_name'], pretrained=True
        )
        self.projection = nn.Sequential(
            nn.Linear(self.backbone_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )
        self.embedding_dim = embedding_dim
        self.backbone_name = backbone
    
    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.projection(features)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings
    
    def get_model_stats(self):
        total_params = sum(p.numel() for p in self.parameters())
        model_size = sum(p.numel() * p.element_size() for p in self.parameters())
        return {'num_parameters': total_params, 'model_size': model_size}


# ============================================================
# DATASETS FOR EMBEDDING EXTRACTION
# ============================================================

class EarImageDataset(Dataset):
    """
    Simple dataset that loads images and returns (image_tensor, image_name).
    Used for extracting embeddings from all images in a directory.
    """
    
    def __init__(self, root_dir, split_csv='dataset_split.csv', split=None):
        """
        Args:
            root_dir: Path to data root (e.g., UERC26_data/data/public)
            split_csv: Path to split CSV (relative to root_dir)
            split: If provided, only load images from this split ('train', 'val', 'test')
                   If None, load ALL images from the CSV
        """
        self.root_dir = root_dir
        self.images = []
        
        split_path = os.path.join(root_dir, split_csv)
        
        if os.path.exists(split_path):
            with open(split_path, 'r') as f:
                reader = csv.reader(f, delimiter='\t')
                for row in reader:
                    if len(row) >= 2:
                        if split is None or row[1] == split:
                            self.images.append(row[0])
        else:
            # Fallback: scan directory for images
            print(f"  No split CSV found, scanning {root_dir} for images...")
            for dirpath, _, filenames in os.walk(root_dir):
                for fname in sorted(filenames):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                        rel_path = os.path.relpath(os.path.join(dirpath, fname), root_dir)
                        self.images.append(rel_path.replace('\\', '/'))
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.root_dir, img_name)
        
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Preprocess: normalize ear side + CLAHE for consistent embeddings
        image = normalize_ear_side(image, target_side='right')
        image = apply_clahe(image)  # Always apply CLAHE at inference for consistency
        
        image = Image.fromarray(image)
        image = self.transform(image)
        
        return image, img_name


class EarImageDatasetTTA(Dataset):
    """
    Dataset with Test-Time Augmentation (TTA).
    For each image, returns both the original and horizontally flipped version.
    The caller should average the embeddings from both for better accuracy.
    """
    
    def __init__(self, root_dir, split_csv='dataset_split.csv', split=None):
        self.root_dir = root_dir
        self.images = []
        
        split_path = os.path.join(root_dir, split_csv)
        
        if os.path.exists(split_path):
            with open(split_path, 'r') as f:
                reader = csv.reader(f, delimiter='\t')
                for row in reader:
                    if len(row) >= 2:
                        if split is None or row[1] == split:
                            self.images.append(row[0])
        else:
            for dirpath, _, filenames in os.walk(root_dir):
                for fname in sorted(filenames):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                        rel_path = os.path.relpath(os.path.join(dirpath, fname), root_dir)
                        self.images.append(rel_path.replace('\\', '/'))
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.root_dir, img_name)
        
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Preprocess
        image = normalize_ear_side(image, target_side='right')
        image = apply_clahe(image)
        
        # Original
        img_orig = Image.fromarray(image)
        img_orig = self.transform(img_orig)
        
        # Flipped (horizontal)
        img_flip = Image.fromarray(cv2.flip(image, 1))
        img_flip = self.transform(img_flip)
        
        return img_orig, img_flip, img_name


# ============================================================
# EMBEDDING EXTRACTION
# ============================================================

def extract_embeddings(model, dataloader, device, use_tta=False):
    """
    Extract embeddings for all images in a dataloader.
    
    Args:
        model: Trained DINOv2EarModel
        dataloader: DataLoader returning (image, image_name) or (orig, flip, name) for TTA
        device: torch device
        use_tta: Whether dataloader returns TTA pairs
    
    Returns:
        dict: {image_name: numpy_embedding}
    """
    model.eval()
    features = {}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="  Extracting", leave=False):
            if use_tta:
                img_orig, img_flip, img_names = batch
                
                # Embed both
                emb_orig = model(img_orig.to(device))
                emb_flip = model(img_flip.to(device))
                
                # Average and re-normalize
                emb = F.normalize((emb_orig + emb_flip) / 2.0, p=2, dim=1)
            else:
                images, img_names = batch
                emb = model(images.to(device))
            
            # Store
            emb_np = emb.cpu().numpy()
            for name, feat in zip(img_names, emb_np):
                features[name] = feat
    
    return features


def evaluate_pairwise(features, pairs_csv_path):
    """
    Evaluate pairwise verification using pre-extracted features.
    
    Args:
        features: dict {image_name: numpy_embedding}
        pairs_csv_path: Path to pairs CSV (img1, img2, label)
    
    Returns:
        metrics dict
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from uerc26_utils import compute_metrics
    
    scores = []
    labels = []
    missing = 0
    
    with open(pairs_csv_path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) >= 3:
                img1, img2, label = row[0], row[1], int(row[2])
                
                if img1 in features and img2 in features:
                    feat1 = features[img1]
                    feat2 = features[img2]
                    
                    # Cosine similarity
                    sim = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2) + 1e-8)
                    scores.append(sim)
                    labels.append(label)
                else:
                    missing += 1
    
    if missing > 0:
        print(f"  Warning: {missing} pairs had missing features (skipped)")
    
    metrics = compute_metrics(np.array(labels), np.array(scores))
    return metrics


def main():
    parser = argparse.ArgumentParser(description='UERC26 Embedding Generation & Evaluation')
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint (e.g., checkpoints/best_model.pt)')
    parser.add_argument('--public_root', type=str, default='UERC26_data/data/public',
                        help='Path to public dataset')
    parser.add_argument('--sequestered_root', type=str, default='UERC26_data/data/sequestered_anonymized',
                        help='Path to sequestered anonymized dataset')
    parser.add_argument('--output_dir', type=str, default='SUBMISSIONS/dinov2_solution',
                        help='Output directory for submission files')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--use_tta', action='store_true',
                        help='Use Test-Time Augmentation (average orig + flipped embeddings)')
    parser.add_argument('--embedding_dim', type=int, default=512)
    parser.add_argument('--backbone', type=str, default=None,
                        help='DINOv2 backbone (auto-detected from checkpoint if not set)')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*60}")
    print(f"UERC26 Embedding Generation & Evaluation")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Model:  {args.model_path}")
    print(f"TTA:    {'ON' if args.use_tta else 'OFF'}")
    
    # ---- Load model ----
    print(f"\n--- Loading model ---")
    
    # Determine settings from checkpoint
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'embedding_dim' in checkpoint:
        args.embedding_dim = checkpoint['embedding_dim']
    
    # Auto-detect backbone from checkpoint, or use command line arg
    backbone = args.backbone
    if backbone is None:
        backbone = checkpoint.get('backbone', 'vits14')  # default to vits14
    print(f"Backbone: {backbone}")
    
    model = DINOv2EarModel(embedding_dim=args.embedding_dim, backbone=backbone)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    stats = model.get_model_stats()
    print(f"Model params:     {stats['num_parameters']:,}")
    print(f"Model size:       {stats['model_size'] / 1e6:.1f} MB")
    print(f"Embedding dim:    {args.embedding_dim}")
    
    if 'metrics' in checkpoint:
        m = checkpoint['metrics']
        print(f"Training metrics: AUC={m.get('AUC', 0)*100:.2f}%, "
              f"VER@0.1%={m.get('VER@0.1%', 0)*100:.2f}%")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ---- Extract embeddings for each split of public data ----
    print(f"\n--- Extracting embeddings from PUBLIC dataset ---")
    
    all_features = {}
    
    for split in ['train', 'val', 'test']:
        print(f"\n  [{split.upper()}]")
        
        if args.use_tta:
            dataset = EarImageDatasetTTA(args.public_root, split=split)
        else:
            dataset = EarImageDataset(args.public_root, split=split)
        
        print(f"  Images: {len(dataset)}")
        
        if len(dataset) == 0:
            print(f"  Skipping (no images)")
            continue
        
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True
        )
        
        features = extract_embeddings(model, loader, device, use_tta=args.use_tta)
        all_features.update(features)
        
        # Save per-split features
        split_path = os.path.join(args.output_dir, f'public_{split}_features.npy')
        np.save(split_path, features)
        print(f"  Saved {len(features)} embeddings → {split_path}")
        
        # Show example
        first_key = next(iter(features))
        print(f"  Example: '{first_key}' → shape {features[first_key].shape}")
    
    # ---- Evaluate on public pairs ----
    print(f"\n--- Evaluating on PUBLIC dataset pairs ---")
    
    for split_name, pairs_csv in [('val', 'pairs_val.csv'), ('test', 'pairs_test.csv')]:
        pairs_path = os.path.join(args.public_root, pairs_csv)
        
        if os.path.exists(pairs_path):
            metrics = evaluate_pairwise(all_features, pairs_path)
            
            print(f"\n  {split_name.upper()} Verification Results:")
            print(f"    VER@0.1%FAR: {metrics['VER@0.1%']*100:.2f}%")
            print(f"    VER@1%FAR:   {metrics['VER@1%']*100:.2f}%")
            print(f"    EER:         {metrics['EER']*100:.2f}%")
            print(f"    AUC:         {metrics['AUC']*100:.2f}%")
            
            # Save metrics
            metrics_path = os.path.join(args.output_dir, f'public_{split_name}_metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            print(f"    Metrics saved → {metrics_path}")
        else:
            print(f"  {pairs_csv} not found, skipping {split_name} evaluation")
    
    # ---- Extract embeddings for SEQUESTERED dataset ----
    print(f"\n--- Extracting embeddings from SEQUESTERED dataset ---")
    
    if os.path.exists(args.sequestered_root):
        if args.use_tta:
            seq_dataset = EarImageDatasetTTA(args.sequestered_root)
        else:
            seq_dataset = EarImageDataset(args.sequestered_root)
        
        print(f"  Images: {len(seq_dataset)}")
        
        seq_loader = DataLoader(
            seq_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True
        )
        
        seq_features = extract_embeddings(model, seq_loader, device, use_tta=args.use_tta)
        
        # Save in the format expected by evaluate_features.py
        # Format: dict {image_name: numpy_array}
        seq_features_path = os.path.join(args.output_dir, 'sequestered_features.npy')
        np.save(seq_features_path, seq_features)
        print(f"  Saved {len(seq_features)} embeddings → {seq_features_path}")
        
        first_key = next(iter(seq_features))
        print(f"  Example: '{first_key}' → shape {seq_features[first_key].shape}")
    else:
        print(f"  Sequestered data not found at {args.sequestered_root}")
    
    # ---- Save model stats for scoring ----
    model_stats_path = os.path.join(args.output_dir, 'model_stats.json')
    with open(model_stats_path, 'w') as f:
        json.dump(stats, f, indent=4)
    print(f"\nModel stats saved → {model_stats_path}")
    
    # ---- Create config.ini for the submission ----
    config_path = os.path.join(args.output_dir, 'config.ini')
    if not os.path.exists(config_path):
        with open(config_path, 'w') as f:
            f.write("[SUBMISSION]\n")
            f.write("track = T1\n")
            f.write("name = dinov2_arcface_solution\n")
            f.write("\n")
            f.write("[MODEL]\n")
            f.write("model_class = DINOv2EarModel\n")
            f.write(f"weights = best_model.pt\n")
            f.write("\n")
            f.write("[DATA]\n")
            f.write("data_path = ../../UERC26_data/data/public\n")
            f.write("data_split = test\n")
        print(f"Config created → {config_path}")
    
    # ---- Copy model weights to submission directory ----
    import shutil
    dst_model = os.path.join(args.output_dir, 'best_model.pt')
    if not os.path.exists(dst_model):
        shutil.copy2(args.model_path, dst_model)
        print(f"Model weights copied → {dst_model}")
    
    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"Done! Submission ready at: {args.output_dir}")
    print(f"{'='*60}")
    print(f"\nFiles created:")
    for f in sorted(os.listdir(args.output_dir)):
        fpath = os.path.join(args.output_dir, f)
        size = os.path.getsize(fpath) / 1e6
        print(f"  {f:40s} ({size:.1f} MB)")
    
    print(f"\nComparison with baseline (ConvNeXt-Base):")
    print(f"  Baseline: 87.5M params, 350 MB, VER@0.1%=26.23%, AUC=92.15%")
    print(f"  Yours:    {stats['num_parameters']/1e6:.1f}M params, {stats['model_size']/1e6:.0f} MB")
    
    print(f"\nTo run official evaluation on sequestered set:")
    print(f"  python evaluate_features.py")
    print(f"  python score.py")


if __name__ == '__main__':
    main()
