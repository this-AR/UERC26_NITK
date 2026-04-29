#!/usr/bin/env python3
"""
UERC26 DINOv2 + AdaFace Training Pipeline
==========================================

Trains DINOv2 ViT-S/14 with AdaFace loss for ear recognition embeddings.
AdaFace adapts the margin per image based on quality (better for blurry ears).

Usage:
    python train_pipeline.py --data_root UERC26_data/data/public --epochs 80

What this script does:
    1. Loads the public ear dataset with ear-side normalization + augmentations
    2. Fine-tunes DINOv2 ViT-S/14 backbone with a projection head
    3. Trains with AdaFace quality-adaptive margin loss (metric learning)
    4. Evaluates pairwise verification on val/test splits every N epochs
    5. Saves the best model weights
"""

import argparse
import os
import sys
import csv
import time
import random
import json
import numpy as np
from PIL import Image
import cv2
cv2.setNumThreads(0)  # VERY IMPORTANT: Prevents OpenCV from freezing/thrashing inside PyTorch DataLoader workers

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from ear_preprocessing import normalize_ear_side, apply_clahe, apply_sharpening

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# DATASETS
# ============================================================

class EarIdentificationDataset(Dataset):
    """
    Dataset for training with ArcFace loss.
    Returns (image_tensor, class_label) where class_label is the subject ID.
    
    Applies:
      - Ear side normalization (always)
      - CLAHE (random, p=0.5)
      - Sharpening (random, p=0.3)
      - Standard augmentations (rotation, color jitter, etc.)
    """
    
    def __init__(self, split, root_dir, split_csv='dataset_split.csv',
                 classes_mapping=None, is_train=True, preload_images=False):
        self.root_dir = root_dir
        self.split = split
        self.is_train = is_train
        self.preload_images = preload_images
        
        # Read split CSV to get images for this split
        self.images = []
        split_path = os.path.join(root_dir, split_csv)
        
        if not os.path.exists(split_path):
            raise FileNotFoundError(
                f"Split CSV not found: {split_path}\n"
                f"Run 'python uerc26_dataset.py' first to generate it."
            )
        
        with open(split_path, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if len(row) >= 2 and row[1] == split:
                    self.images.append(row[0])
        
        # Build class mapping from subjects
        subjects = sorted(set(os.path.dirname(img) for img in self.images))
        
        if classes_mapping is not None:
            self.classes_mapping = classes_mapping
        else:
            self.classes_mapping = {s: i for i, s in enumerate(subjects)}
        
        self.num_classes = len(self.classes_mapping)
        
        # Filter images to only include subjects in the mapping
        self.images = [img for img in self.images 
                       if os.path.dirname(img) in self.classes_mapping]
        
        # Build labels
        self.labels = [self.classes_mapping[os.path.dirname(img)] 
                       for img in self.images]
                       
        self.preloaded_images = {}
        if self.preload_images:
            print(f"  Preloading {len(self.images)} images to RAM for {self.split} split...")
            for img_name in tqdm(self.images, desc=f"Preloading {self.split}", leave=False):
                img_path = os.path.join(self.root_dir, img_name)
                image = cv2.imread(img_path)
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
                    self.preloaded_images[img_name] = image
                else:
                    print(f"Warning: Could not read {img_path} during preloading.")
        
        # Transforms
        if is_train:
            self.transform = transforms.Compose([
                *([] if self.preload_images else [transforms.Resize(256)]),
                transforms.RandomRotation(30),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, 
                                       saturation=0.4, hue=0.1),
                transforms.RandomGrayscale(p=0.2),  # Don't rely on skin color
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], 
                                     [0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                *([] if self.preload_images else [transforms.Resize(256)]),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], 
                                     [0.229, 0.224, 0.225]),
            ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        
        if self.preload_images and img_name in self.preloaded_images:
            image = self.preloaded_images[img_name]
        else:
            img_path = os.path.join(self.root_dir, img_name)
            # Load with OpenCV for preprocessing
            image = cv2.imread(img_path)
            if image is None:
                raise FileNotFoundError(f"Cannot read image: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # --- Ear preprocessing ---
        # 1. Always normalize ear side
        image = normalize_ear_side(image, target_side='right')
        
        # 2. Random CLAHE (training only)
        if self.is_train and random.random() < 0.5:
            image = apply_clahe(image)
        
        # 3. Random sharpening (training only)
        if self.is_train and random.random() < 0.3:
            image = apply_sharpening(image)
        
        # Convert to PIL for torchvision transforms
        image = Image.fromarray(image)
        image = self.transform(image)
        
        label = self.labels[idx]
        return image, label


class EarPairwiseDataset(Dataset):
    """
    Dataset for pairwise verification evaluation.
    Returns (img1_tensor, img2_tensor, label) where label=1 for same person.
    """
    
    def __init__(self, root_dir, pairs_csv):
        self.root_dir = root_dir
        
        self.pairs = []
        self.labels = []
        
        pairs_path = os.path.join(root_dir, pairs_csv)
        if not os.path.exists(pairs_path):
            raise FileNotFoundError(f"Pairs CSV not found: {pairs_path}")
        
        with open(pairs_path, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if len(row) >= 3:
                    self.pairs.append((row[0], row[1]))
                    self.labels.append(int(row[2]))
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225]),
        ])
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        img1_path = os.path.join(self.root_dir, self.pairs[idx][0])
        img2_path = os.path.join(self.root_dir, self.pairs[idx][1])
        
        # Load and preprocess both images
        img1 = cv2.imread(img1_path)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img1 = normalize_ear_side(img1, target_side='right')
        
        img2 = cv2.imread(img2_path)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        img2 = normalize_ear_side(img2, target_side='right')
        
        img1 = Image.fromarray(img1)
        img2 = Image.fromarray(img2)
        
        img1 = self.transform(img1)
        img2 = self.transform(img2)
        
        return img1, img2, self.labels[idx]


# ============================================================
# MODEL
# ============================================================

class DINOv2EarModel(nn.Module):
    """
    DINOv2 backbone + projection head for ear recognition.
    
    Supports two backbones:
      - 'vits14' → DINOv2 ViT-S/14 (21M params, 384-dim) — RECOMMENDED for Track 1
      - 'vitb14' → DINOv2 ViT-B/14 (86M params, 768-dim) — Use if ViT-S underperforms
    
    Architecture:
        Input (3, 224, 224) 
        → DINOv2 backbone → [CLS] token (384 or 768-dim)
        → Linear → BatchNorm → L2 normalize
        → embedding_dim-dimensional normalized embedding
    """
    
    # ┌──────────────────────────────────────────────────────────┐
    # │  BACKBONE OPTIONS:                                       │
    # │                                                          │
    # │  'vits14' = DINOv2 ViT-Small/14                          │
    # │             21M params, 384-dim output                   │
    # │             ★ RECOMMENDED — 4× smaller than baseline     │
    # │             Great Track 1 score (model size penalty)     │
    # │                                                          │
    # │  'vitb14' = DINOv2 ViT-Base/14                           │
    # │             86M params, 768-dim output                   │
    # │             Use this if ViT-S accuracy is too low        │
    # │             Similar size to baseline ConvNeXt-Base        │
    # └──────────────────────────────────────────────────────────┘
    
    BACKBONE_CONFIG = {
        'vits14': {'hub_name': 'dinov2_vits14', 'dim': 384, 'params': '21M'},
        'vitb14': {'hub_name': 'dinov2_vitb14', 'dim': 768, 'params': '86M'},
    }
    
    def __init__(self, embedding_dim=512, backbone='vits14', freeze_backbone=True):
        super().__init__()
        
        if backbone not in self.BACKBONE_CONFIG:
            raise ValueError(f"Unknown backbone '{backbone}'. Choose from: {list(self.BACKBONE_CONFIG.keys())}")
        
        config = self.BACKBONE_CONFIG[backbone]
        self.backbone_dim = config['dim']
        
        # Load DINOv2 backbone
        print(f"Loading DINOv2 {backbone} backbone ({config['params']} params, {config['dim']}-dim)...")
        self.backbone = torch.hub.load(
            'facebookresearch/dinov2', config['hub_name'], 
            pretrained=True
        )
        
        # Freeze backbone initially (Phase 1)
        if freeze_backbone:
            self._freeze_backbone()
        
        # Projection head: maps backbone_dim → embedding_dim
        self.projection = nn.Sequential(
            nn.Linear(self.backbone_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )
        
        self.embedding_dim = embedding_dim
        self.backbone_name = backbone
    
    def _freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("  Backbone FROZEN")
    
    def unfreeze_last_n_blocks(self, n):
        """Unfreeze the last N transformer blocks of the backbone."""
        # First freeze everything
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Then unfreeze last N blocks
        total_blocks = len(self.backbone.blocks)
        for i in range(total_blocks - n, total_blocks):
            for param in self.backbone.blocks[i].parameters():
                param.requires_grad = True
        
        # Also unfreeze the final norm layer
        if hasattr(self.backbone, 'norm'):
            for param in self.backbone.norm.parameters():
                param.requires_grad = True
        
        trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.backbone.parameters())
        print(f"  Backbone: unfroze last {n} blocks ({trainable:,}/{total:,} params trainable)")
    
    def unfreeze_all(self):
        """Unfreeze entire backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("  Backbone FULLY UNFROZEN")
    
    def forward(self, x, return_norms=False):
        """
        Forward pass. Returns L2-normalized embeddings.
        
        Args:
            x: (B, 3, 224, 224) image tensor
            return_norms: If True, also return feature norms (needed by AdaFace)
        
        Returns:
            If return_norms=False: (B, embedding_dim) L2-normalized embeddings
            If return_norms=True:  tuple of (embeddings, norms)
        """
        features = self.backbone(x)
        embeddings = self.projection(features)
        
        # Compute norms BEFORE L2 normalization (AdaFace uses this as quality proxy)
        norms = torch.norm(embeddings, p=2, dim=1)
        
        # L2 normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        if return_norms:
            return embeddings, norms
        return embeddings
    
    def get_model_stats(self):
        """Get model size statistics for scoring."""
        total_params = sum(p.numel() for p in self.parameters())
        model_size = sum(p.numel() * p.element_size() for p in self.parameters())
        return {
            'num_parameters': total_params,
            'model_size': model_size,
        }


class AdaFaceHead(nn.Module):
    """
    AdaFace (Quality-Adaptive Margin) loss head.
    
    Better than ArcFace for blurry/low-quality images because it ADAPTS
    the margin based on image quality (estimated from the feature norm).
    
    How it works:
      - High feature norm = clear image    = PUSH HARDER  (large margin)
      - Low feature norm  = blurry image   = GO EASIER    (small margin)
    
    This prevents overfitting to noise in low-quality ear images.
    
    Reference: Kim et al., "AdaFace", CVPR 2022
    
    Args:
        in_features: Embedding dimension (512)
        out_features: Number of classes/subjects
        s: Scale factor (64.0)
        m: Base angular margin (0.4)
        h: Quality sensitivity (0.333)
        t_alpha: EMA smoothing for norm statistics (0.01)
    """
    
    def __init__(self, in_features, out_features, s=64.0, m=0.4, h=0.333, t_alpha=0.01):
        super().__init__()
        self.s = s
        self.m = m
        self.h = h
        self.t_alpha = t_alpha
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        # Running stats of feature norms (for quality estimation)
        self.register_buffer('batch_mean', torch.ones(1) * 20.0)
        self.register_buffer('batch_std', torch.ones(1) * 100.0)
    
    def forward(self, embeddings, norms, labels):
        """
        Args:
            embeddings: (B, dim) L2-normalized embeddings
            norms: (B,) feature norms BEFORE L2 norm (quality indicator)
            labels: (B,) class labels
        Returns:
            AdaFace loss (scalar)
        """
        # Update running norm statistics (EMA)
        with torch.no_grad():
            self.batch_mean = (1 - self.t_alpha) * self.batch_mean + self.t_alpha * norms.mean()
            self.batch_std  = (1 - self.t_alpha) * self.batch_std  + self.t_alpha * (norms.std() + 1e-6)
        
        # Quality-adaptive margin scaler: [-1, 1]
        # High norm (clear image)  -> positive -> larger margin -> push harder
        # Low norm  (blurry image) -> negative -> smaller margin -> go easier
        margin_scaler = (norms - self.batch_mean) / (self.batch_std + 1e-3)
        margin_scaler = self.h * margin_scaler
        margin_scaler = torch.clamp(margin_scaler, -1.0, 1.0)
        
        # Cosine similarity
        cos_theta = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        cos_theta = cos_theta.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(cos_theta)
        
        one_hot = F.one_hot(labels, self.out_features).float()
        
        # Adaptive margins:
        #   g_angular = fixed angular margin (like ArcFace)
        #   g_add = quality-adaptive additive margin (AdaFace's key innovation)
        g_angular = self.m
        g_add = self.m * (1.0 + margin_scaler)  # per-sample adaptive margin
        
        # For target class: cos(theta + angular) - adaptive_additive
        target_logits = torch.cos(theta + g_angular) - g_add.unsqueeze(1)
        
        output = cos_theta * (1.0 - one_hot) + target_logits * one_hot
        output = output * self.s
        
        return F.cross_entropy(output, labels)


# ---- ArcFace fallback (uncomment to use instead of AdaFace) ----
# class ArcFaceHead(nn.Module):
#     """Fixed margin. Simpler but worse for blurry data."""
#     def __init__(self, in_features, out_features, s=64.0, m=0.5):
#         super().__init__()
#         self.s, self.m, self.out_features = s, m, out_features
#         self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
#         nn.init.xavier_uniform_(self.weight)
#     def forward(self, embeddings, norms, labels):  # norms ignored
#         cos_theta = F.linear(F.normalize(embeddings), F.normalize(self.weight))
#         cos_theta = cos_theta.clamp(-1+1e-7, 1-1e-7)
#         theta = torch.acos(cos_theta)
#         one_hot = F.one_hot(labels, self.out_features).float()
#         output = (cos_theta*(1-one_hot) + torch.cos(theta+self.m)*one_hot) * self.s
#         return F.cross_entropy(output, labels)


# ============================================================
# EVALUATION
# ============================================================

def evaluate_verification(model, dataloader, device):
    """
    Evaluate pairwise verification performance.
    
    For each pair:
        1. Extract embeddings for both images
        2. Compute cosine similarity
        3. Compare against ground truth labels
    
    Returns metrics: VER@0.1%FAR, VER@1%FAR, EER, AUC
    """
    model.eval()
    
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for img1, img2, labels in tqdm(dataloader, desc="  Verif eval", leave=False):
            img1 = img1.to(device)
            img2 = img2.to(device)
            
            emb1 = model(img1)
            emb2 = model(img2)
            
            # Cosine similarity (embeddings are already L2-normalized)
            sim = F.cosine_similarity(emb1, emb2)
            
            all_scores.extend(sim.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    # Import metrics computation from the existing utils
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from uerc26_utils import compute_metrics
    
    metrics = compute_metrics(all_labels, all_scores)
    
    return metrics


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(model, loss_head, optimizer, dataloader, device, epoch, scaler=None):
    """Train for one epoch with optional AMP."""
    model.train()
    loss_head.train()
    
    total_loss = 0.0
    num_batches = 0
    use_amp = scaler is not None
    
    optimizer.zero_grad()
    pbar = tqdm(dataloader, desc=f"  Epoch {epoch}", leave=False)
    for step, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Forward with AMP (mixed precision for ~30-40% speedup)
        with torch.cuda.amp.autocast(enabled=use_amp):
            embeddings, norms = model(images, return_norms=True)
            loss = loss_head(embeddings, norms, labels)
        
        # Backward with GradScaler
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(loss_head.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(loss_head.parameters(), max_norm=1.0)
            optimizer.step()
            
        optimizer.zero_grad()
        
        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description='UERC26 DINOv2 + AdaFace Training')
    
    # Data
    parser.add_argument('--data_root', type=str, default='UERC26_data/data/public',
                        help='Path to public dataset root')
    parser.add_argument('--split_csv', type=str, default='dataset_split.csv')
    
    # Model
    parser.add_argument('--embedding_dim', type=int, default=512,
                        help='Embedding dimension (default: 512)')
    parser.add_argument('--backbone', type=str, default='vits14',
                        choices=['vits14', 'vitb14'],
                        help='DINOv2 backbone: vits14 (21M, recommended) or vitb14 (86M, if vits14 underperforms)')
    
    # Preloading
    parser.add_argument('--preload_images', action='store_true',
                        help='Preload all images to RAM for faster training')
    
    # Training
    parser.add_argument('--epochs', type=int, default=80,
                        help='Total training epochs (default: 80)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size (default: 64, reduce if OOM)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate for projection head (default: 1e-3)')
    parser.add_argument('--backbone_lr', type=float, default=1e-5,
                        help='Learning rate for backbone when unfrozen (default: 1e-5)')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--warmup_epochs', type=int, default=3,
                        help='Linear LR warmup epochs (default: 3)')
    parser.add_argument('--no_amp', action='store_true',
                        help='Disable mixed precision training')
    
    # AdaFace
    parser.add_argument('--adaface_s', type=float, default=64.0,
                        help='AdaFace scale factor')
    parser.add_argument('--adaface_m', type=float, default=0.4,
                        help='AdaFace base angular margin (0.4 recommended)')
    parser.add_argument('--adaface_h', type=float, default=0.333,
                        help='AdaFace quality sensitivity (0.333 recommended)')
    
    # Phased unfreezing schedule
    parser.add_argument('--phase1_epochs', type=int, default=10,
                        help='Epochs with frozen backbone (default: 10)')
    parser.add_argument('--phase2_epochs', type=int, default=40,
                        help='Epochs with last 4 blocks unfrozen (default: 40)')
    # Phase 3 = remaining epochs with full backbone unfrozen
    
    # Evaluation
    parser.add_argument('--eval_every', type=int, default=5,
                        help='Evaluate verification every N epochs')
    
    # Output
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save model checkpoints')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print("cuDNN benchmark: ENABLED")
    print(f"\n{'='*60}")
    print(f"UERC26 DINOv2 + AdaFace Training Pipeline")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Data root: {args.data_root}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Embedding dim: {args.embedding_dim}")
    print(f"Phases: {args.phase1_epochs} frozen → {args.phase2_epochs} partial → rest full")
    
    # Multi-GPU detection
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_gpus > 1:
        print(f"Multi-GPU: {num_gpus} GPUs detected! Using DataParallel for ~{num_gpus}x speedup.")
        for i in range(num_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # ---- Create datasets ----
    print(f"\n--- Loading datasets ---")
    
    train_dataset = EarIdentificationDataset(
        split='train', root_dir=args.data_root,
        split_csv=args.split_csv, is_train=True,
        preload_images=args.preload_images
    )
    print(f"Train: {len(train_dataset)} images, {train_dataset.num_classes} subjects")
    
    val_dataset = EarIdentificationDataset(
        split='val', root_dir=args.data_root,
        split_csv=args.split_csv,
        classes_mapping=train_dataset.classes_mapping,
        is_train=False, preload_images=args.preload_images
    )
    print(f"Val:   {len(val_dataset)} images")
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=True, prefetch_factor=3
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=True, prefetch_factor=3
    )
    
    # Pairwise datasets for verification evaluation
    val_pairs_csv = 'pairs_val.csv'
    test_pairs_csv = 'pairs_test.csv'
    
    val_pair_dataset = None
    test_pair_dataset = None
    
    if os.path.exists(os.path.join(args.data_root, val_pairs_csv)):
        val_pair_dataset = EarPairwiseDataset(args.data_root, val_pairs_csv)
        print(f"Val pairs:  {len(val_pair_dataset)} pairs")
    
    if os.path.exists(os.path.join(args.data_root, test_pairs_csv)):
        test_pair_dataset = EarPairwiseDataset(args.data_root, test_pairs_csv)
        print(f"Test pairs: {len(test_pair_dataset)} pairs")
    
    # ---- Create model ----
    print(f"\n--- Creating model ---")
    
    model = DINOv2EarModel(
        embedding_dim=args.embedding_dim,
        backbone=args.backbone,       # 'vits14' (21M) or 'vitb14' (86M)
        freeze_backbone=True           # Start with frozen backbone (Phase 1)
    )
    model = model.to(device)
    
    stats = model.get_model_stats()
    print(f"Model params: {stats['num_parameters']:,}")
    print(f"Model size:   {stats['model_size'] / 1e6:.1f} MB")
    
    loss_head = AdaFaceHead(
        in_features=args.embedding_dim,
        out_features=train_dataset.num_classes,
        s=args.adaface_s,
        m=args.adaface_m,
        h=args.adaface_h
    )
    loss_head = loss_head.to(device)
    
    # Wrap in DataParallel if multiple GPUs available
    raw_model = model
    if num_gpus > 1:
        model = nn.DataParallel(model)
        print(f"Model wrapped in DataParallel ({num_gpus} GPUs)")
    
    # Mixed Precision Training setup
    use_amp = torch.cuda.is_available() and not args.no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None
    if use_amp:
        print(f"Mixed Precision (AMP): ENABLED — ~30-40% faster, less GPU memory")
    else:
        print(f"Mixed Precision (AMP): disabled")
    
    # ---- Training loop ----
    os.makedirs(args.save_dir, exist_ok=True)
    best_auc = 0.0
    best_epoch = 0
    
    for epoch in range(1, args.epochs + 1):
        if epoch == 1:
            import time
            data_times, gpu_times = [], []
            end = time.time()
            for i, (imgs, lbls) in enumerate(train_loader):
                data_times.append(time.time() - end)
                t = time.time()
                imgs = imgs.to(device, non_blocking=True)
                lbls = lbls.to(device, non_blocking=True)
                gpu_times.append(time.time() - t)
                end = time.time()
                if i >= 19: break
            print(f"  [TIMING] Avg data load: {sum(data_times)/len(data_times)*1000:.1f}ms | "
                  f"Avg GPU transfer: {sum(gpu_times)/len(gpu_times)*1000:.1f}ms")
            print(f"  [TIMING] Target: data load << 50ms per batch")
            
        # === PHASED UNFREEZING ===
        # Use raw_model for unfreezing (DataParallel wraps the model)
        if epoch == 1:
            print(f"\n>>> PHASE 1: Frozen backbone (epochs 1-{args.phase1_epochs})")
            optimizer = torch.optim.AdamW([
                {'params': raw_model.projection.parameters(), 'lr': args.lr},
                {'params': loss_head.parameters(), 'lr': args.lr},
            ], weight_decay=args.weight_decay)
            # Warmup + CosineAnnealing: ramp LR from 1/10th → full over warmup_epochs
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_epochs
            )
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(args.phase1_epochs - args.warmup_epochs, 1), eta_min=1e-6
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[args.warmup_epochs]
            )
        
        elif epoch == args.phase1_epochs + 1:
            print(f"\n>>> PHASE 2: Last 4 blocks unfrozen (epochs {epoch}-{args.phase1_epochs + args.phase2_epochs})")
            raw_model.unfreeze_last_n_blocks(4)
            optimizer = torch.optim.AdamW([
                {'params': [p for p in raw_model.backbone.parameters() if p.requires_grad], 
                 'lr': args.backbone_lr},
                {'params': raw_model.projection.parameters(), 'lr': args.lr * 0.1},
                {'params': loss_head.parameters(), 'lr': args.lr * 0.1},
            ], weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.phase2_epochs, eta_min=1e-7
            )
        
        elif epoch == args.phase1_epochs + args.phase2_epochs + 1:
            remaining = args.epochs - epoch + 1
            print(f"\n>>> PHASE 3: Full model unfrozen (epochs {epoch}-{args.epochs})")
            raw_model.unfreeze_all()
            optimizer = torch.optim.AdamW([
                {'params': raw_model.backbone.parameters(), 'lr': args.backbone_lr * 0.1},
                {'params': raw_model.projection.parameters(), 'lr': args.lr * 0.01},
                {'params': loss_head.parameters(), 'lr': args.lr * 0.01},
            ], weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(remaining, 1), eta_min=1e-8
            )
        
        # Train one epoch (with AMP)
        avg_loss = train_one_epoch(
            model, loss_head, optimizer, train_loader, device, epoch,
            scaler=scaler
        )
        scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:3d}/{args.epochs} | Loss: {avg_loss:.4f} | LR: {current_lr:.2e}")
        
        # === EVALUATION ===
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            print(f"\n  --- Evaluation at epoch {epoch} ---")
            
            # Validation verification
            if val_pair_dataset is not None:
                val_pair_loader = DataLoader(
                    val_pair_dataset, batch_size=args.batch_size,
                    shuffle=False, num_workers=args.num_workers,
                    persistent_workers=True, prefetch_factor=3
                )
                val_metrics = evaluate_verification(model, val_pair_loader, device)
                print(f"  VAL  | VER@0.1%: {val_metrics['VER@0.1%']*100:.2f}% | "
                      f"VER@1%: {val_metrics['VER@1%']*100:.2f}% | "
                      f"EER: {val_metrics['EER']*100:.2f}% | "
                      f"AUC: {val_metrics['AUC']*100:.2f}%")
            
            # Test verification
            if test_pair_dataset is not None:
                test_pair_loader = DataLoader(
                    test_pair_dataset, batch_size=args.batch_size,
                    shuffle=False, num_workers=args.num_workers,
                    persistent_workers=True, prefetch_factor=3
                )
                test_metrics = evaluate_verification(model, test_pair_loader, device)
                print(f"  TEST | VER@0.1%: {test_metrics['VER@0.1%']*100:.2f}% | "
                      f"VER@1%: {test_metrics['VER@1%']*100:.2f}% | "
                      f"EER: {test_metrics['EER']*100:.2f}% | "
                      f"AUC: {test_metrics['AUC']*100:.2f}%")
                
                # Save best model based on AUC
                current_auc = test_metrics['AUC']
                if current_auc > best_auc:
                    best_auc = current_auc
                    best_epoch = epoch
                    save_path = os.path.join(args.save_dir, 'best_model.pt')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': raw_model.state_dict(),
                        'loss_head_state_dict': loss_head.state_dict(),
                        'metrics': test_metrics,
                        'model_stats': raw_model.get_model_stats(),
                        'embedding_dim': args.embedding_dim,
                        'backbone': args.backbone,
                    }, save_path)
                    print(f"  \u2605 NEW BEST! AUC={best_auc*100:.2f}% saved to {save_path}")
            
            print()
        
        # Save periodic checkpoint
        if epoch % 20 == 0:
            ckpt_path = os.path.join(args.save_dir, f'checkpoint_epoch{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': raw_model.state_dict(),
                'loss_head_state_dict': loss_head.state_dict(),
                'embedding_dim': args.embedding_dim,
                'backbone': args.backbone,
            }, ckpt_path)
    
    # ---- Final Summary ----
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"{'='*60}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best AUC:   {best_auc*100:.2f}%")
    print(f"Model saved to: {os.path.join(args.save_dir, 'best_model.pt')}")
    print(f"\nNext steps:")
    print(f"  python generate_embeddings.py --model_path {os.path.join(args.save_dir, 'best_model.pt')}")


if __name__ == '__main__':
    main()
