"""Model registry for the matching evaluation. Pick one with the DT2_MODEL env var.

- tinyvit (default): ITW/ITW_230_split_augment/best_pruned_35per_tinyvit.pt, 7.53 M params.
  Input: BGR->RGB, resize 224, /255 (no mean/std), as in the earlier evaluations.
- swinv2: best_model.pt (torchvision swin_v2_b + Linear/BN projection, 87.5 M params, saved with
  torch.compile). Input as in uerc26_Model_training_(single_phase).ipynb validation: resize 224, RGB,
  ImageNet mean/std; the model upsamples to 256 itself. DT2_CLAHE=1 adds the notebook's LAB CLAHE.
"""
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import MODEL_DIR, PROJECT_DIR, WEIGHTS_PATH

NAME = os.environ.get('DT2_MODEL', 'tinyvit')
CLAHE = os.environ.get('DT2_CLAHE', '0') == '1'
IMG_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


class SwinV2EarModel(nn.Module):
    # Same layout as the training notebook's SwinV2EarModel.
    def __init__(self, embedding_dim=512):
        super().__init__()
        from torchvision import models
        self.backbone = models.swin_v2_b(weights=None)
        self.backbone.head = nn.Identity()
        self.projection = nn.Sequential(nn.Linear(1024, embedding_dim), nn.BatchNorm1d(embedding_dim))

    def forward(self, x):
        if x.shape[-1] != 256 or x.shape[-2] != 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        return F.normalize(self.projection(self.backbone(x)), p=2, dim=1)


def apply_clahe(rgb):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def tag():
    return {'tinyvit': '', 'swinv2': '_swinv2'}[NAME] + ('_clahe' if CLAHE else '')


def build():
    """Returns (model on CUDA in eval mode, weights path, parameter count)."""
    if NAME == 'tinyvit':
        sys.path.insert(0, str(MODEL_DIR))
        from model import PrunedTinyViTWrapper
        model = PrunedTinyViTWrapper().model
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location='cpu'), strict=False)
        weights = WEIGHTS_PATH
    elif NAME == 'swinv2':
        weights = PROJECT_DIR / 'best_model.pt'
        ckpt = torch.load(weights, map_location='cpu', weights_only=True)
        state = {k.removeprefix('_orig_mod.'): v for k, v in ckpt['model_state_dict'].items()}
        model = SwinV2EarModel(ckpt.get('embedding_dim', 512))
        model.load_state_dict(state, strict=True)
    else:
        raise ValueError(f'Unknown DT2_MODEL: {NAME}')
    model = model.to('cuda').eval()
    return model, weights, sum(p.numel() for p in model.parameters())


def load_image(path, flip=False, letterbox=False):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f'Cannot read image: {path}')
    if flip:
        image = cv2.flip(image, 1)
    if letterbox:
        h, w = image.shape[:2]
        side = max(h, w)
        canvas = np.empty((side, side, 3), dtype=image.dtype)
        canvas[:] = image.reshape(-1, 3).mean(axis=0)
        canvas[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = image
        image = canvas
    image = np.ascontiguousarray(cv2.resize(image, (IMG_SIZE, IMG_SIZE))[:, :, ::-1])
    if NAME == 'swinv2' and CLAHE:
        image = apply_clahe(image)
    image = image.astype(np.float32) / 255.0
    if NAME == 'swinv2':
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(image, (2, 0, 1))
