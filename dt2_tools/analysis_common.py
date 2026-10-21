"""Shared helpers for the t-SNE, UBEAR failure and model-comparison analyses.

Embeddings: model -> L2 -> subtract global mean -> re-L2 (same as the main evaluation).
The model comes from models.py (DT2_MODEL / DT2_CLAHE env vars).
Optional image transforms let the failure analysis test mirroring and aspect-ratio handling.
"""
import numpy as np
import torch

import models
from datasets import PROJECT_DIR, discover

ANALYSIS_DIR = PROJECT_DIR / 'analysis'
CACHE_DIR = ANALYSIS_DIR / 'embeddings'
DEVICE = torch.device('cuda')
_model = None


def model():
    global _model
    if _model is None:
        _model = models.build()[0]
    return _model


def embed(paths, flip=None, letterbox=False, batch=64):
    """Raw embeddings. `flip` is an optional boolean mask of images to mirror."""
    flip = np.zeros(len(paths), bool) if flip is None else flip
    out = []
    with torch.no_grad():
        for s in range(0, len(paths), batch):
            x = np.stack([models.load_image(p, f, letterbox) for p, f in zip(paths[s:s + batch], flip[s:s + batch])])
            out.append(model()(torch.from_numpy(x).to(DEVICE)).float().cpu())
    return torch.cat(out).numpy().astype(np.float64)


def mean_centre(raw):
    l2 = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    c = l2 - l2.mean(axis=0)
    return c / np.linalg.norm(c, axis=1, keepdims=True)


def dataset_embeddings(key, centred=True):
    """(subjects, paths, embeddings), cached in analysis/embeddings. Mean-centred unless centred=False."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f'{key}{models.tag()}.npz'
    records = discover(key)
    subjects = np.array([s for s, _ in records])
    paths = [p for _, p in records]
    if cache.exists():
        raw = np.load(cache)['raw']
    else:
        raw = embed(paths)
        np.savez(cache, raw=raw.astype(np.float32))
    raw = raw.astype(np.float64)
    return subjects, paths, mean_centre(raw) if centred else raw / np.linalg.norm(raw, axis=1, keepdims=True)
