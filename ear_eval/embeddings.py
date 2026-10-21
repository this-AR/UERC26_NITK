"""Embedding extraction, normalisation and the on-disk cache.

Main pipeline variant: model -> L2-normalize -> subtract global mean -> re-L2-normalize.
"""
import numpy as np
import torch

from . import models
from .config import CACHE_DIR, model_tag
from .datasets import discover

_model = None


def model():
    global _model
    if _model is None:
        _model = models.build()[0]
    return _model


def embed(paths, flip=None, letterbox=False, batch=64, progress=False):
    """Raw embeddings (float64). `flip` is an optional boolean mask of images to mirror."""
    flip = np.zeros(len(paths), bool) if flip is None else flip
    out = []
    with torch.no_grad():
        for s in range(0, len(paths), batch):
            x = np.stack([models.load_image(p, f, letterbox) for p, f in zip(paths[s:s + batch], flip[s:s + batch])])
            out.append(model()(torch.from_numpy(x).to('cuda')).float().cpu())
            if progress:
                print(f'  [{min(s + batch, len(paths))}/{len(paths)}] embeddings computed', flush=True)
    return torch.cat(out).numpy().astype(np.float64)


def l2(raw):
    return raw / np.linalg.norm(raw, axis=1, keepdims=True)


def mean_centre(raw):
    """L2 -> subtract global mean -> L2. Returns (embeddings, global mean)."""
    x = l2(raw)
    mean = x.mean(axis=0)
    centred = x - mean
    norms = np.linalg.norm(centred, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError('A centred embedding has zero norm.')
    return centred / norms, mean


def dataset_embeddings(key, centred=True):
    """(subjects, paths, embeddings) for one dataset, raw embeddings cached in .cache/embeddings."""
    records = discover(key)
    subjects = np.array([s for s, _ in records])
    paths = [p for _, p in records]
    cache = CACHE_DIR / f'{key}_{model_tag()}.npz'
    if cache.exists():
        raw = np.load(cache)['raw'].astype(np.float64)
    else:
        raw = embed(paths)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(cache, raw=raw.astype(np.float32))
    return subjects, paths, mean_centre(raw)[0] if centred else l2(raw)
