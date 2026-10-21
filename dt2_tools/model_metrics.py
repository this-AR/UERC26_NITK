"""Quick verification/identification metrics for the current model (DT2_MODEL / DT2_CLAHE).

For each dataset, both embedding variants (L2-only, mean-centred), all unique pairs:
EER, VER@0.1%, VER@1%, AUC, DI and Rank-1 (CRR, each image vs all others).
NITJ is also scored with the earlier protocol: right-ear probes vs left-ear gallery.
Writes analysis/model_metrics<tag>.json.

Usage: python model_metrics.py [AWE ITW UBEAR NITJ]
"""
import json
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve

import models
from analysis_common import ANALYSIS_DIR, dataset_embeddings


def metrics(scores, labels):
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    k = int(np.argmin(np.abs(fpr - fnr)))
    g, i = scores[labels == 1], scores[labels == 0]
    return {'EER': 100 * (fpr[k] + fnr[k]) / 2, 'VER@0.1%': 100 * np.interp(1e-3, fpr, tpr),
            'VER@1%': 100 * np.interp(1e-2, fpr, tpr), 'AUC': float(roc_auc_score(labels, scores)),
            'DI': float((g.mean() - i.mean()) / np.sqrt((g.var(ddof=1) + i.var(ddof=1)) / 2)),
            'genuine': int(len(g)), 'impostor': int(len(i))}


def all_vs_all(emb, subjects):
    e = torch.from_numpy(emb).float().cuda()
    ids = torch.from_numpy(np.unique(subjects, return_inverse=True)[1]).cuda()
    sim = e @ e.T
    iu = torch.triu_indices(len(e), len(e), 1, device='cuda')
    out = metrics(sim[iu[0], iu[1]].cpu().numpy().astype(np.float64),
                  (ids[iu[0]] == ids[iu[1]]).cpu().numpy().astype(np.int8))
    sim.fill_diagonal_(-float('inf'))
    out['CRR'] = 100 * float((ids[sim.argmax(1)] == ids).float().mean())
    return out


def right_vs_left(emb, subjects, paths):
    side = np.array([p.stem.split('_')[4] for p in paths])
    r, l = side == 'R', side == 'L'
    e = torch.from_numpy(emb).float().cuda()
    sim = e[r] @ e[l].T
    lab = torch.from_numpy(subjects[r][:, None] == subjects[l][None, :]).cuda()
    out = metrics(sim.flatten().cpu().numpy().astype(np.float64), lab.flatten().cpu().numpy().astype(np.int8))
    out['CRR'] = 100 * float(lab.gather(1, sim.argmax(1, keepdim=True)).float().mean())
    return out


def main(keys):
    path = ANALYSIS_DIR / f'model_metrics{models.tag() or "_tinyvit"}.json'
    results = json.loads(path.read_text()) if path.exists() else {}
    for key in keys:
        for variant, centred in (('l2', False), ('mean_centred', True)):
            subjects, paths, emb = dataset_embeddings(key, centred=centred)
            results[f'{key}|{variant}'] = all_vs_all(emb, subjects)
            if key == 'NITJ':
                results[f'NITJ_RvsL|{variant}'] = right_vs_left(emb, subjects, paths)
        for k in [k for k in results if k.split('|')[0] in (key, f'{key}_RvsL')]:
            m = results[k]
            print(f'{k:<26} EER {m["EER"]:6.2f}  CRR {m["CRR"]:6.2f}  VER@0.1% {m["VER@0.1%"]:6.2f}  '
                  f'VER@1% {m["VER@1%"]:6.2f}  AUC {m["AUC"]:.4f}  DI {m["DI"]:.3f}', flush=True)
        path.write_text(json.dumps(results, indent=1))


if __name__ == '__main__':
    main(sys.argv[1:] or ['AWE', 'ITW', 'UBEAR', 'NITJ'])
