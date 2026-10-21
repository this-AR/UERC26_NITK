"""Verification and identification metrics on cosine similarities (higher = more similar).

verification(): EER, VER@FAR=0.1%, VER@FAR=1%, AUC and the decidability index
    DI = (mu_genuine - mu_impostor) / sqrt((sd_genuine^2 + sd_impostor^2) / 2).
rank1(): CRR as the fraction of probes whose most similar gallery image is the same subject.
"""
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve


def verification(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(np.int8)
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    k = int(np.argmin(np.abs(fpr - fnr)))
    g, i = scores[labels == 1], scores[labels == 0]
    return {
        'EER': 100 * float((fpr[k] + fnr[k]) / 2),
        'VER@0.1%': 100 * float(np.interp(1e-3, fpr, tpr)),
        'VER@1%': 100 * float(np.interp(1e-2, fpr, tpr)),
        'AUC': float(roc_auc_score(labels, scores)),
        'DI': float((g.mean() - i.mean()) / np.sqrt((g.var(ddof=1) + i.var(ddof=1)) / 2)),
        'genuine': int(len(g)),
        'impostor': int(len(i)),
    }


def unique_pairs(emb, subjects):
    """Cosine scores and genuine labels for all pairs i < j (computed on the GPU)."""
    e = torch.from_numpy(np.asarray(emb)).float().cuda()
    ids = torch.from_numpy(np.unique(subjects, return_inverse=True)[1]).cuda()
    iu = torch.triu_indices(len(e), len(e), 1, device='cuda')
    sim = e @ e.T
    scores = sim[iu[0], iu[1]].cpu().numpy().astype(np.float64)
    labels = (ids[iu[0]] == ids[iu[1]]).cpu().numpy().astype(np.int8)
    return scores, labels


def rank1(sim, probe_ids, gallery_ids, gallery_ok=None):
    """CRR (%) and number of probes that have at least one genuine gallery image.

    sim: probes x gallery similarity matrix (numpy or torch). Self-matches must already be
    excluded (set to -inf) by the caller. gallery_ok: optional boolean mask of allowed pairs.
    """
    s = torch.as_tensor(sim).float().cuda().clone()
    p = torch.as_tensor(np.unique(np.r_[probe_ids, gallery_ids], return_inverse=True)[1]).cuda()
    pid, gid = p[:len(probe_ids)], p[len(probe_ids):]
    if gallery_ok is not None:
        s[~torch.as_tensor(gallery_ok).cuda()] = -float('inf')
    same = pid[:, None] == gid[None, :]
    has_genuine = (same & torch.isfinite(s)).any(dim=1)
    hit = gid[s.argmax(dim=1)] == pid
    return 100 * float(hit[has_genuine].float().mean()), int(has_genuine.sum())


def all_vs_all(emb, subjects):
    """Verification on all unique pairs + CRR with every other image as gallery."""
    out = verification(*unique_pairs(emb, subjects))
    e = torch.from_numpy(np.asarray(emb)).float().cuda()
    sim = e @ e.T
    sim.fill_diagonal_(-float('inf'))
    out['CRR'] = rank1(sim, subjects, subjects)[0]
    return out


def right_vs_left(emb, subjects, sides):
    """Earlier NITJ protocol: right-ear probes against the left-ear gallery."""
    e = torch.from_numpy(np.asarray(emb)).float().cuda()
    r, l = sides == 'R', sides == 'L'
    sim = e[r] @ e[l].T
    labels = (subjects[r][:, None] == subjects[l][None, :]).ravel()
    out = verification(sim.cpu().numpy().ravel(), labels)
    out['CRR'] = rank1(sim, subjects[r], subjects[l])[0]
    return out
