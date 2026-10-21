"""Why does UBEAR fail? Controlled tests on the mean-centred embeddings.

UBEAR file name: s_<subject>_<session>_<side L/R>_<age>_<gender>_<pose>_<frame>.tiff
Identity = subject (left and right ear of one person are the same identity).

Writes results/analysis/<model>/ubear/report.txt, results.json and figures.

Usage (from the project root): python -m experiments.ubear_failure_analysis
"""
import json

import cv2
import matplotlib
import numpy as np
from scipy.stats import spearmanr

from ear_eval.config import ANALYSIS_DIR, model_tag
from ear_eval.datasets import discover
from ear_eval.embeddings import dataset_embeddings, embed, mean_centre
from ear_eval.metrics import rank1, verification
from ear_eval.plotting import INK, INK2, PALETTE, SURFACE

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

OUT = ANALYSIS_DIR / model_tag() / 'ubear'
BLUE, ORANGE, AQUA = PALETTE[:3]
lines = []


def log(text=''):
    print(text, flush=True)
    lines.append(text)


def crr(sim, ident, gallery_ok=None):
    """Rank-1: each probe vs all other images (optionally only where gallery_ok[i, j])."""
    s = sim.copy()
    np.fill_diagonal(s, -np.inf)
    return rank1(s, ident, ident, gallery_ok)


def fmt(m):
    return (f"EER {m['EER']:6.2f}%  VER@0.1% {m['VER@0.1%']:6.2f}%  VER@1% {m['VER@1%']:6.2f}%  "
            f"AUC {m['AUC']:.4f}  (genuine {m['genuine']}, impostor {m['impostor']})")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    subjects, paths, emb = dataset_embeddings('UBEAR')
    parts = [p.stem.split('_') for p in paths]
    session = np.array([p[2] for p in parts])
    side = np.array([p[3] for p in parts])
    gender = np.array([p[5].upper() for p in parts])
    pose = np.array([p[6] for p in parts])
    seq = np.array([f'{s}|{a}|{b}|{c}' for s, a, b, c in zip(subjects, session, side, pose)])
    n = len(paths)
    iu = np.triu_indices(n, 1)
    same = lambda a: (a[iu[0]] == a[iu[1]])  # noqa: E731
    gen = same(subjects)

    sim = emb @ emb.T
    scores = sim[iu]
    log('UBEAR failure analysis (mean-centred embeddings, cosine similarity, unique pairs)')
    log('=' * 90)
    log(f'{n} images, {len(np.unique(subjects))} subjects, {len(np.unique(seq))} video sequences '
        f'(subject+session+side+pose)')
    log(f'side L/R: {(side == "L").sum()}/{(side == "R").sum()}   session 1/2: {(session == "1").sum()}/'
        f'{(session == "2").sum()}   poses: ' + ', '.join(f'{p}={c}' for p, c in zip(*np.unique(pose, return_counts=True))))
    log()
    base = verification(scores, gen)
    base_crr = crr(sim, subjects)
    log('[T0] Baseline (identity = subject, all pairs)')
    log(f'     {fmt(base)}   CRR {base_crr[0]:.2f}%')
    results = {'baseline': {**base, 'CRR': base_crr[0]}}

    # ---- T1: left vs right ear ---------------------------------------------------------
    log('\n[T1] Left vs right ear of the same person')
    same_side = same(side)
    imp = scores[~gen]
    for name, m in (('same-side genuine', gen & same_side), ('cross-side genuine (L vs R)', gen & ~same_side)):
        g = scores[m]
        r = verification(np.concatenate([g, imp]), np.r_[np.ones(len(g)), np.zeros(len(imp))])
        log(f'     {name:<30} n={len(g):>6}  mean {g.mean():.3f} ± {g.std():.3f}  vs all impostors: EER {r["EER"]:.2f}%')
        results[f'T1 {name}'] = {'n': len(g), 'mean': g.mean(), 'EER': r['EER']}
    log(f'     impostor                       n={len(imp):>6}  mean {imp.mean():.3f} ± {imp.std():.3f}')
    share = (gen & ~same_side).sum() / gen.sum()
    log(f'     {100 * share:.1f}% of all genuine pairs are cross-side')
    ident_side = np.char.add(np.char.add(subjects, '_'), side)
    m_side = verification(scores, same(ident_side))
    c_side = crr(sim, ident_side)
    log(f'     identity = subject+side      {fmt(m_side)}   CRR {c_side[0]:.2f}%')
    results['T1 identity subject+side'] = {**m_side, 'CRR': c_side[0]}

    # ---- T2: mirroring left ears ---------------------------------------------------------
    log('\n[T2] Mirror every left-ear image (horizontal flip) before embedding')
    raw_flip = embed(paths, flip=(side == 'L'))
    emb_flip = mean_centre(raw_flip)[0]
    sim_flip = emb_flip @ emb_flip.T
    s_flip = sim_flip[iu]
    m_flip = verification(s_flip, gen)
    c_flip = crr(sim_flip, subjects)
    g_cross = s_flip[gen & ~same_side]
    log(f'     identity = subject           {fmt(m_flip)}   CRR {c_flip[0]:.2f}%')
    log(f'     cross-side genuine mean {scores[gen & ~same_side].mean():.3f} -> {g_cross.mean():.3f} after mirroring')
    results['T2 mirrored'] = {**m_flip, 'CRR': c_flip[0], 'cross_side_mean': g_cross.mean()}

    # ---- T3: session -----------------------------------------------------------------------
    log('\n[T3] Session (genuine pairs, same side only, to isolate the session effect)')
    same_sess = same(session)
    for name, m in (('same session', gen & same_side & same_sess), ('cross session', gen & same_side & ~same_sess)):
        g = scores[m]
        if len(g) == 0:
            continue
        r = verification(np.concatenate([g, imp]), np.r_[np.ones(len(g)), np.zeros(len(imp))])
        log(f'     {name:<15} n={len(g):>6}  mean {g.mean():.3f} ± {g.std():.3f}  EER vs impostors {r["EER"]:.2f}%')
        results[f'T3 {name}'] = {'n': len(g), 'mean': g.mean(), 'EER': r['EER']}
    two_sess = len(np.unique(subjects[session == '2']))
    log(f'     subjects with a session-2 capture: {two_sess} of {len(np.unique(subjects))}')

    # ---- T4: pose --------------------------------------------------------------------------
    log('\n[T4] Head pose (genuine pairs, same side, same session)')
    poses = ['M', 'U', 'D', 'T', 'O']
    mat = np.full((5, 5), np.nan)
    base_mask = gen & same_side & same_sess
    for a, pa in enumerate(poses):
        for b, pb in enumerate(poses):
            m = base_mask & (((pose[iu[0]] == pa) & (pose[iu[1]] == pb)) | ((pose[iu[0]] == pb) & (pose[iu[1]] == pa)))
            if m.any():
                mat[a, b] = scores[m].mean()
    log('     mean genuine cosine by pose pair:')
    log('           ' + '  '.join(f'{p:>6}' for p in poses))
    for a, pa in enumerate(poses):
        log(f'       {pa:>3} ' + '  '.join(f'{v:6.3f}' for v in mat[a]))
    same_pose = same(pose)
    for name, m in (('same pose', base_mask & same_pose), ('different pose', base_mask & ~same_pose)):
        g = scores[m]
        r = verification(np.concatenate([g, imp]), np.r_[np.ones(len(g)), np.zeros(len(imp))])
        log(f'     {name:<15} n={len(g):>6}  mean {g.mean():.3f}  EER vs impostors {r["EER"]:.2f}%')
        results[f'T4 {name}'] = {'n': len(g), 'mean': g.mean(), 'EER': r['EER']}
    results['T4 pose matrix'] = mat.tolist()

    # ---- T5: near-duplicate video frames ---------------------------------------------------
    log('\n[T5] Near-duplicate video frames (why CRR is high while EER is high)')
    same_seq_full = seq[:, None] == seq[None, :]
    s_nd = sim.copy()
    np.fill_diagonal(s_nd, -np.inf)
    nn = s_nd.argmax(axis=1)
    log(f'     rank-1 neighbour is a frame from the SAME video sequence: {100 * (seq[nn] == seq).mean():.2f}% of probes')
    g_seq = scores[same(seq)]
    log(f'     genuine pairs inside one sequence: n={len(g_seq)}, mean {g_seq.mean():.3f}; '
        f'genuine pairs across sequences: mean {scores[gen & ~same(seq)].mean():.3f}')
    c_noseq = crr(sim, subjects, ~same_seq_full)
    same_sess_full = session[:, None] == session[None, :]
    c_nosess = crr(sim, subjects, ~same_sess_full)
    same_side_full = side[:, None] == side[None, :]
    c_noside = crr(sim, subjects, ~same_side_full)
    log(f'     CRR, gallery excludes the probe\'s own sequence      : {c_noseq[0]:.2f}% ({c_noseq[1]} probes)')
    log(f'     CRR, gallery = other session only                   : {c_nosess[0]:.2f}% ({c_nosess[1]} probes)')
    log(f'     CRR, gallery = opposite ear only                    : {c_noside[0]:.2f}% ({c_noside[1]} probes)')
    m_noseq = verification(scores[~same(seq)], gen[~same(seq)])
    log(f'     verification without within-sequence pairs           : {fmt(m_noseq)}')
    results['T5'] = {'nn_same_sequence_%': 100 * (seq[nn] == seq).mean(), 'CRR_no_own_sequence': c_noseq[0],
                     'CRR_other_session': c_nosess[0], 'CRR_opposite_ear': c_noside[0], 'EER_no_within_seq': m_noseq['EER']}

    # ---- T6: image quality -----------------------------------------------------------------
    log('\n[T6] Image quality and geometry')
    q = []
    for p in paths:
        im = cv2.imread(str(p))
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        q.append((cv2.Laplacian(g, cv2.CV_64F).var(), im.shape[0], im.shape[1], g.mean(), g.std()))
    q = np.array(q)
    sharp, h, w, bright, contrast = q.T
    aspect = h / w
    # Per-image difficulty: mean similarity to same-subject images from OTHER sequences.
    diff_seq_gen = (subjects[:, None] == subjects[None, :]) & ~same_seq_full
    gscore = np.array([sim[i, diff_seq_gen[i]].mean() if diff_seq_gen[i].any() else np.nan for i in range(n)])
    ok = ~np.isnan(gscore)
    for name, v in (('sharpness (Laplacian var)', sharp), ('height (px)', h), ('width (px)', w),
                    ('aspect h/w', aspect), ('brightness', bright), ('contrast', contrast)):
        rho, pv = spearmanr(v[ok], gscore[ok])
        log(f'     {name:<26} median {np.median(v):8.2f}  [{np.percentile(v, 5):.2f} .. {np.percentile(v, 95):.2f}]  '
            f'Spearman rho vs genuine score {rho:+.3f} (p={pv:.1e})')
        results[f'T6 {name}'] = {'median': float(np.median(v)), 'rho': rho}
    for name, v in (('sharpness', sharp), ('width', w)):
        qs = np.quantile(v, [0.25, 0.5, 0.75])
        log(f'     mean genuine score by {name} quartile: ' + '  '.join(
            f'Q{i + 1} {np.nanmean(gscore[(np.digitize(v, qs) == i)]):.3f}' for i in range(4)))

    # ---- T7: aspect-ratio-preserving resize ------------------------------------------------
    log('\n[T7] Aspect-ratio-preserving input (letterbox to square) instead of stretching to 224x224')
    raw_lb = embed(paths, letterbox=True)
    e_lb = mean_centre(raw_lb)[0]
    sim_lb = e_lb @ e_lb.T
    m_lb = verification(sim_lb[iu], gen)
    c_lb = crr(sim_lb, subjects)
    log(f'     letterbox                    {fmt(m_lb)}   CRR {c_lb[0]:.2f}%')
    raw_both = embed(paths, flip=(side == 'L'), letterbox=True)
    e_both = mean_centre(raw_both)[0]
    sim_both = e_both @ e_both.T
    m_both = verification(sim_both[iu], gen)
    c_both = crr(sim_both, subjects)
    log(f'     letterbox + mirror L         {fmt(m_both)}   CRR {c_both[0]:.2f}%')
    results['T7 letterbox'] = {**m_lb, 'CRR': c_lb[0]}
    results['T7 letterbox+mirror'] = {**m_both, 'CRR': c_both[0]}

    # ---- T8: what drives high-scoring impostors ---------------------------------------------
    log('\n[T8] Hardest impostors (top 0.1% impostor scores) - shared attributes vs base rate')
    imp_idx = np.flatnonzero(~gen)
    top = imp_idx[np.argsort(scores[imp_idx])[-max(1, len(imp_idx) // 1000):]]
    for name, attr in (('same side', side), ('same gender', gender), ('same pose', pose), ('same session', session)):
        base_rate = same(attr)[imp_idx].mean()
        top_rate = same(attr)[top].mean()
        log(f'     {name:<13} top impostors {100 * top_rate:5.1f}%   all impostors {100 * base_rate:5.1f}%')
        results[f'T8 {name}'] = {'top': 100 * top_rate, 'base': 100 * base_rate}

    # ---- T9: compare geometry with the other datasets ---------------------------------------
    log('\n[T9] Image geometry across datasets (stretch factor when resized to 224x224)')
    for key in ('AWE', 'ITW', 'NITJ', 'UBEAR'):
        recs = discover(key)[::max(1, len(discover(key)) // 300)]
        shp = np.array([cv2.imread(str(p)).shape[:2] for _, p in recs])
        ar = shp[:, 0] / shp[:, 1]
        log(f'     {key:<6} median size {int(np.median(shp[:, 0]))}x{int(np.median(shp[:, 1]))}  '
            f'aspect h/w median {np.median(ar):.2f} [{np.percentile(ar, 5):.2f} .. {np.percentile(ar, 95):.2f}]')

    (OUT / 'report.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (OUT / 'results.json').write_text(json.dumps(results, indent=1, default=float))

    figures(scores, gen, same_side, s_flip, mat, poses, sim, subjects, seq, paths, gscore, sharp, w)


def figures(scores, gen, same_side, s_flip, mat, poses, sim, subjects, seq, paths, gscore, sharp, w):
    imp = scores[~gen]
    # 1. score distributions
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    bins = np.linspace(-0.7, 1.0, 120)
    for data, colour, label in ((imp, INK2, 'Impostor'), (scores[gen & same_side], BLUE, 'Genuine, same ear side'),
                                (scores[gen & ~same_side], ORANGE, 'Genuine, left vs right ear'),
                                (s_flip[gen & ~same_side], AQUA, 'Genuine, left vs right after mirroring')):
        ax.hist(data, bins=bins, density=True, histtype='step', lw=2, color=colour, label=f'{label} (n={len(data)})')
    ax.set_xlabel('Cosine similarity (mean-centred)', color=INK2)
    ax.set_ylabel('Density', color=INK2)
    ax.set_title('UBEAR: genuine vs impostor scores, split by ear side', color=INK, loc='left')
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, labelcolor=INK2)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / 'score_distributions_by_side.png', dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # 2. pose matrix (sequential, one hue)
    fig, ax = plt.subplots(figsize=(6.5, 5.5), facecolor=SURFACE)
    im = ax.imshow(mat, cmap='Blues', vmin=np.nanmin(mat), vmax=np.nanmax(mat))
    ax.set_xticks(range(5), poses)
    ax.set_yticks(range(5), poses)
    for a in range(5):
        for b in range(5):
            if not np.isnan(mat[a, b]):
                ax.text(b, a, f'{mat[a, b]:.2f}', ha='center', va='center',
                        color='white' if mat[a, b] > np.nanmean(mat) else INK, fontsize=10)
    ax.set_title('Mean genuine cosine by head-pose pair\n(same side, same session)', color=INK, loc='left')
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(OUT / 'pose_matrix.png', dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # 3. failure gallery: CRR failures (probe | best genuine | best impostor)
    s = sim.copy()
    np.fill_diagonal(s, -np.inf)
    same_subj = subjects[:, None] == subjects[None, :]
    best_g = np.where(same_subj, s, -np.inf).argmax(1)
    best_i = np.where(~same_subj, s, -np.inf).argmax(1)
    fail = np.flatnonzero(s[np.arange(len(s)), best_i] > s[np.arange(len(s)), best_g])
    gallery(fail[:12], best_g, best_i, s, paths, 'crr_failures.png',
            f'UBEAR identification failures ({len(fail)} probes): probe | best genuine | best impostor (wins)')

    # 4. hardest genuine pairs across sequences
    diff_seq = same_subj & (seq[:, None] != seq[None, :])
    gs = np.where(diff_seq, s, np.inf)
    worst = np.argsort(gs.min(1))[:12]
    gallery(worst, gs.argmin(1), best_i, s, paths, 'worst_genuine_pairs.png',
            'UBEAR lowest-scoring genuine pairs: probe | least similar same-person image | best impostor')

    # 5. quality vs genuine score
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=SURFACE)
    for ax, v, name in ((axes[0], sharp, 'Sharpness (Laplacian variance, log scale)'), (axes[1], w, 'Image width (px)')):
        ax.set_facecolor(SURFACE)
        ax.scatter(v, gscore, s=8, c=BLUE, alpha=0.4, linewidths=0)
        if 'log' in name:
            ax.set_xscale('log')
        ax.set_xlabel(name, color=INK2)
        ax.set_ylabel('Mean genuine score (other sequences)', color=INK2)
        ax.grid(alpha=0.25)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    fig.suptitle('UBEAR: image quality vs how well an image matches its own subject', color=INK, x=0.02, ha='left')
    fig.tight_layout()
    fig.savefig(OUT / 'quality_vs_score.png', dpi=150, facecolor=SURFACE)
    plt.close(fig)


def gallery(rows, best_g, best_i, s, paths, name, title):
    fig, axes = plt.subplots(len(rows), 3, figsize=(7.5, 2.6 * len(rows)), facecolor=SURFACE)
    axes = np.atleast_2d(axes)
    for r, i in enumerate(rows):
        for c, (j, label) in enumerate(((i, 'probe'), (best_g[i], 'genuine'), (best_i[i], 'impostor'))):
            ax = axes[r, c]
            im = cv2.imread(str(paths[j]))[:, :, ::-1]
            ax.imshow(im)
            ax.set_xticks([])
            ax.set_yticks([])
            sc = '' if c == 0 else f' {s[i, j]:.2f}'
            ax.set_title(f'{label}{sc}\n{paths[j].stem}', fontsize=7, color=INK2)
    fig.suptitle(title, color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=110, facecolor=SURFACE)
    plt.close(fig)


if __name__ == '__main__':
    main()
