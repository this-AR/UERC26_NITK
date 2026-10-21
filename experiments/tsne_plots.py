"""t-SNE of the mean-centred embeddings for every dataset.

Per dataset: one panel with every image in grey and 8 randomly chosen subjects
highlighted in the fixed categorical palette (more than 8 subject colours would be
indistinguishable). UBEAR also gets panels coloured by ear side, session and pose.
Writes results/analysis/<model>/tsne/<KEY>_tsne.png and <KEY>_tsne_8subjects.png.

Usage (from the project root): python -m experiments.tsne_plots [AWE ITW UBEAR NITJ]
"""
import sys

import matplotlib
import numpy as np
from sklearn.manifold import TSNE

from ear_eval.config import ANALYSIS_DIR, model_tag
from ear_eval.datasets import KEYS
from ear_eval.embeddings import dataset_embeddings
from ear_eval.plotting import GREY, INK, INK2, PALETTE, SURFACE

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

OUT = ANALYSIS_DIR / model_tag() / 'tsne'


def style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, color=INK, fontsize=12, loc='left')


def categorical(ax, xy, labels, order, title, names=None):
    style(ax, title)
    ax.scatter(xy[:, 0], xy[:, 1], s=4, c=GREY, linewidths=0)
    for colour, value in zip(PALETTE, order):
        m = labels == value
        ax.scatter(xy[m, 0], xy[m, 1], s=22, c=colour, edgecolors=SURFACE, linewidths=0.8,
                   label=f'{(names or {}).get(value, value)} (n={m.sum()})')
    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=9, labelcolor=INK2,
              markerscale=1.2)


def run(key):
    subjects, paths, emb = dataset_embeddings(key)
    n = len(subjects)
    print(f'{key}: t-SNE on {n} embeddings ...', flush=True)
    xy = TSNE(n_components=2, perplexity=30, init='pca', metric='cosine', random_state=0,
              n_jobs=-1).fit_transform(emb)
    rng = np.random.default_rng(0)
    uniq = np.unique(subjects)
    pick = rng.choice(uniq, size=min(8, len(uniq)), replace=False)

    if key != 'UBEAR':
        fig, ax = plt.subplots(figsize=(9, 7), facecolor=SURFACE)
        categorical(ax, xy, subjects, pick,
                    f'{key}: t-SNE of mean-centred embeddings ({n} images, {len(uniq)} subjects)\n'
                    'grey = all images, colour = 8 random subjects')
    else:
        stems = [p.stem.split('_') for p in paths]
        side = np.array([s[3] for s in stems])
        session = np.array([s[2] for s in stems])
        pose = np.array([s[6] for s in stems])
        fig, axes = plt.subplots(2, 2, figsize=(16, 13), facecolor=SURFACE)
        categorical(axes[0, 0], xy, subjects, pick, f'UBEAR: 8 random subjects ({n} images, {len(uniq)} subjects)')
        categorical(axes[0, 1], xy, side, ['L', 'R'], 'Coloured by ear side', {'L': 'Left ear', 'R': 'Right ear'})
        categorical(axes[1, 0], xy, session, ['1', '2'], 'Coloured by session', {'1': 'Session 1', '2': 'Session 2'})
        categorical(axes[1, 1], xy, pose, ['M', 'U', 'D', 'T', 'O'], 'Coloured by head pose',
                    {'M': 'M (middle)', 'U': 'U (up)', 'D': 'D (down)', 'T': 'T (towards)', 'O': 'O (outwards)'})
        fig.suptitle('UBEAR: t-SNE of mean-centred embeddings', color=INK, fontsize=14, x=0.02, ha='left')
    fig.tight_layout()
    fig.savefig(OUT / f'{key}_tsne.png', dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # One extra view per dataset: the same 8 subjects, zoomed so their clusters are readable.
    fig, ax = plt.subplots(figsize=(9, 7), facecolor=SURFACE)
    m = np.isin(subjects, pick)
    categorical(ax, xy[m], subjects[m], pick, f'{key}: the 8 highlighted subjects only')
    fig.tight_layout()
    fig.savefig(OUT / f'{key}_tsne_8subjects.png', dpi=150, facecolor=SURFACE)
    plt.close(fig)


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    for k in sys.argv[1:] or KEYS:
        run(k)
