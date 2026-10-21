"""Post-processing after max_min.sh / main.sh for one dataset.

- GIBA graph (GIBA_graph.sh) from <name>_CRR.dat
- FAR & FRR vs threshold plot (gnuplot) from <name>_FAR_FRR_Data.dat
- FA / FR / CRR-failure HTML galleries (fafrcrrhtml.py adapted to real image paths), full lists, paginated
- Independent cross-check metrics from the saved embeddings (sklearn ROC, rank-1)

Usage: python extras.py <AWE|ITW|NITJ|UBEAR>
"""
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve

from datasets import output_dir

PAGE_SIZE = 500
BASH = r'C:\Program Files\Git\bin\bash.exe'
CRR_RE = re.compile(r'^(\d+) (\d+) \t \[\((\d+),(\d+)\)([-\d.e]+)\] \t \[\((\d+),(\d+)\)([-\d.e]+)\] \t ([-\d.e]+) \t (\d)')


def load_id_map(out):
    paths = {}
    with (out / 'id_map.tsv').open() as f:
        next(f)
        for line in f:
            sid, pid, subject, path = line.rstrip('\n').split('\t')
            paths[(int(sid), int(pid))] = (subject, Path(path))
    return paths


def rel(path, html_dir):
    return Path(os.path.relpath(path, html_dir)).as_posix()


def run_sh(cmd, cwd):
    env = dict(os.environ)
    env['PATH'] = os.pathsep.join([str(Path(__file__).parent / 'bin'), r'C:\Program Files\gnuplot\bin', env['PATH']])
    subprocess.run([BASH, '-c', cmd], cwd=cwd, env=env, check=True)


def giba(res, name, run_dir):
    # GIBA_graph.sh plots columns 7 (best genuine) and 9 (best impostor) per test image.
    rows = []
    with (res / f'{name}_CRR.dat').open() as f:
        for line in f:
            m = CRR_RE.match(line)
            if m:
                rows.append(m.groups())
    with (res / f'{name}_GIBA.dat').open('w', newline='\n') as f:
        for r in rows:
            f.write(f'{r[0]}\t{r[1]}\t{r[2]}\t{r[3]}\t{r[5]}\t{r[6]}\t{r[4]}\t{r[9]}\t{r[7]}\t{r[8]}\n')
    run_sh(f'./GIBA_graph.sh "{name}/{name}_GIBA.png" "{name}/{name}_GIBA.dat" 2>/dev/null', run_dir)
    return rows


def far_frr_threshold_plot(res, name, run_dir):
    script = f'''gnuplot << PLOTTHIS
set terminal png
set output "{name}/{name}_FAR_FRR_vs_Threshold.png"
set grid
set title "FAR and FRR Vs Threshold"
set key center right
set xlabel "Threshold (normalized score x 10000)"
set ylabel "Error Rate (%)"
plot "{name}/{name}_FAR_FRR_Data.dat" using 1:3 title 'FRR' w lines lw 2, \\
     "{name}/{name}_FAR_FRR_Data.dat" using 1:6 title 'FAR' w lines lw 2
PLOTTHIS'''
    run_sh(script, run_dir)


def write_pages(entries, html_dir, stem, title, render):
    pages = max(1, (len(entries) + PAGE_SIZE - 1) // PAGE_SIZE)
    for p in range(pages):
        chunk = entries[p * PAGE_SIZE:(p + 1) * PAGE_SIZE]
        nav = ' '.join(
            f'<a href="{stem}_{q + 1:05d}.html">{q + 1}</a>' if q != p else f'<b>{q + 1}</b>'
            for q in range(max(0, p - 10), min(pages, p + 11)))
        with (html_dir / f'{stem}_{p + 1:05d}.html').open('w', encoding='utf-8') as f:
            f.write(f'<html>\n<head><meta charset="utf-8"><title>{title}</title></head>\n<body>\n')
            f.write(f'<h2>{title}</h2><p>{len(entries)} entries, page {p + 1} of {pages}. '
                    f'<a href="{stem}_00001.html">first</a> | {nav} | '
                    f'<a href="{stem}_{pages:05d}.html">last</a></p>\n')
            for e in chunk:
                f.write(render(e))
            f.write('</body>\n</html>\n')
    return pages


def galleries(res, name, ids, crr_rows):
    html_dir = res / 'html'
    html_dir.mkdir(exist_ok=True)

    def img(key, width):
        subject, path = ids[key]
        return (f'<img src="{rel(path, html_dir)}" alt="{subject}" '
                f'style="float:left;width:{width}%;height:100%;object-fit:contain;">\n')

    def pair_entry(e):
        s1, p1, s2, p2, score = e
        return ('<div style="width:60%;height:300px;">\n' + img((s1, p1), 50) + img((s2, p2), 50) +
                f'</div>\n<p>{ids[(s1, p1)][0]}.{p1} ; {ids[(s2, p2)][0]}.{p2} ; score {score}</p>\n')

    def crr_entry(r):
        return ('<div style="width:60%;height:300px;">\n' + img((int(r[0]), int(r[1])), 33) +
                img((int(r[2]), int(r[3])), 33) + img((int(r[5]), int(r[6])), 33) +
                f'</div>\n<p>probe {ids[(int(r[0]), int(r[1]))][0]}.{r[1]} ; best genuine '
                f'{ids[(int(r[2]), int(r[3]))][0]}.{r[3]} ({r[4]}) ; best impostor '
                f'{ids[(int(r[5]), int(r[6]))][0]}.{r[6]} ({r[7]})</p>\n')

    counts = {}
    for kind, title in (('FA', 'False Accepts at EER threshold'), ('FR', 'False Rejects at EER threshold')):
        entries = []
        with (res / f'{name}.{kind}').open() as f:
            for line in f:
                parts = line.split()
                if len(parts) == 6 and parts[0].isdigit():
                    entries.append((int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), parts[5]))
        counts[kind] = (len(entries), write_pages(entries, html_dir, kind, f'{name} - {title}', pair_entry))
    failed = [r for r in crr_rows if r[9] == '0']
    counts['CRR'] = (len(failed), write_pages(failed, html_dir, 'CRR', f'{name} - Identification failures '
                                              '(probe | best genuine | best impostor)', crr_entry))
    return counts


def cross_check(out, ids):
    emb = torch.from_numpy(np.load(out / 'embeddings_final.npy')).cuda()
    keys = list(ids)  # id_map.tsv order == embedding order
    subj = torch.tensor([k[0] for k in keys], device='cuda')
    n = len(keys)
    sim = emb @ emb.T
    iu = torch.triu_indices(n, n, 1, device='cuda')
    scores = sim[iu[0], iu[1]].cpu().numpy().astype(np.float64)
    labels = (subj[iu[0]] == subj[iu[1]]).cpu().numpy().astype(np.int8)
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    k = int(np.argmin(np.abs(fpr - fnr)))
    sim.fill_diagonal_(-float('inf'))
    rank1 = float((subj[sim.argmax(dim=1)] == subj).float().mean())
    return {
        'unique pairs': len(scores), 'genuine': int(labels.sum()), 'impostor': int(len(labels) - labels.sum()),
        'EER (%)': 100 * float((fpr[k] + fnr[k]) / 2),
        'VER@FAR=0.1% (%)': 100 * float(np.interp(0.001, fpr, tpr)),
        'VER@FAR=1% (%)': 100 * float(np.interp(0.01, fpr, tpr)),
        'AUC': float(roc_auc_score(labels, scores)),
        'Rank-1 (%)': 100 * rank1,
    }


def main(key):
    out = output_dir(key)
    run_dir = out / 'run'
    res = run_dir / key
    ids = load_id_map(out)
    crr_rows = giba(res, key, run_dir)
    far_frr_threshold_plot(res, key, run_dir)
    counts = galleries(res, key, ids, crr_rows)
    check = cross_check(out, ids)
    with (out / 'cross_check.txt').open('w') as f:
        f.write(f'{key}: independent cross-check (sklearn, unique pairs i<j, cosine similarity)\n')
        for k, v in check.items():
            f.write(f'{k}: {v:.6f}\n' if isinstance(v, float) else f'{k}: {v}\n')
        f.write('\nHTML galleries (html/): entries, pages\n')
        for k, (c, p) in counts.items():
            f.write(f'{k}: {c} entries, {p} pages\n')
    print((out / 'cross_check.txt').read_text())


if __name__ == '__main__':
    main(sys.argv[1])
