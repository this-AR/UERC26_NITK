"""Step 3: post-processing after max_min.sh / main.sh for one dataset.

- GIBA graph (GIBA_graph.sh) from <KEY>_CRR.dat
- FAR & FRR vs threshold plot (gnuplot) from <KEY>_FAR_FRR_Data.dat
- Independent cross-check metrics from the saved embeddings (sklearn ROC, rank-1) -> cross_check.txt
- Optional (DT2_HTML=1): FA / FR / CRR-failure HTML galleries, full lists, paginated
- Tidy: keep only the metrics and images in results/<model>/<KEY>/:
  <KEY>.sts, *.png, <KEY>_FAR_FRR_Data.dat (ROC data), run_info.txt, cross_check.txt.
  DT2_KEEP_ALL=1 keeps the _work folder (pair-score files, FA/FR lists, histograms).

Usage (from the project root): python -m pipeline.postprocess <AWE|ITW|UBEAR|NITJ>
"""
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np

from ear_eval.config import result_dir
from ear_eval.metrics import all_vs_all
from ear_eval.plotting import run_bash

PAGE_SIZE = 500
CRR_RE = re.compile(r'^(\d+) (\d+) \t \[\((\d+),(\d+)\)([-\d.e]+)\] \t \[\((\d+),(\d+)\)([-\d.e]+)\] \t ([-\d.e]+) \t (\d)')


def load_id_map(work):
    ids = {}
    with (work / 'id_map.tsv').open() as f:
        next(f)
        for line in f:
            sid, pid, subject, path = line.rstrip('\n').split('\t')
            ids[(int(sid), int(pid))] = (subject, Path(path))
    return ids


def giba(res, key, work):
    # GIBA_graph.sh plots columns 7 (best genuine) and 9 (best impostor) per test image.
    rows = []
    with (res / f'{key}_CRR.dat').open() as f:
        for line in f:
            m = CRR_RE.match(line)
            if m:
                rows.append(m.groups())
    with (res / f'{key}_GIBA.dat').open('w', newline='\n') as f:
        for r in rows:
            f.write(f'{r[0]}\t{r[1]}\t{r[2]}\t{r[3]}\t{r[5]}\t{r[6]}\t{r[4]}\t{r[9]}\t{r[7]}\t{r[8]}\n')
    run_bash(f'./GIBA_graph.sh "{key}/{key}_GIBA.png" "{key}/{key}_GIBA.dat" 2>/dev/null', work)
    return rows


def far_frr_threshold_plot(key, work):
    run_bash(f'''gnuplot << PLOTTHIS
set terminal png
set output "{key}/{key}_FAR_FRR_vs_Threshold.png"
set grid
set title "FAR and FRR Vs Threshold"
set key center right
set xlabel "Threshold (normalized score x 10000)"
set ylabel "Error Rate (%)"
plot "{key}/{key}_FAR_FRR_Data.dat" using 1:3 title 'FRR' w lines lw 2, \\
     "{key}/{key}_FAR_FRR_Data.dat" using 1:6 title 'FAR' w lines lw 2
PLOTTHIS''', work)


# ---- optional HTML galleries (fafrcrrhtml.py adapted to real image paths) ------------------------
def _write_pages(entries, html_dir, stem, title, render):
    pages = max(1, (len(entries) + PAGE_SIZE - 1) // PAGE_SIZE)
    for p in range(pages):
        nav = ' '.join(f'<a href="{stem}_{q + 1:05d}.html">{q + 1}</a>' if q != p else f'<b>{q + 1}</b>'
                       for q in range(max(0, p - 10), min(pages, p + 11)))
        with (html_dir / f'{stem}_{p + 1:05d}.html').open('w', encoding='utf-8') as f:
            f.write(f'<html>\n<head><meta charset="utf-8"><title>{title}</title></head>\n<body>\n'
                    f'<h2>{title}</h2><p>{len(entries)} entries, page {p + 1} of {pages}. '
                    f'<a href="{stem}_00001.html">first</a> | {nav} | <a href="{stem}_{pages:05d}.html">last</a></p>\n')
            for e in entries[p * PAGE_SIZE:(p + 1) * PAGE_SIZE]:
                f.write(render(e))
            f.write('</body>\n</html>\n')
    return pages


def galleries(res, key, ids, crr_rows, html_dir):
    html_dir.mkdir(parents=True, exist_ok=True)

    def img(k, width):
        subject, path = ids[k]
        return (f'<img src="{Path(os.path.relpath(path, html_dir)).as_posix()}" alt="{subject}" '
                f'style="float:left;width:{width}%;height:100%;object-fit:contain;">\n')

    def pair_entry(e):
        s1, p1, s2, p2, score = e
        return ('<div style="width:60%;height:300px;">\n' + img((s1, p1), 50) + img((s2, p2), 50) +
                f'</div>\n<p>{ids[(s1, p1)][0]}.{p1} ; {ids[(s2, p2)][0]}.{p2} ; score {score}</p>\n')

    def crr_entry(r):
        k = [(int(r[0]), int(r[1])), (int(r[2]), int(r[3])), (int(r[5]), int(r[6]))]
        return ('<div style="width:60%;height:300px;">\n' + ''.join(img(x, 33) for x in k) +
                f'</div>\n<p>probe {ids[k[0]][0]}.{r[1]} ; best genuine {ids[k[1]][0]}.{r[3]} ({r[4]}) ; '
                f'best impostor {ids[k[2]][0]}.{r[6]} ({r[7]})</p>\n')

    for kind, title in (('FA', 'False Accepts at EER threshold'), ('FR', 'False Rejects at EER threshold')):
        entries = []
        with (res / f'{key}.{kind}').open() as f:
            for line in f:
                parts = line.split()
                if len(parts) == 6 and parts[0].isdigit():
                    entries.append((int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), parts[5]))
        _write_pages(entries, html_dir, kind, f'{key} - {title}', pair_entry)
    failed = [r for r in crr_rows if r[9] == '0']
    _write_pages(failed, html_dir, 'CRR', f'{key} - Identification failures (probe | best genuine | best impostor)',
                 crr_entry)


# ---- cross-check and tidy ------------------------------------------------------------------------
def cross_check(out, work, ids):
    emb = np.load(work / 'embeddings_final.npy')
    subjects = np.array([sid for sid, _ in ids])  # id_map order == embedding order
    m = all_vs_all(emb, subjects)
    with (out / 'cross_check.txt').open('w') as f:
        f.write(f'{out.name}: independent cross-check (sklearn, unique pairs i<j, cosine similarity)\n')
        f.write(f'unique pairs: {m["genuine"] + m["impostor"]}\ngenuine: {m["genuine"]}\nimpostor: {m["impostor"]}\n')
        for k, name in (('EER', 'EER (%)'), ('VER@0.1%', 'VER@FAR=0.1% (%)'), ('VER@1%', 'VER@FAR=1% (%)'),
                        ('AUC', 'AUC'), ('DI', 'DI (similarity scale)'), ('CRR', 'Rank-1 (%)')):
            f.write(f'{name}: {m[k]:.6f}\n')
    print((out / 'cross_check.txt').read_text())


def tidy(out, res, key):
    keep = [res / f'{key}.sts', res / f'{key}_FAR_FRR_Data.dat', *res.glob('*.png')]
    for f in keep:
        shutil.move(str(f), out / f.name)
    if (res / 'html').exists():
        shutil.move(str(res / 'html'), out / 'html')
    shutil.rmtree(out / '_work')


def main(key):
    out = result_dir(key)
    work = out / '_work'
    res = work / key
    ids = load_id_map(work)
    crr_rows = giba(res, key, work)
    far_frr_threshold_plot(key, work)
    if os.environ.get('DT2_HTML') == '1':
        galleries(res, key, ids, crr_rows, res / 'html')
    cross_check(out, work, ids)
    if os.environ.get('DT2_KEEP_ALL') != '1':
        tidy(out, res, key)


if __name__ == '__main__':
    main(sys.argv[1])
