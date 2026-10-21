"""Compile the Swin-V2 results against the pruned TinyViT into results/results_swinv2_vs_tinyvit.md.

Inputs:
- scripts_d_t_2 results: results/swinv2/<KEY>/<KEY>.sts (+ run_info.txt) and
  results/tinyvit/summary.txt (TinyViT, earlier run).
- sklearn metrics for both models and both embedding variants: results/analysis/model_metrics_{swinv2,tinyvit}.json
Also draws the combined ROC (roc_plot.sh style) from each dataset's <KEY>_FAR_FRR_Data.dat.

Usage (from the project root): python -m experiments.compare_models
"""
import json
import re

from ear_eval.config import ANALYSIS_DIR, RESULTS_DIR
from ear_eval.datasets import KEYS
from ear_eval.plotting import PALETTE, run_bash

NEW = RESULTS_DIR / 'swinv2'
OLD = RESULTS_DIR / 'tinyvit'


def sts(key):
    t = (NEW / key / f'{key}.sts').read_text()
    eer = re.search(r'\(EER\) = \s*\(([-\d.]+)\)\s+([-\d.]+)', t)
    far = re.findall(r'\(FAR\) = \s*([-\d.e]+)', t)
    return {'EER': float(eer.group(1)), 'thr': float(eer.group(2)),
            'CRR': float(re.search(r'\(CRR\) =\s+([-\d.e]+)', t).group(1)),
            'DI': float(re.search(r'\(DI\) = ([-\d.e]+)', t).group(1)),
            'FAR': float(far[0]), 'FRR': float(far[1]),
            'FR': re.search(r'Falsely Reject Matching =\s+(\d+) out of total (\d+)', t).groups(),
            'FA': re.search(r'Falsely Accept Matching =\s+(\d+) out of total (\d+)', t).groups(),
            'fail': re.search(r'Total Failed Subject =\s+(\d+) \(out of (\d+)\)', t).groups()}


def old_scripts():
    out = {}
    for line in (OLD / 'summary.txt').read_text().splitlines():
        p = line.split()
        if p and p[0] in KEYS and len(p) == 13:
            out[p[0]] = {'EER': float(p[5]), 'CRR': float(p[6]), 'DI': float(p[7]), 'FAR': float(p[8]), 'FRR': float(p[9])}
    return out


def combined_roc():
    curves = []
    for c, k in zip(PALETTE, KEYS):
        for root, dash, label in ((NEW, 1, 'Swin-V2'), (OLD, 2, 'TinyViT')):
            f = root / k / f'{k}_FAR_FRR_Data.dat'
            if f.exists():
                curves.append(f'"{f.as_posix()}" using 6:3 title \'{k} {label}\' w lines lw 2 dt {dash} lc rgb "{c}"')
    run_bash(f'''gnuplot << PLOTTHIS
set terminal png enhanced font 'Verdana,12' size 1000,750
set output "{(NEW / 'ROC_all_datasets.png').as_posix()}"
set grid
set title "ROC based Comparative Analysis{' (solid = Swin-V2, dashed = TinyViT)' if any('TinyViT' in c for c in curves) else ' - Swin-V2'}"
set key right top
set logscale x
set yrange [0:100]
set xrange [0.01:100]
set xlabel "FAR (%) (Plotted in Log-Scale)"
set ylabel "FRR (%)"
plot {', '.join(curves)}
PLOTTHIS''')


def main():
    combined_roc()
    new_q = json.loads((ANALYSIS_DIR / 'model_metrics_swinv2.json').read_text())
    old_q = json.loads((ANALYSIS_DIR / 'model_metrics_tinyvit.json').read_text())
    old_s = old_scripts()
    new_s = {k: sts(k) for k in KEYS}
    info = {k: dict(l.split(': ', 1) for l in (NEW / k / 'run_info.txt').read_text().splitlines() if ': ' in l) for k in KEYS}

    o = ['# New model (Swin-V2-B) vs pruned TinyViT: ear matching results\n']
    o.append('| | New model | Previous model |')
    o.append('|---|---|---|')
    o.append('| File | `best_model.pt` | `ITW/ITW_230_split_augment/best_pruned_35per_tinyvit.pt` |')
    o.append('| Architecture | torchvision Swin-V2-B + Linear(1024→512) + BatchNorm, AdaFace-trained (epoch 15, val VER 0.877) | Pruned TinyViT-11M + Linear(448→512) + BatchNorm |')
    o.append('| Parameters | 87.5 M | 7.53 M |')
    o.append('| Input | resize 224, RGB, ImageNet mean/std (model upsamples to 256); no CLAHE | resize 224, RGB, /255 |')
    o.append('')
    o.append('Both models are evaluated the same way:')
    o.append('- embed → L2 → subtract global mean → L2;')
    o.append('- all images vs all images;')
    o.append('- score = 1 − cosine, normalised by `max_min.sh`;')
    o.append('- metrics from `main.sh` (scripts_d_t_2).')
    o.append('')
    o.append('## 1. Main results (scripts_d_t_2, mean-centred)\n')
    o.append('| Dataset | Images / subjects | Model | EER (%) | CRR (%) | DI | FAR (%) | FRR (%) | VER@0.1% (%) | VER@1% (%) | AUC |')
    o.append('|---|---|---|---|---|---|---|---|---|---|---|')
    for k in KEYS:
        n, q = new_s[k], new_q[f'{k}|mean_centred']
        p, pq = old_s[k], old_q[f'{k}|mean_centred']
        o.append(f'| {k} | {info[k]["images"]} / {info[k]["subjects"]} | **Swin-V2** | **{n["EER"]:.3f}** | **{n["CRR"]:.2f}** | {n["DI"]:.3f} '
                 f'| {n["FAR"]:.3f} | {n["FRR"]:.3f} | {q["VER@0.1%"]:.2f} | {q["VER@1%"]:.2f} | {q["AUC"]:.4f} |')
        o.append(f'| | | TinyViT | {p["EER"]:.3f} | {p["CRR"]:.2f} | {p["DI"]:.3f} | {p["FAR"]:.3f} | {p["FRR"]:.3f} '
                 f'| {pq["VER@0.1%"]:.2f} | {pq["VER@1%"]:.2f} | {pq["AUC"]:.4f} |')
    o.append('')
    o.append('Where the numbers come from:')
    o.append('- EER, CRR, DI, FAR and FRR come from the scripts (`.sts`).')
    o.append('- VER and AUC come from sklearn on the same embeddings, because the scripts don\'t compute them.')
    o.append('- The scripts\' DI is negative because the scores are distances, and its size depends on the score scale, so compare DI between models with the positive DI in section 3 instead.')
    o.append('')
    o.append('## 2. Change (Swin-V2 minus TinyViT)\n')
    o.append('| Dataset | ΔEER (pts) | ΔCRR (pts) | ΔVER@0.1% (pts) | ΔVER@1% (pts) | ΔAUC |')
    o.append('|---|---|---|---|---|---|')
    for k in KEYS:
        n, p = new_s[k], old_s[k]
        q, pq = new_q[f'{k}|mean_centred'], old_q[f'{k}|mean_centred']
        o.append(f'| {k} | {n["EER"] - p["EER"]:+.2f} | {n["CRR"] - p["CRR"]:+.2f} | {q["VER@0.1%"] - pq["VER@0.1%"]:+.2f} '
                 f'| {q["VER@1%"] - pq["VER@1%"]:+.2f} | {q["AUC"] - pq["AUC"]:+.4f} |')
    o.append('')
    o.append('## 3. L2-only vs mean-centred, both models (sklearn, unique pairs; DI = (μg − μi)/√((σg² + σi²)/2))\n')
    o.append('| Dataset | Model | Variant | EER (%) | Rank-1 / CRR (%) | DI | VER@0.1% (%) | VER@1% (%) | AUC |')
    o.append('|---|---|---|---|---|---|---|---|---|')
    for k in KEYS + ['NITJ_RvsL']:
        for model, res in (('Swin-V2', new_q), ('TinyViT', old_q)):
            for v, vn in (('l2', 'L2-only'), ('mean_centred', 'mean-centred')):
                m = res[f'{k}|{v}']
                o.append(f'| {k.replace("_RvsL", " (R vs L)")} | {model} | {vn} | {m["EER"]:.3f} | {m["CRR"]:.2f} | {m["DI"]:.3f} '
                         f'| {m["VER@0.1%"]:.2f} | {m["VER@1%"]:.2f} | {m["AUC"]:.4f} |')
    o.append('')
    o.append('`NITJ (R vs L)` uses the earlier NITJ protocol: right-ear probes against a left-ear gallery (51 000 genuine and 25 959 000 impostor pairs).')
    o.append('')
    o.append('## 4. New model: full script output\n')
    o.append('| Dataset | Genuine (ordered) | Impostor (ordered) | EER threshold | False rejects | False accepts | Failed probes |')
    o.append('|---|---|---|---|---|---|---|')
    for k in KEYS:
        n = new_s[k]
        o.append(f'| {k} | {info[k]["genuine pairs"]} | {info[k]["impostor pairs"]} | {n["thr"]:.4f} | {n["FR"][0]} / {n["FR"][1]} '
                 f'| {n["FA"][0]} / {n["FA"][1]} | {n["fail"][0]} / {n["fail"][1]} |')
    o.append('')
    o.append('## 5. Graphs (new model)\n')
    o.append('- Combined ROC, all datasets: [swinv2/ROC_all_datasets.png](swinv2/ROC_all_datasets.png). '
             'The TinyViT ROC data was deleted in an earlier clean-up, so only Swin-V2 curves are drawn; '
             'TinyViT numbers are in the tables above and its own combined ROC is [tinyvit/ROC_all_datasets.png](tinyvit/ROC_all_datasets.png).')
    for k in KEYS:
        b = f'swinv2/{k}/{k}'
        o.append(f'- {k}: [ROC]({b}.png) · [Genuine vs Impostor]({b}_GI.png) · [FAR & FRR vs threshold]({b}_FAR_FRR_vs_Threshold.png) '
                 f'· [Best genuine vs best impostor]({b}_GIBA.png)')
    o.append('')
    out = RESULTS_DIR / 'results_swinv2_vs_tinyvit.md'
    out.write_text('\n'.join(o) + '\n', encoding='utf-8')
    print('written', out)


if __name__ == '__main__':
    main()
