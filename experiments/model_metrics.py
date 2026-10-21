"""Quick verification / identification metrics for the selected model (DT2_MODEL / DT2_CLAHE).

For each dataset, both embedding variants (L2-only, mean-centred), all unique pairs:
EER, VER@0.1%, VER@1%, AUC, DI and Rank-1 (CRR, each image vs all others).
NITJ is also scored with the earlier protocol: right-ear probes vs left-ear gallery.
Writes results/analysis/model_metrics_<model>.json (existing entries for other datasets are kept).

Usage (from the project root): python -m experiments.model_metrics [AWE ITW UBEAR NITJ]
"""
import json
import sys

import numpy as np

from ear_eval.config import ANALYSIS_DIR, model_tag
from ear_eval.datasets import KEYS
from ear_eval.embeddings import dataset_embeddings
from ear_eval.metrics import all_vs_all, right_vs_left


def main(keys):
    path = ANALYSIS_DIR / f'model_metrics_{model_tag()}.json'
    results = json.loads(path.read_text()) if path.exists() else {}
    for key in keys:
        for variant, centred in (('l2', False), ('mean_centred', True)):
            subjects, paths, emb = dataset_embeddings(key, centred=centred)
            results[f'{key}|{variant}'] = all_vs_all(emb, subjects)
            if key == 'NITJ':
                sides = np.array([p.stem.split('_')[4] for p in paths])
                results[f'NITJ_RvsL|{variant}'] = right_vs_left(emb, subjects, sides)
        for k in [k for k in results if k.split('|')[0] in (key, f'{key}_RvsL')]:
            m = results[k]
            print(f'{k:<26} EER {m["EER"]:6.2f}  CRR {m["CRR"]:6.2f}  VER@0.1% {m["VER@0.1%"]:6.2f}  '
                  f'VER@1% {m["VER@1%"]:6.2f}  AUC {m["AUC"]:.4f}  DI {m["DI"]:.3f}', flush=True)
        ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=1))


if __name__ == '__main__':
    main(sys.argv[1:] or KEYS)
