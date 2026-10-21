#!/bin/bash
# Full scripts_d_t_2 matching evaluation for the given datasets (default: all four).
#
#   DT2_MODEL=swinv2 bash pipeline/run_matching.sh [AWE ITW UBEAR NITJ]
#
# Per dataset:
#   1. pipeline/make_scores.py   embeddings -> mean-centred -> all-vs-all data file
#   2. matching_scripts/max_min.sh -> main.sh   CRR, EER, DI, histograms, ROC (unmodified scripts)
#   3. pipeline/postprocess.py   GIBA + FAR/FRR plots, sklearn cross-check, keep metrics + images only
# Results land in results/<model>/<KEY>/. Env: DT2_MODEL, DT2_CLAHE, DT2_HTML=1, DT2_KEEP_ALL=1.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
. "$ROOT/tools/env.sh"
DATASETS=${@:-AWE ITW UBEAR NITJ}
cd "$ROOT"

for key in $DATASETS; do
    echo "==================== $key ($(date +%T))"
    $PYTHON -m pipeline.make_scores "$key" | grep -v "embeddings computed\|rows \["
    work=$($PYTHON -c "from ear_eval.config import result_dir; print(result_dir('$key').as_posix())")/_work
    work=$(cygpath -u "$work")
    cp "$ROOT/matching_scripts/"* "$work/"
    chmod +x "$work/"*.sh
    echo "max_min.sh + main.sh started ($(date +%T))"
    (cd "$work" && ./max_min.sh "$key.txt" > max_min.log 2>&1)
    echo "pairs in: $(wc -l < "$work/$key.txt")  after max_min filter: $(wc -l < "$work/$key.txtf")"
    echo "postprocess started ($(date +%T))"
    $PYTHON -m pipeline.postprocess "$key"
done
echo "ALL DONE ($(date +%T))"
