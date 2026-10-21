# Ear recognition: matching evaluation (AWE, ITW, UBEAR, NITJ)

This workspace evaluates ear-recognition models with the **`scripts_d_t_2` matching scripts**. Those scripts report CRR, EER (where FAR = FRR), the decidability index (DI), genuine/impostor histograms and ROC curves.

Each dataset is evaluated all-vs-all: every image is compared with every other image. Embeddings are processed as **embed → L2 → subtract global mean → L2**.

Two models are supported:
- the pruned TinyViT, 7.53 M parameters;
- Swin-V2-B (`best_model.pt`), 87.4 M parameters.

The history of the work, with every result, is in [WORK_LOG.md](WORK_LOG.md).

## Layout

```
ear_eval/            shared library
  config.py          every path + model selection (DT2_MODEL, DT2_CLAHE)
  datasets.py        image discovery for AWE / ITW / UBEAR / NITJ
  models.py          model registry (tinyvit, swinv2) and input preprocessing
  embeddings.py      embedding extraction, L2 / mean-centring, cache (.cache/)
  metrics.py         EER, VER@FAR, AUC, DI, rank-1 / CRR
  plotting.py        palette, figure styling, gnuplot runner
pipeline/            the scripts_d_t_2 evaluation
  run_matching.sh    runs the three steps below for each dataset
  make_scores.py     1. embeddings -> all-vs-all data file (main.sh format)
  postprocess.py     3. GIBA + FAR/FRR plots, sklearn cross-check, keep metrics + images only
matching_scripts/    2. original shell/awk scripts (max_min.sh -> main.sh), unchanged
experiments/         analyses built on the library
  model_metrics.py   quick metrics: L2 vs mean-centred, NITJ right-vs-left protocol
  compare_models.py  results/results_swinv2_vs_tinyvit.md + combined ROC
  tsne_plots.py      t-SNE per dataset
  ubear_failure_analysis.py   why UBEAR fails (pose, ear side, session, video frames, quality)
tools/               env.sh, calc / bc stand-ins for the shell scripts
legacy/              earlier evaluators and the UERC26 starter kit (kept for reference)
results/             metrics and figures only
  tinyvit/           summary.txt + combined ROC (TinyViT)
  swinv2/<KEY>/      <KEY>.sts, plots, FAR/FRR data, run_info.txt, cross_check.txt
  analysis/          model_metrics_*.json, tinyvit/tsne, tinyvit/ubear, README.md
  results_swinv2_vs_tinyvit.md, results_comparison.md
```

The data and weights stay where they were delivered:
- datasets: `awe/`, `ITW/ITW_230_split_augment/test/`, `cropped_img/` (UBEAR) and `NITJ_Cropped/Cropped/`;
- weights: `ITW/ITW_230_split_augment/best_pruned_35per_tinyvit.pt` and `best_model.pt`.

All of these paths are defined in `ear_eval/config.py`.

## Setup

1. Create the environment with `uv venv` and install the packages from `requirements.txt` (the CUDA build of torch).
2. Install gnuplot (`winget install gnuplot.gnuplot`).
3. Run everything from the project root in Git Bash.

The matching scripts call `calc`; `tools/bin` provides awk-based stand-ins for `calc` and `bc`. `pipeline/run_matching.sh` sources `tools/env.sh` itself.

## Running

```bash
# Full matching evaluation (results/<model>/<KEY>/), default model tinyvit
DT2_MODEL=swinv2 bash pipeline/run_matching.sh AWE ITW UBEAR NITJ

# Quick metrics for both embedding variants -> results/analysis/model_metrics_<model>.json
DT2_MODEL=swinv2 .venv/Scripts/python.exe -m experiments.model_metrics

# Model comparison report (needs both model_metrics JSONs and results/swinv2/)
.venv/Scripts/python.exe -m experiments.compare_models

# t-SNE and the UBEAR failure analysis -> results/analysis/<model>/
.venv/Scripts/python.exe -m experiments.tsne_plots
.venv/Scripts/python.exe -m experiments.ubear_failure_analysis
```

Pipeline options:
- `DT2_KEEP_ALL=1` keeps the pair-score files, FA/FR lists and histograms in `_work/`. NITJ needs about 10 GB for these.
- `DT2_HTML=1` adds the FA, FR and CRR image galleries.
- `DT2_CLAHE=1` (Swin-V2 only) applies the notebook's CLAHE preprocessing.

## Results (EER % / CRR %, scripts_d_t_2, mean-centred)

| Dataset | Swin-V2 | TinyViT |
|---|---|---|
| AWE | 2.65 / 99.40 | 6.01 / 96.60 |
| ITW | 7.52 / 85.62 | 10.06 / 78.64 |
| UBEAR | 18.36 / 98.93 | 23.17 / 98.39 |
| NITJ | 11.13 / 93.05 | 15.96 / 88.96 |

Full tables are in [results/results_swinv2_vs_tinyvit.md](results/results_swinv2_vs_tinyvit.md). The UBEAR findings are in [results/analysis/README.md](results/analysis/README.md).
