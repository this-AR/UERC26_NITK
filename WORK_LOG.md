# Work Log: Ear Recognition Evaluation (ITW workspace)

A running history of the work in `C:\Users\hp\Downloads\ITW`. A new entry is added at the end after every task. Each entry records what was done, where the files are (or that they were deleted), the results, and what comes next.

---

## 2026-10-21 · 1. Environment setup
- **What:** got the `scripts_d_t_2 (Matching Results)` shell/awk pipeline running on Windows with the GPU.
- **Done:**
  - Installed `torch 2.11.0+cu128`, timm, sklearn, opencv and matplotlib into `.venv` (RTX 4060).
  - Installed gnuplot 6.0.4.
  - Added `calc` and `bc` replacements, because `calc` was opening the Windows Calculator.
- **Files:** `dt2_tools/bin/`, `dt2_tools/env.sh`
- **Result:** the original `max_min.sh` and `main.sh` run unmodified. The pruned TinyViT `best_pruned_35per_tinyvit.pt` has 7.526 M parameters.

## 2026-10-21 · 2. Full matching evaluation, TinyViT, 4 datasets
- **What:** all-vs-all evaluation on AWE, ITW, UBEAR (`cropped_img`) and NITJ.
  - Pipeline: embed → L2 → subtract global mean → L2.
  - Score: 1 − cosine, then `max_min.sh` and `main.sh`.
  - Left and right ears count as the same person.
- **Code:** `dt2_tools/make_scores.py`, `extras.py`, `finalize.py`, `run_all.sh`
- **Results (EER % / CRR %):**

  | Dataset | EER (%) | CRR (%) |
  |---|---|---|
  | AWE | 6.005 | 96.60 |
  | ITW | 10.055 | 78.64 |
  | UBEAR | 23.171 | 98.39 |
  | NITJ | 15.963 | 88.96 |

- **Note:** ITW `test/106` contains 10 duplicate images, so `max_min.sh` dropped 20 zero-distance rows.
- **Files:** results were in `dt2_results/<KEY>` and zipped to `<KEY>_dt2_results.zip`. All were uploaded to Google Drive by the user and then **deleted locally**. Only `dt2_results/combined/` (summary and combined ROC) remains.

## 2026-10-21 · 3. Comparison with the earlier results
- **What:** compared the earlier Python evaluation (L2-only and mean-centred) with the scripts results.
- **File:** `results_comparison.md`
- **Results:**
  - Mean-centring helps on every dataset.
  - The scripts reproduce the earlier mean-centred EER to within 0.01 points.
  - NITJ earlier result (right-ear probes vs left-ear gallery, L2-only) is reproduced exactly: EER 19.81, CRR 56.73. Mean-centred under the same protocol gives EER 17.31.
- **Note:** the earlier per-run result folders were later deleted (they're on Drive), so the plot links in this file no longer resolve.

## 2026-10-21 · 4. Git
- **What:** created a local git repo and pushed the code only (no data, weights or results) to the new branch `adi_ear_matching_eval` on `github.com/this-AR/UERC26_NITK`, based on `main`.
- **Details:** sparse checkout keeps main's folders out of the local workspace. `.gitignore` excludes images, CSVs, `.pt`, `.npy`, zips and result folders.

## 2026-10-21 · 5. Clean-up
- Deleted the AWE, ITW, UBEAR and NITJ result folders and zips after the user uploaded them.
- Deleted the earlier `l2_normalized` and `raw_mean_centered_l2` result folders; the evaluator `.py` files are kept.

## 2026-10-21 · 6. t-SNE and UBEAR failure analysis (TinyViT)
- **Files:**
  - `analysis/README.md` (the write-up)
  - `analysis/tsne/*.png`
  - `analysis/ubear/report.txt` and its figures
  - code: `dt2_tools/tsne_plots.py`, `ubear_failure_analysis.py`
- **t-SNE:** AWE subjects form tight clusters. UBEAR is organised by head pose more than by identity.
- **Why UBEAR fails:**
  1. **Head pose (main cause):** same-pose genuine EER 1.8 %, different-pose 19.7 %. Outwards-pose images are edge-on slivers.
  2. **CRR inflated by near-duplicate video frames:** for 94 % of probes the best match is another frame from the same video. Excluding the probe's own video gives CRR 69.7 %.
  3. **Left and right ears counted as one identity:** treating each ear separately gives EER 19.2 % (from 23.2 %). Mirroring doesn't help.
  4. **Session change:** cross-session EER 24.3 % against 17.6 % within a session.
- **Ruled out:** image quality and aspect ratio (padding instead of stretching made results worse).

## 2026-10-21 · 7. HPC access
- **Connection:** SSH to `fcse_shwetasharma@172.16.0.77` (AI-server, H100 NVL, Slurm) with key `~/.ssh/hpc_claude`.
- **Read-only checks:**
  - `~/friend-gpu/run.sh` (GPU lending from "Sam") is reviewed and its status is `normal`.
  - `/home` has 23 GB free.
- **On hold:** user said **don't lend the GPU now**, so nothing has been run there.

## 2026-10-22 · 8. New model evaluation: `best_model.pt` (Swin-V2-B), in progress
- **Model:** torchvision `swin_v2_b` + Linear(1024→512) + BatchNorm, 87.4 M parameters, AdaFace-trained (epoch 15, val VER 0.877). Saved with `torch.compile`, so its keys carry the `_orig_mod.` prefix.
- **Preprocessing:** follows the training notebook `uerc26_Model_training_(single_phase).ipynb`: resize 224, RGB, ImageNet mean/std, and the model upsamples to 256.
  - Without CLAHE is better. With CLAHE: AWE EER 2.84 vs 2.65, ITW 7.82 vs 7.50.
- **Code:**
  - `dt2_tools/models.py`: model registry; `DT2_MODEL=swinv2` selects the new model.
  - `model_metrics.py`: quick sklearn metrics.
  - `compile_model_comparison.py`: compiles the results file.
- **Results so far (mean-centred):**
  - AWE: EER 2.65, CRR 99.40.
  - ITW: EER 7.50, CRR 85.79.
  - UBEAR (scripts cross-check): EER 18.35, Rank-1 98.93.
  - NITJ: running.
- **Instruction from the user:** metrics only. Keep one detailed metrics file and delete all intermediate files once the run finishes.
- **Expected output:** `results_swinv2_vs_tinyvit.md`.

## 2026-10-22 · 9. Housekeeping: embeddings removed, analysis results on GitHub
- **Embeddings deleted (about 65 MB):**
  - the CLAHE-test caches;
  - the Swin-V2 AWE and ITW caches;
  - the t-SNE coordinate files (`*_tsne_xy.npy`);
  - per-dataset `embeddings_*.npy` in `dt2_results_swinv2/`.
- **Still kept until run 8 finishes:** the TinyViT caches `analysis/embeddings/{AWE,ITW,UBEAR,NITJ}.npz` and `dt2_results_swinv2/NITJ/embeddings_final.npy`. They will be deleted afterwards.
- **Pushed to `adi_ear_matching_eval`, same folder structure:**
  - code;
  - `WORK_LOG.md`;
  - `analysis/README.md`;
  - the t-SNE plots (`analysis/tsne/*.png`);
  - the UBEAR analysis figures and numbers (`analysis/ubear/`);
  - the TinyViT summary and combined ROC (`dt2_results/combined/`).
- **`.gitignore`:** result figures are now versioned. Datasets, weights, embeddings, pair-score files and zips stay ignored.

## 2026-10-22 · 10. New model `best_model.pt` (Swin-V2-B): evaluation complete
- **What:** the full `scripts_d_t_2` evaluation on all 4 datasets with the new model, compared against the TinyViT.
  - Pipeline: mean-centred (embed → L2 → subtract global mean → L2), all-vs-all pairs.
  - Run settings: `DT2_MODEL=swinv2 DT2_NO_HTML=1 DT2_FINALIZE=0 bash run_all.sh`, then `model_metrics.py` for both models, then `compile_model_comparison.py`.
- **Main file:** `results_swinv2_vs_tinyvit.md` (all metrics, both models, both embedding variants, the NITJ right-vs-left protocol, and the deltas).
- **Results from the scripts (EER % / CRR %):**

  | Dataset | Swin-V2 EER | Swin-V2 CRR | TinyViT EER | TinyViT CRR | ΔEER (pts) |
  |---|---|---|---|---|---|
  | AWE | 2.645 | 99.40 | 6.005 | 96.60 | −3.36 |
  | ITW | 7.515 | 85.62 | 10.055 | 78.64 | −2.54 |
  | UBEAR | 18.356 | 98.93 | 23.171 | 98.39 | −4.81 |
  | NITJ | 11.134 | 93.05 | 15.963 | 88.96 | −4.83 |

  - NITJ right-ear vs left-ear protocol: EER 12.25 % (TinyViT 17.31 %), CRR 75.27 % (TinyViT 56.98 %).
  - VER@1 % improves by 8 to 13 points; AUC improves on every dataset.
- **Checks:**
  - The TinyViT metrics recomputed in this run match entry 2 exactly.
  - ITW again drops 20 duplicate-image rows in `max_min.sh`, so its CRR is 85.62 % against a sklearn Rank-1 of 85.79 %.
- **Files:**
  - Pushed: `dt2_results_swinv2/<KEY>/` (`.sts`, plots, CRR/hist/FAR-FRR `.dat`, `.FR`, `.FA`, `run_info`, `cross_check`, `id_map`, global mean) and `dt2_results_swinv2/combined/ROC_all_datasets.png`.
  - Not pushed: the pair-score files (`*.txt`/`*.txtf`, up to 2.7 GB) and `NITJ.FA` (260 MB, over GitHub's 100 MB limit). Both were deleted locally, since they can be regenerated with the run command above.
  - Also deleted: the embedding caches in `analysis/embeddings/` and `NITJ/embeddings_final.npy`, the duplicate script copies in each `run/`, and `run_swinv2.log`.
- **Next:** the user may want the t-SNE and UBEAR analysis repeated with the new model, as metrics only.
