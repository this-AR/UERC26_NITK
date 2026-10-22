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

## 2026-10-22 · 11. Clean-up and codebase restructure
- **Space:** results went from 87 MB to 5.9 MB.
  - Deleted: FA/FR lists, intermediate `.dat` files (CRR, GIBA, histograms), `id_map.tsv`, global-mean vectors, `max_min.log`, every embedding cache, and the old `dt2_tools/`, `analysis/`, `dt2_results*/` folders.
  - Kept: metric text files (`.sts`, `run_info.txt`, `cross_check.txt`, `summary.txt`, `report.txt`, JSON), images, and `<KEY>_FAR_FRR_Data.dat` (the ROC data needed for the combined ROC).
- **New layout** (see `README.md`). Data folders and weights did not move.

  | Old | New |
  |---|---|
  | `dt2_tools/datasets.py`, `models.py`, `analysis_common.py` | `ear_eval/datasets.py`, `models.py`, `embeddings.py` (+ new `config.py`, `metrics.py`, `plotting.py`) |
  | `dt2_tools/run_all.sh`, `make_scores.py`, `extras.py` | `pipeline/run_matching.sh`, `make_scores.py`, `postprocess.py` |
  | `dt2_tools/model_metrics.py`, `compile_model_comparison.py`, `tsne_plots.py`, `ubear_failure_analysis.py` | `experiments/model_metrics.py`, `compare_models.py`, `tsne_plots.py`, `ubear_failure_analysis.py` |
  | `dt2_tools/finalize.py` (relocation + zips) | removed (no longer needed) |
  | `scripts_d_t_2 (Matching Results)/scripts_d_t_2 (Matching Results)/` | `matching_scripts/` (unchanged contents) |
  | `dt2_tools/env.sh`, `bin/` | `tools/env.sh`, `tools/bin/` |
  | root `evaluate*.py`, `extract_features.py`, `score.py`, `uerc26_*.py` | `legacy/uerc26_starter/`, `legacy/evaluate_pairs_csv.py` |
  | `*/itw_full_pair_evaluation/evaluate_itw_all_pairs.py`, `awe_result/evaluate_awe.py`, `ubear_results/evaluate_cropped_img.py` | `legacy/full_pair_eval/evaluate_all_pairs.py`, `evaluate_awe.py`, `evaluate_ubear.py` (outputs now go to `results/legacy/<dataset>/`) |
  | `dt2_results/combined/` | `results/tinyvit/` |
  | `dt2_results_swinv2/<KEY>/run/<KEY>/` | `results/swinv2/<KEY>/` (flat) |
  | `analysis/{tsne,ubear}`, `analysis/*.json`, `analysis/README.md` | `results/analysis/tinyvit/{tsne,ubear}`, `results/analysis/*.json`, `results/analysis/README.md` |
  | `results_*.md` | `results/results_*.md` |

- **Code changes:**
  - Every path is now in `ear_eval/config.py`.
  - The metric code that had been copied into three scripts is now one module, `ear_eval/metrics.py`.
  - The pipeline writes to `results/<model>/<KEY>/_work/` and tidies itself, keeping only metrics and images. `DT2_KEEP_ALL=1` keeps the scratch files; `DT2_HTML=1` adds the galleries.
  - Entry points run as modules from the project root, for example `python -m experiments.model_metrics`.
  - Added a top-level `README.md` and `requirements.txt`.
- **Verification:**
  - `DT2_MODEL=swinv2 bash pipeline/run_matching.sh AWE` reproduces `AWE.sts` byte-for-byte.
  - Swin-V2 quick metrics are identical, and every file compiles.
  - Re-embedding TinyViT gave tiny float differences (VER@0.1 % 62.00 vs 62.04, from a different batch size), so the recorded TinyViT JSON was kept.

## 2026-10-22 · 12. UERC26 inspection, data-leakage check, retraining brainstorm
- **UERC26 data added at `UERC26_data/data/`** (git-ignored).
  - `public/`: 248,655 images (234,651 jpg + 14,004 png) of 1,310 subjects, 10 to 645 images each (median 150). Images are tiny web crops: median 61×31 px, h/w 1.82.
  - Split: 1,179 subjects shared between train (179,172 images) and val (44,804); 131 subjects are test only (24,679 images). `pairs_{train,val,test}.csv` are also provided.
  - `sequestered_anonymized/`: 1,670 unlabelled images.
- **Leakage found:** all **1,000 AWE images are byte-identical to UERC public subjects 0001–0100** (712 train, 178 val, 110 test). The Swin-V2 model was trained on UERC train/val, so **its AWE results (EER 2.65 %) are in-domain and not a generalisation result.** ITW, UBEAR and NITJ have no overlap with UERC (checked by exact bytes and a 16×16 average hash).
- **Training recipe** (`uerc26_Model_training_(single_phase).ipynb`): colour-only augmentation (ColorJitter, RandomGrayscale 0.2, CLAHE, sharpening). There is **no geometric augmentation** (no rotation, perspective, scale or crop), which fits UBEAR's pose failure.
- **NITJ split CSV:** 459 subjects in train/val and 51 test-only subjects, the same scheme as UERC.
- **Mean-centring:** lowers EER in all 16 model × dataset rows. Its only negatives are CRR changes of 0.1 point or less (1 AWE probe, 4 UBEAR probes), which is noise.
- **Brainstorm and plan:** in the chat answer of this date. Its points: clean subject-disjoint test splits first, UBEAR training data sampled by subject and video sequence, a domain-balanced mix, and partial versus full fine-tuning. **Nothing has been retrained yet**; waiting for the user's decision.

## 2026-10-22 · 13. Analysis results saved; HPC unreachable
- **Saved:**
  - `results/analysis/findings_and_plan.md`: inferences, mistakes to avoid, data plan, next steps for items 5/6/7, and what can run locally vs. on the HPC.
  - `results/analysis/data_overlap.txt`: AWE fully inside UERC26 public; ITW, UBEAR and NITJ clean.
  - `experiments/data_overlap_check.py`: reusable leakage check.
  - `ear_eval/config.py`: new `UERC_PUBLIC_DIR`.
- **ITW:** `itw_dataset.zip` holds only the same 197-subject test set, so locally ITW is test-only.
- **HPC:** `172.16.0.77` timed out (laptop now on `172.31.99.243`, likely off the campus network or VPN), and the ngrok tunnel was refused. Nothing was running there.
- **Next:** waiting for the user to pick from §6 of the plan. The first suggested step is to freeze the split CSVs, then run zero-shot baselines.

## 2026-10-22 · 14. Decision: training happens on the HPC only
- All training (items 5, 6 and 7) will run on the HPC (H100, borrowed through `~/friend-gpu/run.sh`). The local RTX 4060 is used only for evaluation and analysis, and for preparing training code and split files.
- **Local preparation possible now:** frozen split CSVs, the fair UBEAR measures, zero-shot baselines, Swin-V2 t-SNE and failure analysis, and the training scripts.
- **Blocked:** the HPC is unreachable until the laptop is back on the campus network or VPN. The GPU is borrowed only after the user says go.

## 2026-10-22 · 15. Frozen training/test splits and the HPC data package
- **Code:**
  - `training/make_splits.py` → `splits/manifest.csv` (264,552 rows, one per image) and `splits/README.md`, seed 42.
  - `training/pack_for_hpc.py` → `hpc_package/ear_data.tar` (1.42 GB, git-ignored; paths inside match the manifest).
- **Splits (images / subjects):**

  | Dataset | Train | Val | Test |
  |---|---|---|---|
  | UERC | 179,172 / 1,179 | 44,804 / same subjects | 24,679 / 131 |
  | NITJ | 7,344 / 459 (918 subject+side classes) | 1,836 / same subjects | 1,020 / 51 |
  | UBEAR | 1,099 / 31 (62 classes) | 204 / 6 (subject-disjoint) | 3,109 / 89 |
  | ITW | – | – | 1,175 / 197 |
  | AWE | – | – | 110 / 11 (only the subjects in UERC's test split) |

- **UBEAR:**
  - 37 of 126 subjects (29%) are used for training, stratified by gender and session-2 capture. Train and val keep at most 5 evenly spaced frames per video; test keeps every frame.
  - Every split covers all 5 poses. 15 of the 21 session-2 subjects are in the test set.
- **Checks:** the script asserts that no test subject appears in train or val within any dataset, and that every listed file exists.
- **Next:** write the HPC training script, then copy the tar over once the HPC is reachable.

## 2026-10-22 · 16. Training plan agreed
- **Plan:** `training/PLAN.md`.
  - **Dropped:** TinyViT re-distillation.
  - **Kept:**
    - B1/B2: zero-shot baselines, Swin-V2 and DINOv3.
    - **F1/F2:** fine-tune the trained Swin-V2 on UBEAR only, and on UBEAR + the other datasets.
    - **R1/R2:** full retrain on all training data, Swin-V2 and DINOv3.
    - **L1/L2:** LoRA, Swin-V2 and DINOv3.
- **Shared settings:** AdaFace, subject+side classes, geometric augmentation added, a 60/20/20 UERC/NITJ/UBEAR mix, bf16 within the 22 GB budget. Every model is evaluated with the same pipeline on the frozen test splits.
- **Open:** the DINOv3 variant and weights; whether item 6 (TinyViT on 30% of UBEAR) is still wanted.

## 2026-10-22 · 17. Preprocessing script; metrics-only results policy; DINOv3 choice
- **`training/preprocess.py`:** converts every manifest image to **224 × 224 RGB WebP q95** (stretched; greyscale is copied to 3 channels; no augmentation baked in), using multiprocessing across all CPU cores. It skips files already done and writes `data_224/manifest.csv`.
  - Checked locally on 200 images (40 per dataset): all 224×224×3, about 4–8 KB each, so the full set is about 1.4 GB.
- **Incident:** a mistyped test command started converting the full manifest locally. I stopped it and deleted the partial output (about 245k WebP files in the scratchpad). The real preprocessing will run on the HPC's CPU cores.
- **Results policy:** for each method, keep only a metrics md/txt file. Delete embeddings, score files and plots from the HPC as soon as metrics are written. Keep only the best checkpoint per method; ask before deleting trained models.
- **Decisions:** DINOv3 = ViT-B/16 (86 M). Item 6 (TinyViT on 30% of UBEAR) only if time allows.
- **Plan updated:** `training/PLAN.md` (data preparation steps, results policy).
