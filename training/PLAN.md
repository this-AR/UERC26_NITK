# Training plan (all training runs on the HPC)

Agreed on 2026-10-22. TinyViT re-distillation is **dropped**. Data: the frozen splits in `splits/manifest.csv` (see `splits/README.md`).
- **Train** = UERC train + NITJ train + UBEAR train (31 subjects).
- **Validation** = UERC val, NITJ val, UBEAR val (6 unseen subjects).
- **Test** = UERC test (131 subjects), NITJ test (51), UBEAR test (89), ITW (197) and the 11 clean AWE subjects. Test is never used for training or model selection.

## Experiments

| # | Model | Starts from | Training data | What is trained | Purpose |
|---|---|---|---|---|---|
| B1 | Swin-V2-B | current `best_model.pt` (UERC-trained) | none | nothing | zero-shot baseline on the frozen test sets |
| B2 | DINOv3 | official pretrained weights | none | nothing | zero-shot baseline |
| **F1** | Swin-V2-B | `best_model.pt` | UBEAR train only | whole model, low LR | fine-tune on UBEAR alone: best-case UBEAR, and how much it forgets |
| **F2** | Swin-V2-B | `best_model.pt` | UBEAR + NITJ train + UERC train (replay) | whole model, low LR | fine-tune on UBEAR and the other datasets |
| **R1** | Swin-V2-B | ImageNet weights (as in the original notebook) | all train data: UERC + NITJ + UBEAR | whole model | full retrain on every dataset |
| **R2** | DINOv3 | official pretrained weights | all train data | whole model | full retrain, DINOv3 |
| **L1** | Swin-V2-B | `best_model.pt` | all train data | LoRA adapters + head (backbone frozen) | cheaper alternative to F2 / R1 |
| **L2** | DINOv3 | official pretrained weights | all train data | LoRA adapters + head | foundation model with adapters (item 7) |

Order: B1, B2 → F1 (quick, checks the training code) → F2 → R1, R2 → L1, L2.

## Shared settings
- **Loss:** AdaFace, as in `uerc26_Model_training_(single_phase).ipynb`. Classes are subject + ear side where the side is known: UERC 1,179, NITJ 918, UBEAR 62.
  - F runs keep the trained UERC classifier rows and add new rows for NITJ and UBEAR.
- **Augmentation:** the notebook's colour augmentation, **plus geometric**: rotation ±15–20°, perspective, random-resized crop/scale. No horizontal flip, because left and right ears are separate classes.
- **Batch mix** for multi-dataset runs: about 60% UERC, 20% NITJ, 20% UBEAR by sampling share, with a fixed number of images per class in each batch.
- **Precision and memory:** bf16, batch about 64, to stay inside the 22 GB GPU budget lent by `~/friend-gpu/run.sh`.
- **Learning rate:**
  - fine-tune (F): backbone about 1e-5, head about 1e-3;
  - full retrain (R): as in the notebook (backbone 1e-4, head 5e-4, cosine schedule);
  - LoRA: adapters about 1e-4.
- **Checkpoints:** keep only the best, chosen by validation verification. `/home` on the HPC has about 23 GB free.

## Evaluation (identical for every model)
- The `scripts_d_t_2` pipeline (`pipeline/run_matching.sh`) on each test set: CRR, EER, DI, ROC.
- sklearn VER@0.1%, VER@1%, AUC.
- UBEAR also: CRR with the probe's own video excluded, and EER per pose.

**Success:**
- UBEAR test EER below the B1 baseline, and cross-video CRR up.
- ITW, NITJ test and UERC test no more than about 0.5 EER points worse than B1 (no forgetting).

## Data preparation (on the HPC, all CPU cores)
1. Copy `hpc_package/ear_data.tar` (1.42 GB, built by `python -m training.pack_for_hpc`) and the code to `~/friend-gpu/`. Run `tar -xf ear_data.tar`.
2. Run `python -m training.preprocess --root . --out data_224 --workers <cores>`.
   - Every image becomes **224 × 224, 3-channel RGB, WebP q95**: stretched, no padding, no augmentation baked in.
   - It writes `data_224/manifest.csv`. All training **and** evaluation read from `data_224/`, so every model sees identical inputs.
   - Size: about 5 KB per image, about 1.4 GB in total. Checked locally on 200 images from all five datasets.
3. Check that the image count equals the manifest rows, then delete `ear_data.tar` and the original images. Only `data_224/` stays.

## Results policy (to keep HPC disk free)
- **For every method, keep only the metrics.** One `results/<method>.md` (and/or `.txt`) per run, with the evaluation tables.
- Delete each run's embeddings, pair-score files, FA/FR lists and plots from the HPC **as soon as its metrics file is written**.
- Checkpoints: keep only the current best per method while it is needed for evaluation. Ask before deleting a trained model.
- Copy the metrics files back to this repo (`results/`), add a `WORK_LOG.md` entry, and commit.

## Decisions
- **DINOv3: ViT-B/16 (86 M parameters)**, about the same size as Swin-V2-B. Its weights are gated on Hugging Face; Meta's licence must be accepted before download.
- **Item 6** (TinyViT fine-tuned on 30% of UBEAR): only if time allows, after the runs above.
