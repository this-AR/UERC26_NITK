# Findings so far and plan for retraining (items 5, 6, 7)

Status on 2026-10-22. Sources:
- `results/results_swinv2_vs_tinyvit.md`
- `results/analysis/README.md` (UBEAR failure analysis)
- `results/analysis/data_overlap.txt`
- `WORK_LOG.md`

All numbers use the mean-centred embeddings (embed → L2 → subtract global mean → L2), all-vs-all matching, and the `scripts_d_t_2` metrics.

## 1. Results in one table (EER % / CRR %)

| Dataset | Swin-V2-B (88 M, trained on UERC) | Pruned TinyViT (7.5 M) | Is it a fair test? |
|---|---|---|---|
| AWE | 2.65 / 99.40 | 6.01 / 96.60 | **No.** All of AWE is inside UERC (see §2.1) |
| ITW | 7.52 / 85.62 | 10.06 / 78.64 | Yes (no UERC overlap) |
| NITJ | 11.13 / 93.05 | 15.96 / 88.96 | Yes |
| NITJ, right-ear vs left-ear | 12.25 / 75.27 | 17.31 / 56.98 | Yes |
| UBEAR | 18.36 / 98.93 | 23.17 / 98.39 | EER yes. **CRR no:** inflated by video frames (see §2.3) |

## 2. What the results tell us

### 2.1 AWE is leaked into the training data
All 1,000 AWE images are byte-identical to UERC26 public subjects 0001–0100: 712 are in train, 178 in val and 110 in test. Swin-V2 was trained on UERC train/val, so its AWE score is an in-domain score, not generalisation.

ITW, UBEAR and NITJ share no images with UERC, checked both by exact bytes and by a near-duplicate hash (`experiments/data_overlap_check.py`). Only these three measure generalisation.

### 2.2 Mean-centring always helps EER
It lowers EER in all 16 model × dataset × protocol rows, by 0.6 to 2.5 points. It helps most on the datasets furthest from UERC: NITJ −2.1 to −2.5, UBEAR −1.1 to −2.2.

The only negatives are CRR changes of 0.1 point or less: 1 probe on AWE, 4 on UBEAR. That's noise.

So each dataset has a strong shared "domain" direction in embedding space. That is a domain shift, and retraining should target it.

### 2.3 Why UBEAR looks abnormal (high CRR, worst EER)
- **CRR ~99% is an artefact.** UBEAR images are frames from 1,296 short videos. For 94% of probes, the best match is another frame from the same video. With the probe's own video excluded from the gallery, CRR is **69.7%** (TinyViT). It is 57.1% with only the other session as gallery, and 45.4% with only the opposite ear.
- **EER is driven by head pose.**
  - Same-pose genuine pairs: EER **1.8%**.
  - Different-pose genuine pairs: **19.7%**.
  - Pose pair T (towards) vs O (outwards): genuine pairs average a cosine of 0.14, close to the impostor average (0.00).
  - O-pose crops are often edge-on slivers.
- **The training recipe explains the pose failure.** `uerc26_Model_training_(single_phase).ipynb` uses only colour augmentation (ColorJitter, RandomGrayscale, CLAHE, sharpening). It has **no geometric augmentation**: no rotation, perspective, scale or crop.
- **The images look very different from the training data.** UERC: tiny colour web crops, median 61×31 px. UBEAR: controlled greyscale frames, median 185×126 px.
- **Smaller factors:**
  - Left vs right ear of one person: genuine mean 0.24 against 0.42 for same-side pairs. Mirroring doesn't help.
  - Session change: EER 24.3% across sessions against 17.6% within one.
  - Image quality: only a weak effect.

### 2.4 Bigger model, better transfer
Swin-V2 beats TinyViT by 2.5 to 4.8 EER points on every fair dataset. It also raises right-vs-left CRR on NITJ from 57% to 75%.

The pruned student loses more under domain shift than in-domain. Improving the teacher and then re-distilling should carry over to the 7 M model.

### 2.5 Left vs right ear is a labelling decision
Every evaluation so far counts a person's left and right ears as one identity. That penalises both models on UBEAR and NITJ. For training, classes should be **subject + side**. At test time, scores from both ears can still be combined at the person level.

## 3. Mistakes to avoid when retraining

| Risk | Rule |
|---|---|
| Testing on AWE or on any UERC subject | Use only clean test sets: ITW, the held-out UBEAR / NITJ subjects, UERC's 131 test-only subjects |
| Splitting UBEAR or NITJ by **image** | Split **by subject**; otherwise neighbouring frames end up in both train and test |
| "Use 20–30% of UBEAR **images**" | Use 20–30% of UBEAR **subjects** for training; test on the remaining 70–80% |
| Training on every frame | Take 3–5 frames per video, covering all 5 poses and both sessions |
| Mixing datasets by raw count | UERC has 179k training images against ~1.3k for UBEAR. Sample by share instead (§4). |
| One class per person across both ears | Use subject + side classes for training |
| Each track using its own split | One frozen, seeded set of split CSVs for Sanskar, Aditya and Madhavi |
| Reporting UBEAR CRR all-vs-all | Report EER, CRR excluding the probe's own video, and per-pose results |

## 4. Proposed data plan

| Dataset | Train | Test (frozen) | Notes |
|---|---|---|---|
| UERC26 public (1,310 subjects, 248,655 images) | existing train/val: 1,179 subjects | existing test-only: 131 subjects | Contains AWE (subjects 0001–0100) |
| UBEAR (126 subjects, 4,412 images, 1,296 videos) | 25–30% of subjects (~32–38), 3–5 frames per video | the other ~88–94 subjects | Seeded, saved as CSV |
| NITJ (510 subjects, 10,200 images) | existing train/val: 459 subjects | existing test-only: 51 subjects | Classes = subject + side |
| ITW (197 subjects, 1,175 images) | none (`itw_dataset.zip` holds only the test set) | all | Clean check against UERC |
| AWE | none | only the 11 subjects in UERC's test split, or drop it | Leaked otherwise |

- **Batch mix by sampling share:** about 60% UERC, 20% NITJ and 20% UBEAR, with a fixed number of images per class in each batch.
- **Augmentation to add:** rotation ±15–20°, perspective, random-resized crop/scale, plus the current colour augmentation. No horizontal flip when classes are subject + side.

## 5. The three work items

### Item 5: Swin-V2 88 M (Sanskar, Aditya)
- **Done:** evaluation on all datasets (§1).
- **Next:**
  1. **Zero-shot baseline on the frozen UBEAR / NITJ test subjects.** No training, and it runs locally.
  2. **Fine-tune on the UERC + UBEAR-train + NITJ-train mix.** Low learning rate (about 1e-5 for the backbone), 5–10 epochs, plus the geometric augmentation. *Recommended first training run.* Needs the HPC.
  3. **UBEAR-only fine-tune** as an upper bound for UBEAR. Freeze the backbone except the last stage + head, or use LoRA. A full fine-tune on about 1k images will make the model forget the other datasets, so measure ITW and NITJ too.
  4. **Retraining from scratch on the mix:** only if step 2 clearly hits a ceiling. It costs about a day on the H100 and is unlikely to beat step 2.

### Item 6: pruned TinyViT 7 M (Aditya)
- **Done:** UBEAR failure analysis and t-SNE (TinyViT).
- **Next:**
  1. Same zero-shot baseline on the frozen UBEAR test subjects.
  2. **Fine-tune on 20–30% of UBEAR subjects, then test on the rest.** Full fine-tuning is fine at 7 M parameters, and it fits the local RTX 4060. Run two versions: UBEAR-only, and UBEAR mixed with a small UERC/NITJ sample, to see how much the model forgets.
  3. **Better long-term:** re-distil from the improved Swin teacher (item 5, step 2), reusing `swin88_to_tiny21`, `tinyvit21_to_11` and the pruning notebooks.
  4. Repeat the t-SNE and failure analysis on the Swin-V2 model and on the fine-tuned models, to show whether pose clusters merge into identity clusters.

### Item 7: DINOv3 and DINOv3 + LoRA (Madhavi)
1. Zero-shot DINOv3 embeddings first, as a reference for how well a general foundation model transfers.
2. Then LoRA on the same mix and frozen splits as item 5. LoRA suits the small UBEAR and NITJ sets.
3. Add DINOv3 to `ear_eval/models.py` so its results come out of the same pipeline and are directly comparable. That needs the DINOv3 weights, which are gated on Hugging Face and require accepting the licence.

## 6. What can run now vs. what needs the HPC

| Task | Where | Notes |
|---|---|---|
| Freeze split CSVs (UBEAR 30% by subject, NITJ, UERC, clean AWE) | local, minutes | Blocks everything else |
| Honest UBEAR protocol in the pipeline (own video excluded, per pose) | local | Metrics only |
| Zero-shot baselines, both models, on the frozen test subjects | local, minutes | Reference point for items 5, 6, 7 |
| Swin-V2 t-SNE + UBEAR failure analysis | local, ~10 min | Same tests as for TinyViT |
| TinyViT fine-tune on 30% of UBEAR (item 6) | local RTX 4060 | ~1.3k images, about minutes per epoch |
| Swin-V2 partial fine-tune or LoRA on UBEAR | local, possible | 8 GB GPU: small batch with bf16 / gradient checkpointing; slow but works |
| Swin-V2 fine-tune on the full UERC mix | **HPC** | 179k images. HPC is not reachable right now, and the GPU must be lent from Sam |
| Swin-V2 retrain from scratch | **HPC** | Only if needed |
| DINOv3 zero-shot / LoRA | local (ViT-B scale) or HPC | Needs the gated weights |
