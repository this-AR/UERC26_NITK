# t-SNE Visualisation and UBEAR Failure Analysis

Model: `best_pruned_35per_tinyvit.pt` (7.526 M parameters). Embeddings are processed as embed → L2 → subtract global mean → L2, the same as the main evaluation. Scores are cosine similarity. Code: `dt2_tools/tsne_plots.py`, `dt2_tools/ubear_failure_analysis.py`. Full numeric output: `ubear/report.txt`, `ubear/results.json`.

## 1. t-SNE of each dataset (`tsne/`)

Settings: perplexity 30, cosine metric, PCA initialisation. Every image is shown in grey, with 8 random subjects highlighted (more than 8 subject colours can't be told apart).

| Dataset | Plot | What it shows |
|---|---|---|
| AWE | `tsne/AWE_tsne.png` | Tight, well-separated subject clusters. A few subjects split into 2–3 sub-clusters, matching its low EER (6.0 %). |
| ITW | `tsne/ITW_tsne.png` | Compact clusters, but more subjects are spread out, matching EER 10.1 % and CRR 78.6 %. |
| NITJ | `tsne/NITJ_tsne.png` | Most subjects form one tight cluster plus a second small group or scattered points elsewhere. This fits each subject having two ears (left-vs-right CRR is only 57 %), but the split has not been confirmed by colouring the points by side. |
| UBEAR | `tsne/UBEAR_tsne.png` | Subjects break into many small fragments. The pose panel shows the map is organised by **head pose** (O and U form their own regions) more than by identity. Side and session are mixed throughout. |

`*_tsne_8subjects.png` shows only the highlighted subjects, so their clusters are easy to read.

## 2. Why UBEAR fails

The headline numbers conflict: **EER 23.2 % (worst of the four datasets) but CRR 98.4 % (best)**. The tests below explain both.

### Summary of tests

| Test | Setting | EER (%) | CRR (%) | Other |
|---|---|---|---|---|
| T0 Baseline | identity = subject | 23.17 | 98.39 | VER@1% 36.5, AUC 0.852 |
| T4 Pose | genuine pairs with the **same** pose | **1.81** | | mean genuine 0.885 |
| T4 Pose | genuine pairs with a **different** pose | **19.66** | | mean genuine 0.366 |
| T1 Ear side | same-side genuine only | 19.09 | | mean 0.422 |
| T1 Ear side | left-vs-right genuine only | 26.75 | | mean 0.236 |
| T1 Ear side | identity = subject + side | 19.16 | 98.14 | VER@1% 48.6, AUC 0.890 |
| T2 Mirror left ears | identity = subject | 23.13 | 98.25 | L–R genuine mean 0.236 → 0.250 only |
| T3 Session | same-session genuine | 17.55 | | mean 0.458 |
| T3 Session | cross-session genuine | 24.27 | | mean 0.274 (only 21 of 126 subjects have session 2) |
| T5 Video frames | gallery excludes the probe's own video sequence | 24.17 | **69.65** | 94.2 % of rank-1 matches were frames from the same video |
| T5 Video frames | gallery = other session only | | 57.13 | |
| T5 Video frames | gallery = opposite ear only | | 45.44 | |
| T7 Input geometry | letterbox (keep aspect ratio) instead of stretching | 27.11 | 98.03 | worse; the model was trained on stretched input |
| T7 Input geometry | letterbox + mirror | 26.93 | 98.07 | worse |

### Problem 1: head-pose variation (the main cause)
UBEAR records each ear under 5 head poses: M (middle), U (up), D (down), T (towards) and O (outwards).
- **Same pose:** genuine pairs separate almost perfectly from impostors (EER 1.8 %).
- **Different pose:** EER jumps to 19.7 %.

Mean genuine cosine by pose pair (`ubear/pose_matrix.png`):

| | M | U | D | T | O |
|---|---|---|---|---|---|
| **M** | 0.90 | 0.44 | 0.52 | 0.45 | 0.39 |
| **U** | 0.44 | 0.92 | 0.26 | 0.26 | 0.29 |
| **D** | 0.52 | 0.26 | 0.89 | 0.32 | 0.33 |
| **T** | 0.45 | 0.26 | 0.32 | 0.89 | **0.14** |
| **O** | 0.39 | 0.29 | 0.33 | **0.14** | 0.78 |

- **T vs O** genuine pairs average 0.14, barely above the impostor mean (0.00 ± 0.16).
- **O (outwards) images are partly unusable.** The ear is seen almost edge-on, so many crops are thin slivers (median h/w 1.98, 3.6 % of them taller than 3:1) with very little ear visible. See `ubear/worst_genuine_pairs.png`.
- **U (up) crops have a very different shape:** median h/w 0.94 (wide), against about 1.5 for the other poses.
- **The model picks up pose as well as identity.** Among the hardest impostors (top 0.1 %), 57.7 % share the probe's pose, against a base rate of 21.6 %.

### Problem 2: left and right ears counted as one identity
- Half of all genuine pairs (50.5 %) compare a left ear with a right ear. These average 0.24, against 0.42 for same-side pairs.
- **Mirroring the left ears doesn't help** (0.236 → 0.250, EER 23.17 → 23.13). The two ears really do look different to the model; it isn't just a reflection problem.
- Treating each ear as its own identity lowers EER from 23.2 % to 19.2 % and raises VER@1% from 36.5 % to 48.6 %.
- NITJ shows the same effect: right-vs-left EER 17.3 %, against 16.0 % for all-vs-all.

### Problem 3: CRR is inflated by near-duplicate video frames
- UBEAR images are frames from short videos: 1,296 sequences, each one subject + session + side + pose.
- For 94 % of probes, the closest match is another frame from the **same video**, which is almost the same image.
- That is why CRR reaches 98.4 % even though verification is poor.
- **Removing the probe's own video from the gallery drops CRR to 69.7 %**. Restricting the gallery to the other session gives 57.1 %, and to the opposite ear gives 45.4 %.
- About 70 % is the more honest measure of UBEAR identification.

### Problem 4: session change
- Genuine pairs across the two sessions average 0.27 (EER 24.3 %), against 0.46 (EER 17.6 %) within a session.
- Session 2 images vary more in lighting: brightness sd 47.7 against 37.8 in session 1. The worst pairs (subject s_073) set a bright session-2 image against a dark session-1 image.
- Only 21 of 126 subjects were captured in session 2.

### Not the main cause
- **Sharpness, size and brightness** correlate only weakly with how well an image matches its own subject (|Spearman ρ| ≤ 0.28, `ubear/quality_vs_score.png`).
- Counter-intuitively, the correlations are negative: sharper and brighter images match slightly worse. This most likely reflects pose (the hard poses are captured differently), not quality itself.
- **Aspect ratio:** keeping it (letterboxing) makes results worse (EER 27.1 %), because the model was trained on stretched 224×224 inputs. Keep the current resize.

### Dataset label issues found
- One file has pose code `R`, which isn't one of the 5 poses: `s_124_1_L_19_F_R_028.tiff`.
- 34 files use lower-case gender `f`.
- Neither affects the metrics.

## 3. What would help

1. **Pose robustness (biggest gain).** Train or fine-tune with strong viewpoint augmentation (perspective warps, out-of-plane rotation, random aspect stretch). Alternatively, enrol several poses per subject and match against all of them (multi-template / max-score fusion).
2. **Report UBEAR with an honest protocol.** Exclude same-sequence pairs, or use a cross-session or cross-pose protocol, so that near-duplicate frames don't inflate CRR. Report left and right ears separately.
3. **Treat left and right ears as different identities**, or fuse a person's left-ear and right-ear scores, rather than expecting one ear to match the other.
4. **Handle the O pose:** flag or remove edge-on sliver crops (for example h/w > 3), or give that pose a lower weight in fusion.
5. **Illumination normalisation** (CLAHE or histogram equalisation) for cross-session robustness. This hasn't been tested yet.

## Figures (`ubear/`)
- `score_distributions_by_side.png`: impostor scores against same-side, left-vs-right, and mirrored left-vs-right genuine scores.
- `pose_matrix.png`: mean genuine score for each pose pair.
- `crr_failures.png`: identification failures (probe | best genuine | best impostor).
- `worst_genuine_pairs.png`: lowest-scoring genuine pairs (edge-on O-pose slivers, lighting changes between sessions).
- `quality_vs_score.png`: sharpness and width against genuine score.
