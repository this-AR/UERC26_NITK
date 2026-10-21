# Ear Recognition – Comparison of Evaluation Methods

Model for every result: `best_pruned_35per_tinyvit.pt` (pruned TinyViT, **7.526 M parameters**, 512-d embedding). Every image is matched against every other image of the same dataset.

## Methods compared

| | Earlier method (existing result folders) | Today’s method (scripts_d_t_2 “Matching Results”) |
|---|---|---|
| Embedding variants | (a) L2-normalized; (b) mean-centred: L2 → subtract global mean → L2 | Mean-centred only: L2 → subtract global mean → L2 |
| Pairs | Unique unordered pairs (i < j) | All ordered pairs (i ≠ j); each pair appears twice |
| Score | Cosine similarity (higher = genuine) | Dissimilarity 1 − cosine, min-max normalized by `max_min.sh` (lower = genuine) |
| Metric code | `uerc26_utils.compute_metrics` (sklearn ROC) | `main.sh`: histogram.sh, roccurve.sh (10 000-step quantization), eer.awk, identification.awk |
| Reports | EER, VER@0.1%, VER@1%, AUC | CRR, EER (FAR = FRR), DI, FA/FR lists, genuine/impostor histogram, ROC |

## Main results table

| Dataset | Method | Pairs | EER (%) | CRR (%) | DI | VER@FAR 0.1% (%) | VER@FAR 1% (%) | AUC |
|---|---|---|---|---|---|---|---|---|
| AWE | Earlier – L2-normalized | 499500 | 6.601 | — | 3.194 | 59.07 | 80.73 | 0.9830 |
| AWE | Earlier – Mean-centred (raw → L2 → −μ → L2) | 499500 | 5.999 | — | 3.312 | 62.09 | 82.31 | 0.9859 |
| AWE | **Today – scripts_d_t_2 (mean-centred)** | 999000 ordered | **6.005** | **96.60** | -1.913 | 62.00 | 82.33 | 0.9859 |
| ITW | Earlier – L2-normalized | 689725 | 10.764 | — | 2.486 | 38.88 | 63.53 | 0.9569 |
| ITW | Earlier – Mean-centred (raw → L2 → −μ → L2) | 689725 | 10.055 | — | 2.605 | 46.25 | 69.73 | 0.9622 |
| ITW | **Today – scripts_d_t_2 (mean-centred)** | 1379450 ordered | **10.055** | **78.64** | -1.570 | 46.28 | 69.76 | 0.9622 |
| UBEAR | Earlier – L2-normalized | 9730666 | 25.344 | — | 1.350 | 19.72 | 33.35 | 0.8292 |
| UBEAR | Earlier – Mean-centred (raw → L2 → −μ → L2) | 9730666 | 23.173 | — | 1.454 | 21.59 | 36.49 | 0.8516 |
| UBEAR | **Today – scripts_d_t_2 (mean-centred)** | 19461332 ordered | **23.171** | **98.39** | -0.734 | 21.59 | 36.49 | 0.8516 |
| NITJ | Earlier – L2-normalized (Right-ear probes vs Left-ear gallery) | 26010000 (R × L) | 19.813 | 56.73 | 1.724 | 23.12 | 44.18 | 0.8809 |
| NITJ | **Today – scripts_d_t_2 (mean-centred)** | 104029800 ordered | **15.963** | **88.96** | -0.708 | 36.60 | 56.56 | 0.9194 |

Notes on the table:
- **CRR** (rank-1 identification) is only produced by today’s scripts; the earlier method did not compute it.
- **DI, earlier method**: (μ_genuine − μ_impostor) / √((σ²_genuine + σ²_impostor)/2), using the genuine/impostor statistics in each `metrics.txt`. Positive because scores are similarities.
- **DI, today**: as printed by `histogram.sh`. It is negative because the scores are dissimilarities, and `stat.awk` averages over histogram bins instead of over pairs, so its size is **not directly comparable** with the earlier DI.
- **VER and AUC, today**: the scripts do not compute these, so they come from `cross_check.txt` (sklearn on today’s embeddings, unique pairs). Its EER agrees with the scripts’ EER to within 0.002 percentage points on every dataset.
- **NITJ, earlier result** comes from a different protocol: 5 100 right-ear probes against a 5 100 left-ear gallery (51 000 genuine, 25 959 000 impostor), L2-only. It is **not directly comparable** with today’s all-vs-all row, which also matches L–L and R–R. See the same-protocol comparison below.

## NITJ – same-protocol comparison (Right-ear probes vs Left-ear gallery)

The earlier NITJ protocol was rerun on today’s embeddings for both variants, so the two methods can be compared directly (sklearn ROC; CRR = rank-1 of each right-ear probe against the left-ear gallery; DI = (μ_g − μ_i)/√((σ²_g + σ²_i)/2)).

| Method | Pairs | Genuine / Impostor | EER (%) | CRR (%) | DI | VER@FAR 0.1% (%) | VER@FAR 1% (%) | AUC | Genuine mean ± sd | Impostor mean ± sd |
|---|---|---|---|---|---|---|---|---|---|---|
| Earlier result (as reported) | 26010000 | 51000 / 25959000 | 19.813 | 56.73 | 1.724 | 23.12 | 44.18 | 0.8809 | 0.5433 ± 0.1738 | 0.2707 ± 0.1407 |
| Today’s embeddings – L2-only | 26010000 | 51000 / 25959000 | 19.817 | 56.73 | 1.724 | 23.10 | 44.18 | 0.8808 | 0.5432 ± 0.1738 | 0.2706 ± 0.1407 |
| Today’s embeddings – **mean-centred** | 26010000 | 51000 / 25959000 | **17.314** | **56.98** | **1.947** | **24.59** | **48.59** | **0.9074** | 0.3709 ± 0.2205 | −0.0068 ± 0.1634 |

- The L2-only row reproduces the earlier result almost exactly, which confirms the earlier NITJ run was L2-only.
- Mean-centring improves every metric under the same protocol: EER −2.50 points, VER@1% +4.41 points, AUC +0.027, DI +0.22.
- Cross-ear matching (right vs left) is much harder than all-vs-all. CRR is 57 % here, against 89 % when same-side images are also in the gallery.

## Today’s method – full script output

| Dataset | Images | Subjects | Genuine (ordered) | Impostor (ordered) | EER (%) | EER threshold (normalized) | FAR (%) | FRR (%) | False rejects | False accepts | CRR (%) | Failed probes | Rank-1 cross-check (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AWE | 1000 | 100 | 9000 | 990000 | 6.005 | 0.4917 | 6.005 | 6.022 | 542 / 9000 | 59454 / 990000 | 96.60 | 34 / 1000 | 96.60 |
| ITW | 1175 | 197 | 6620 | 1372830 | 10.055 | 0.5088 | 10.041 | 10.091 | 666 / 6600 | 137840 / 1372830 | 78.64 | 251 / 1175 | 79.15 |
| UBEAR | 4412 | 126 | 164988 | 19296344 | 23.171 | 0.5471 | 23.168 | 23.175 | 38236 / 164988 | 4470608 / 19296344 | 98.39 | 71 / 4412 | 98.39 |
| NITJ | 10200 | 510 | 193800 | 103836000 | 15.963 | 0.4956 | 15.949 | 15.975 | 30960 / 193800 | 16560528 / 103836000 | 88.96 | 1126 / 10200 | 88.95 |

In ITW, `test/106/` contains 10 byte-identical duplicate images (`xxxx.png` / `xxxx(1).png`). Each duplicate pair has distance 0, and `max_min.sh` drops every row with score ≤ 1e-7. That removed 20 genuine rows (6 620 → 6 600) and is why the ITW CRR (78.64 %) is lower than the cross-check Rank-1 (79.15 %). No rows were dropped for AWE, UBEAR or NITJ.

## Earlier method – score statistics (cosine similarity)

| Dataset | Variant | Genuine pairs | Impostor pairs | EER threshold | Genuine mean ± sd | Impostor mean ± sd |
|---|---|---|---|---|---|---|
| AWE | L2-normalized | 4500 | 495000 | 0.3392 | 0.5857 ± 0.1495 | 0.1295 ± 0.1359 |
| AWE | Mean-centred (raw → L2 → −μ → L2) | 4500 | 495000 | 0.2319 | 0.5214 ± 0.1705 | -0.0057 ± 0.1469 |
| ITW | L2-normalized | 3310 | 686415 | 0.3739 | 0.5713 ± 0.1591 | 0.2035 ± 0.1358 |
| ITW | Mean-centred (raw → L2 → −μ → L2) | 3310 | 686415 | 0.1928 | 0.4585 ± 0.2012 | -0.0030 ± 0.1493 |
| UBEAR | L2-normalized | 82494 | 9648172 | 0.3996 | 0.5405 ± 0.1985 | 0.3154 ± 0.1271 |
| UBEAR | Mean-centred (raw → L2 → −μ → L2) | 82494 | 9648172 | 0.1099 | 0.3279 ± 0.2802 | -0.0029 ± 0.1583 |

## Method comparison (EER, lower is better)

| Dataset | Earlier L2-only | Earlier mean-centred | Today scripts (mean-centred) | Gain from mean-centring |
|---|---|---|---|---|
| AWE | 6.601 | 5.999 | 6.005 | −0.602 pts |
| ITW | 10.764 | 10.055 | 10.055 | −0.709 pts |
| UBEAR | 25.344 | 23.173 | 23.171 | −2.170 pts |
| NITJ (all-vs-all) | — | — | 15.963 | — |
| NITJ (R vs L protocol) | 19.813 (earlier) / 19.817 (today) | 17.314 | — | −2.499 pts |

Findings:
1. **Mean-centring helps on every dataset.** Compared with L2-only it lowers EER and raises VER and AUC on AWE, ITW, UBEAR and NITJ (NITJ under the R vs L protocol: EER 19.81 → 17.31). The biggest gain is at strict operating points: VER@0.1% goes up by 1.9 points on UBEAR, 3.0 on AWE and 7.4 on ITW.
2. **The two evaluation codes agree.** Today’s scripts reproduce the earlier mean-centred EER to within about 0.01 points (AWE 6.005 vs 5.999, UBEAR 23.171 vs 23.173). The small gaps come from the 10 000-step score quantization in `roccurve.sh`, plus the ITW duplicate rows dropped by `max_min.sh`.
3. **The scripts add identification results.** CRR is high on AWE (96.6 %) and UBEAR (98.4 %). It is lower on NITJ (89.0 %) and ITW (78.6 %).
4. **UBEAR has high CRR but high EER.** Each probe’s nearest neighbour is usually the right subject, but genuine and impostor scores overlap heavily under one global threshold.
5. **Dataset difficulty (EER, best to worst):** AWE 6.0 % < ITW 10.1 % < NITJ 16.0 % < UBEAR 23.2 %.

## Graphs

Combined ROC, today’s method on all datasets: [dt2_results/combined/ROC_all_datasets.png](dt2_results/combined/ROC_all_datasets.png)

| Dataset | Method | ROC | Genuine vs Impostor distribution | FAR / FRR | Other |
|---|---|---|---|---|---|
| AWE | Earlier – L2-normalized | [ROC](awe/awe_result/l2_normalized/roc.png) | [Genuine vs Impostor](awe/awe_result/l2_normalized/genuine_vs_impostor_histogram.png) | [FAR vs FRR](awe/awe_result/l2_normalized/far_vs_frr.png) | — |
| AWE | Earlier – Mean-centred (raw → L2 → −μ → L2) | [ROC](awe/awe_result/raw_mean_centered_l2/roc.png) | [Genuine vs Impostor](awe/awe_result/raw_mean_centered_l2/genuine_vs_impostor_histogram.png) | [FAR vs FRR](awe/awe_result/raw_mean_centered_l2/far_vs_frr.png) | — |
| AWE | Today – scripts_d_t_2 | [ROC](dt2_results/AWE/run/AWE/AWE.png) | [Genuine vs Impostor](dt2_results/AWE/run/AWE/AWE_GI.png) | [FAR & FRR vs threshold](dt2_results/AWE/run/AWE/AWE_FAR_FRR_vs_Threshold.png) | [Best genuine vs best impostor](dt2_results/AWE/run/AWE/AWE_GIBA.png) |
| ITW | Earlier – L2-normalized | [ROC](ITW/ITW_230_split_augment/itw_full_pair_evaluation/l2_normalized/roc.png) | [Genuine vs Impostor](ITW/ITW_230_split_augment/itw_full_pair_evaluation/l2_normalized/genuine_vs_impostor_histogram.png) | [FAR vs FRR](ITW/ITW_230_split_augment/itw_full_pair_evaluation/l2_normalized/far_vs_frr.png) | — |
| ITW | Earlier – Mean-centred (raw → L2 → −μ → L2) | [ROC](ITW/ITW_230_split_augment/itw_full_pair_evaluation/raw_mean_centered_l2/roc.png) | [Genuine vs Impostor](ITW/ITW_230_split_augment/itw_full_pair_evaluation/raw_mean_centered_l2/genuine_vs_impostor_histogram.png) | [FAR vs FRR](ITW/ITW_230_split_augment/itw_full_pair_evaluation/raw_mean_centered_l2/far_vs_frr.png) | — |
| ITW | Today – scripts_d_t_2 | [ROC](dt2_results/ITW/run/ITW/ITW.png) | [Genuine vs Impostor](dt2_results/ITW/run/ITW/ITW_GI.png) | [FAR & FRR vs threshold](dt2_results/ITW/run/ITW/ITW_FAR_FRR_vs_Threshold.png) | [Best genuine vs best impostor](dt2_results/ITW/run/ITW/ITW_GIBA.png) |
| UBEAR | Earlier – L2-normalized | [ROC](cropped_img/ubear_results/l2_normalized/roc.png) | [Genuine vs Impostor](cropped_img/ubear_results/l2_normalized/genuine_vs_impostor_histogram.png) | [FAR vs FRR](cropped_img/ubear_results/l2_normalized/far_vs_frr.png) | — |
| UBEAR | Earlier – Mean-centred (raw → L2 → −μ → L2) | [ROC](cropped_img/ubear_results/raw_mean_centered_l2/roc.png) | [Genuine vs Impostor](cropped_img/ubear_results/raw_mean_centered_l2/genuine_vs_impostor_histogram.png) | [FAR vs FRR](cropped_img/ubear_results/raw_mean_centered_l2/far_vs_frr.png) | — |
| UBEAR | Today – scripts_d_t_2 | [ROC](dt2_results/UBEAR/run/UBEAR/UBEAR.png) | [Genuine vs Impostor](dt2_results/UBEAR/run/UBEAR/UBEAR_GI.png) | [FAR & FRR vs threshold](dt2_results/UBEAR/run/UBEAR/UBEAR_FAR_FRR_vs_Threshold.png) | [Best genuine vs best impostor](dt2_results/UBEAR/run/UBEAR/UBEAR_GIBA.png) |
| NITJ | Today – scripts_d_t_2 | [ROC](dt2_results/NITJ/run/NITJ/NITJ.png) | [Genuine vs Impostor](dt2_results/NITJ/run/NITJ/NITJ_GI.png) | [FAR & FRR vs threshold](dt2_results/NITJ/run/NITJ/NITJ_FAR_FRR_vs_Threshold.png) | [Best genuine vs best impostor](dt2_results/NITJ/run/NITJ/NITJ_GIBA.png) |

