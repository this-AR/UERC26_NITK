# New model (Swin-V2-B) vs pruned TinyViT: ear matching results

| | New model | Previous model |
|---|---|---|
| File | `best_model.pt` | `ITW/ITW_230_split_augment/best_pruned_35per_tinyvit.pt` |
| Architecture | torchvision Swin-V2-B + Linear(1024→512) + BatchNorm, AdaFace-trained (epoch 15, val VER 0.877) | Pruned TinyViT-11M + Linear(448→512) + BatchNorm |
| Parameters | 87.5 M | 7.53 M |
| Input | resize 224, RGB, ImageNet mean/std (model upsamples to 256); no CLAHE | resize 224, RGB, /255 |

Both models are evaluated the same way:
- embed → L2 → subtract global mean → L2;
- all images vs all images;
- score = 1 − cosine, normalised by `max_min.sh`;
- metrics from `main.sh` (scripts_d_t_2).

## 1. Main results (scripts_d_t_2, mean-centred)

| Dataset | Images / subjects | Model | EER (%) | CRR (%) | DI | FAR (%) | FRR (%) | VER@0.1% (%) | VER@1% (%) | AUC |
|---|---|---|---|---|---|---|---|---|---|---|
| AWE | 1000 / 100 | **Swin-V2** | **2.645** | **99.40** | -2.455 | 2.644 | 2.644 | 83.67 | 94.47 | 0.9971 |
| | | TinyViT | 6.005 | 96.60 | -1.913 | 6.005 | 6.022 | 62.00 | 82.33 | 0.9859 |
| ITW | 1175 / 197 | **Swin-V2** | **7.515** | **85.62** | -1.744 | 7.508 | 7.515 | 54.62 | 78.04 | 0.9765 |
| | | TinyViT | 10.055 | 78.64 | -1.570 | 10.041 | 10.091 | 46.28 | 69.76 | 0.9622 |
| UBEAR | 4412 / 126 | **Swin-V2** | **18.356** | **98.93** | -0.885 | 18.349 | 18.358 | 29.64 | 47.33 | 0.8989 |
| | | TinyViT | 23.171 | 98.39 | -0.734 | 23.168 | 23.175 | 21.59 | 36.49 | 0.8516 |
| NITJ | 10200 / 510 | **Swin-V2** | **11.134** | **93.05** | -0.925 | 11.126 | 11.144 | 51.81 | 70.03 | 0.9567 |
| | | TinyViT | 15.963 | 88.96 | -0.708 | 15.949 | 15.975 | 36.60 | 56.56 | 0.9194 |

Where the numbers come from:
- EER, CRR, DI, FAR and FRR come from the scripts (`.sts`).
- VER and AUC come from sklearn on the same embeddings, because the scripts don't compute them.
- The scripts' DI is negative because the scores are distances, and its size depends on the score scale, so compare DI between models with the positive DI in section 3 instead.

## 2. Change (Swin-V2 minus TinyViT)

| Dataset | ΔEER (pts) | ΔCRR (pts) | ΔVER@0.1% (pts) | ΔVER@1% (pts) | ΔAUC |
|---|---|---|---|---|---|
| AWE | -3.36 | +2.80 | +21.67 | +12.13 | +0.0112 |
| ITW | -2.54 | +6.98 | +8.34 | +8.28 | +0.0143 |
| UBEAR | -4.81 | +0.54 | +8.06 | +10.84 | +0.0473 |
| NITJ | -4.83 | +4.09 | +15.22 | +13.46 | +0.0373 |

## 3. L2-only vs mean-centred, both models (sklearn, unique pairs; DI = (μg − μi)/√((σg² + σi²)/2))

| Dataset | Model | Variant | EER (%) | Rank-1 / CRR (%) | DI | VER@0.1% (%) | VER@1% (%) | AUC |
|---|---|---|---|---|---|---|---|---|
| AWE | Swin-V2 | L2-only | 3.293 | 99.50 | 3.952 | 79.18 | 92.13 | 0.9947 |
| AWE | Swin-V2 | mean-centred | 2.652 | 99.40 | 4.195 | 83.67 | 94.47 | 0.9971 |
| AWE | TinyViT | L2-only | 6.582 | 96.20 | 3.193 | 59.04 | 80.69 | 0.9830 |
| AWE | TinyViT | mean-centred | 6.004 | 96.60 | 3.311 | 62.00 | 82.33 | 0.9859 |
| ITW | Swin-V2 | L2-only | 8.067 | 84.94 | 2.824 | 49.76 | 75.23 | 0.9721 |
| ITW | Swin-V2 | mean-centred | 7.496 | 85.79 | 2.923 | 54.62 | 78.04 | 0.9765 |
| ITW | TinyViT | L2-only | 10.786 | 78.38 | 2.486 | 38.88 | 63.50 | 0.9569 |
| ITW | TinyViT | mean-centred | 10.054 | 79.15 | 2.604 | 46.28 | 69.76 | 0.9622 |
| UBEAR | Swin-V2 | L2-only | 19.466 | 98.93 | 1.706 | 26.17 | 43.44 | 0.8884 |
| UBEAR | Swin-V2 | mean-centred | 18.354 | 98.93 | 1.772 | 29.64 | 47.33 | 0.8989 |
| UBEAR | TinyViT | L2-only | 25.343 | 98.48 | 1.350 | 19.71 | 33.33 | 0.8292 |
| UBEAR | TinyViT | mean-centred | 23.171 | 98.39 | 1.454 | 21.59 | 36.49 | 0.8516 |
| NITJ | Swin-V2 | L2-only | 13.281 | 92.82 | 2.313 | 47.15 | 65.18 | 0.9396 |
| NITJ | Swin-V2 | mean-centred | 11.134 | 93.05 | 2.536 | 51.81 | 70.03 | 0.9567 |
| NITJ | TinyViT | L2-only | 18.496 | 88.91 | 1.835 | 34.66 | 52.33 | 0.8941 |
| NITJ | TinyViT | mean-centred | 15.965 | 88.95 | 2.044 | 36.60 | 56.56 | 0.9194 |
| NITJ (R vs L) | Swin-V2 | L2-only | 14.600 | 72.94 | 2.198 | 35.93 | 58.17 | 0.9301 |
| NITJ (R vs L) | Swin-V2 | mean-centred | 12.245 | 75.27 | 2.454 | 41.57 | 64.22 | 0.9497 |
| NITJ (R vs L) | TinyViT | L2-only | 19.817 | 56.73 | 1.724 | 23.10 | 44.18 | 0.8808 |
| NITJ (R vs L) | TinyViT | mean-centred | 17.314 | 56.98 | 1.947 | 24.59 | 48.59 | 0.9074 |

`NITJ (R vs L)` uses the earlier NITJ protocol: right-ear probes against a left-ear gallery (51 000 genuine and 25 959 000 impostor pairs).

## 4. New model: full script output

| Dataset | Genuine (ordered) | Impostor (ordered) | EER threshold | False rejects | False accepts | Failed probes |
|---|---|---|---|---|---|---|
| AWE | 9000 | 990000 | 0.4701 | 238 / 9000 | 26176 / 990000 | 6 / 1000 |
| ITW | 6620 | 1372830 | 0.4991 | 496 / 6600 | 103068 / 1372830 | 169 / 1175 |
| UBEAR | 164988 | 19296344 | 0.5376 | 30288 / 164988 | 3540732 / 19296344 | 47 / 4412 |
| NITJ | 193800 | 103836000 | 0.4738 | 21598 / 193800 | 11553290 / 103836000 | 709 / 10200 |

## 5. Graphs (new model)

- Combined ROC, all datasets: [dt2_results_swinv2/combined/ROC_all_datasets.png](dt2_results_swinv2/combined/ROC_all_datasets.png). The TinyViT ROC data was deleted in the earlier clean-up, so only Swin-V2 curves are drawn; TinyViT numbers are in the tables above.
- AWE: [ROC](dt2_results_swinv2/AWE/run/AWE/AWE.png) · [Genuine vs Impostor](dt2_results_swinv2/AWE/run/AWE/AWE_GI.png) · [FAR & FRR vs threshold](dt2_results_swinv2/AWE/run/AWE/AWE_FAR_FRR_vs_Threshold.png) · [Best genuine vs best impostor](dt2_results_swinv2/AWE/run/AWE/AWE_GIBA.png)
- ITW: [ROC](dt2_results_swinv2/ITW/run/ITW/ITW.png) · [Genuine vs Impostor](dt2_results_swinv2/ITW/run/ITW/ITW_GI.png) · [FAR & FRR vs threshold](dt2_results_swinv2/ITW/run/ITW/ITW_FAR_FRR_vs_Threshold.png) · [Best genuine vs best impostor](dt2_results_swinv2/ITW/run/ITW/ITW_GIBA.png)
- UBEAR: [ROC](dt2_results_swinv2/UBEAR/run/UBEAR/UBEAR.png) · [Genuine vs Impostor](dt2_results_swinv2/UBEAR/run/UBEAR/UBEAR_GI.png) · [FAR & FRR vs threshold](dt2_results_swinv2/UBEAR/run/UBEAR/UBEAR_FAR_FRR_vs_Threshold.png) · [Best genuine vs best impostor](dt2_results_swinv2/UBEAR/run/UBEAR/UBEAR_GIBA.png)
- NITJ: [ROC](dt2_results_swinv2/NITJ/run/NITJ/NITJ.png) · [Genuine vs Impostor](dt2_results_swinv2/NITJ/run/NITJ/NITJ_GI.png) · [FAR & FRR vs threshold](dt2_results_swinv2/NITJ/run/NITJ/NITJ_FAR_FRR_vs_Threshold.png) · [Best genuine vs best impostor](dt2_results_swinv2/NITJ/run/NITJ/NITJ_GIBA.png)

