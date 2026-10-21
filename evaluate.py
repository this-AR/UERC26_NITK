"""
Baseline vs Mean-Centered Evaluation on cropped_img Dataset
============================================================
- Uses pre-defined pairs from pairs_test.csv
- Model: best_pruned_35per_tinyvit.pt (unchanged)
- Two scoring methods:
    1. Baseline:      L2-normalize -> cosine similarity
    2. Mean-Centered: subtract global mean -> L2-normalize -> cosine similarity
- All outputs written to THIS folder (cropped_img/)
"""

import sys
import os
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve, roc_auc_score

# ---- Paths ---------------------------------------------------------------
DATASET_DIR  = os.path.dirname(os.path.abspath(__file__))   # cropped_img/
MODEL_DIR    = r'C:\Users\hp\Downloads\ITW\ITW\ITW_230_split_augment'
WEIGHTS_PATH = os.path.join(MODEL_DIR, 'best_pruned_35per_tinyvit.pt')
PAIRS_CSV    = os.path.join(DATASET_DIR, 'pairs_test.csv')
IMAGE_LIST   = os.path.join(DATASET_DIR, 'image_list.csv')
OUT_DIR      = DATASET_DIR
IMG_SIZE     = 224
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sys.path.insert(0, MODEL_DIR)
from model import PrunedTinyViTWrapper

# ---- Load model ----------------------------------------------------------
print('Loading model ...')
wrapper = PrunedTinyViTWrapper()
model   = wrapper.model
state   = torch.load(WEIGHTS_PATH, map_location='cpu')
model.load_state_dict(state, strict=False)
model = model.to(DEVICE)
model.eval()
print(f'  Device: {DEVICE}')
print('Done.\n')

# ---- Load pairs ----------------------------------------------------------
print(f'Loading pairs from: {PAIRS_CSV}')
pairs = []
with open(PAIRS_CSV, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split('\t')
        if len(parts) < 3:
            continue
        path_a, path_b, label = parts[0], parts[1], int(parts[2])
        # Normalise backslashes, strip subject folder prefix
        fname_a = os.path.basename(path_a.replace('\\', os.sep))
        fname_b = os.path.basename(path_b.replace('\\', os.sep))
        pairs.append((fname_a, fname_b, label))

print(f'  Total pairs: {len(pairs)}')
n_genuine  = sum(1 for _, _, l in pairs if l == 1)
n_imposter = sum(1 for _, _, l in pairs if l == 0)
print(f'  Genuine:  {n_genuine}')
print(f'  Imposter: {n_imposter}\n')

# ---- Collect unique images -----------------------------------------------
needed = set()
for a, b, _ in pairs:
    needed.add(a)
    needed.add(b)
print(f'Unique images needed: {len(needed)}')

# Find each image in dataset dir (flat folder)
img_path_map = {}
for fname in os.listdir(DATASET_DIR):
    if fname in needed:
        img_path_map[fname] = os.path.join(DATASET_DIR, fname)

missing = needed - set(img_path_map.keys())
if missing:
    print(f'  WARNING: {len(missing)} images not found! First 5: {list(missing)[:5]}')
print(f'  Found: {len(img_path_map)} images\n')

# ---- Extract embeddings --------------------------------------------------
def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f'Cannot read: {path}')
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img[:, :, ::-1].copy().astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img).unsqueeze(0)

print('Extracting raw embeddings ...')
raw_embs = {}
img_list = sorted(img_path_map.keys())
with torch.no_grad():
    for idx, fname in enumerate(img_list):
        t   = load_image(img_path_map[fname]).to(DEVICE)
        emb = model(t).cpu().numpy().flatten()
        raw_embs[fname] = emb
        if (idx + 1) % 500 == 0 or (idx + 1) == len(img_list):
            print(f'  [{idx+1}/{len(img_list)}]')

print(f'  Embeddings shape: ({len(raw_embs)}, {next(iter(raw_embs.values())).shape[0]})\n')

# ---- Compute global mean (from all extracted images) ----------------------
E_all  = np.array([raw_embs[f] for f in img_list])
G_mean = E_all.mean(axis=0)
print(f'Global mean norm: {np.linalg.norm(G_mean):.4f}\n')

# ---- Scoring helpers -------------------------------------------------------
def l2(v):
    n = np.linalg.norm(v)
    return v / max(n, 1e-12)

def baseline_score(fa, fb):
    return float(np.dot(l2(raw_embs[fa]), l2(raw_embs[fb])))

def centered_score(fa, fb):
    ca = l2(raw_embs[fa] - G_mean)
    cb = l2(raw_embs[fb] - G_mean)
    return float(np.dot(ca, cb))

# ---- Metrics helper --------------------------------------------------------
def compute_metrics(labels_arr, scores_arr):
    fpr, tpr, thr = roc_curve(labels_arr, scores_arr)
    fnr = 1 - tpr
    eer_idx = np.argmin(np.abs(fnr - fpr))
    eer     = float((fpr[eer_idx] + fnr[eer_idx]) / 2)
    order   = np.argsort(fpr)
    fpr_s, tpr_s = fpr[order], tpr[order]
    interp  = interp1d(fpr_s, tpr_s, bounds_error=False, fill_value=(0.0, 1.0))
    auc     = float(roc_auc_score(labels_arr, scores_arr))
    return {
        'EER':      eer,
        'VER@0.1%': float(interp(0.001)),
        'VER@1%':   float(interp(0.01)),
        'AUC':      auc,
        'fpr': fpr, 'tpr': tpr, 'thr': thr,
    }

# ---- Run both methods -------------------------------------------------------
methods = {
    'Baseline (L2)':      baseline_score,
    'Mean-Centered + L2': centered_score,
}

results   = {}
gen_scores_all  = {}
imp_scores_all  = {}

for method_name, score_fn in methods.items():
    print(f'Evaluating: {method_name}')
    labels, scores = [], []
    gen_s, imp_s   = [], []

    for fa, fb, label in pairs:
        if fa not in raw_embs or fb not in raw_embs:
            continue
        s = score_fn(fa, fb)
        labels.append(label)
        scores.append(s)
        if label == 1: gen_s.append(s)
        else:          imp_s.append(s)

    labels_arr = np.array(labels)
    scores_arr = np.array(scores)

    m = compute_metrics(labels_arr, scores_arr)
    results[method_name]       = m
    gen_scores_all[method_name]  = np.array(gen_s)
    imp_scores_all[method_name]  = np.array(imp_s)

    print(f'  EER={m["EER"]:.4f}  VER@0.1%={m["VER@0.1%"]:.4f}  VER@1%={m["VER@1%"]:.4f}  AUC={m["AUC"]:.4f}')

# ---- Write score TXT files -------------------------------------------------
for method_name, score_fn in methods.items():
    safe_name = method_name.replace(' ', '_').replace('(', '').replace(')', '').replace('+', 'plus')
    out_path  = os.path.join(OUT_DIR, f'scores_{safe_name}.txt')
    with open(out_path, 'w') as f:
        f.write('ImageA\tImageB\tLabel\tScore\n')
        for fa, fb, label in pairs:
            if fa not in raw_embs or fb not in raw_embs:
                continue
            s = score_fn(fa, fb)
            f.write(f'{fa}\t{fb}\t{label}\t{s:.6f}\n')
    print(f'Saved: {out_path}')

# ---- Write results TXT table -----------------------------------------------
results_path = os.path.join(OUT_DIR, 'results_cropped_img.txt')
with open(results_path, 'w') as f:
    f.write('Evaluation on cropped_img dataset\n')
    f.write(f'Model: {WEIGHTS_PATH}\n')
    f.write(f'Total pairs: {len(pairs)}  Genuine: {n_genuine}  Imposter: {n_imposter}\n\n')
    f.write(f'{"Method":<25} {"EER":>8} {"VER@0.1%":>10} {"VER@1%":>8} {"AUC":>8}\n')
    f.write('-' * 65 + '\n')
    for name, m in results.items():
        f.write(f'{name:<25} {m["EER"]:>8.4f} {m["VER@0.1%"]:>10.4f} {m["VER@1%"]:>8.4f} {m["AUC"]:>8.4f}\n')
    f.write('-' * 65 + '\n')
    # Delta row
    b = results['Baseline (L2)']
    c = results['Mean-Centered + L2']
    f.write(f'{"Improvement":.<25} {c["EER"]-b["EER"]:>+8.4f} {c["VER@0.1%"]-b["VER@0.1%"]:>+10.4f} {c["VER@1%"]-b["VER@1%"]:>+8.4f} {c["AUC"]-b["AUC"]:>+8.4f}\n')

print(f'\nSaved results: {results_path}')

# Print to console too
print('\n' + '=' * 65)
print(f'{"Method":<25} {"EER":>8} {"VER@0.1%":>10} {"VER@1%":>8} {"AUC":>8}')
print('-' * 65)
for name, m in results.items():
    print(f'{name:<25} {m["EER"]:>8.4f} {m["VER@0.1%"]:>10.4f} {m["VER@1%"]:>8.4f} {m["AUC"]:>8.4f}')
print('-' * 65)
b = results['Baseline (L2)']
c = results['Mean-Centered + L2']
print(f'{"Improvement":.<25} {c["EER"]-b["EER"]:>+8.4f} {c["VER@0.1%"]-b["VER@0.1%"]:>+10.4f} {c["VER@1%"]-b["VER@1%"]:>+8.4f} {c["AUC"]-b["AUC"]:>+8.4f}')
print('=' * 65)

# ============================================================================
# PLOTS
# ============================================================================
colors = {'Baseline (L2)': 'steelblue', 'Mean-Centered + L2': 'darkorange'}

# 1. Score distributions side-by-side
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Genuine vs Imposter Score Distributions\ncropped_img Dataset', fontsize=13, fontweight='bold')
for ax, (name, _) in zip(axes, methods.items()):
    gen_s = gen_scores_all[name]
    imp_s = imp_scores_all[name]
    ax.hist(imp_s, bins=120, density=True, alpha=0.65, color='teal',
            label=f'Imposter (mean={imp_s.mean():.3f})')
    ax.hist(gen_s, bins=120, density=True, alpha=0.65, color='purple',
            label=f'Genuine  (mean={gen_s.mean():.3f})')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Ideal imposter mean')
    ax.set_title(name, fontsize=11)
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Density')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'score_distributions.png'), dpi=200, bbox_inches='tight')
plt.close(fig)
print('Saved: score_distributions.png')

# 2. ROC curves overlay
fig, ax = plt.subplots(figsize=(8, 7))
for name, m in results.items():
    ax.plot(m['fpr'], m['tpr'], color=colors[name], linewidth=2.5,
            label=f'{name}  (AUC={m["AUC"]:.4f}, EER={m["EER"]:.4f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Chance')
ax.set_title('ROC Curve Comparison\ncropped_img Dataset', fontsize=13, fontweight='bold')
ax.set_xlabel('False Positive Rate (FAR)')
ax.set_ylabel('True Positive Rate (VER)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'roc_comparison.png'), dpi=200, bbox_inches='tight')
plt.close(fig)
print('Saved: roc_comparison.png')

# 3. FAR / FRR curves (DET-style)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('FAR and FRR vs Threshold\ncropped_img Dataset', fontsize=13, fontweight='bold')
for ax, (name, m) in zip(axes, results.items()):
    fpr, tpr, thr = m['fpr'], m['tpr'], m['thr']
    frr = 1 - tpr
    # roc_curve returns len(fpr) = len(tpr) = len(thr) + 1
    # Drop the first element of fpr/frr to align with thr
    fpr_plot = fpr[1:]
    frr_plot = frr[1:]
    thr_plot = np.clip(thr, -1.5, 1.5)   # guard against Inf sentinels
    # trim to same length
    n = min(len(thr_plot), len(fpr_plot))
    thr_plot, fpr_plot, frr_plot = thr_plot[:n], fpr_plot[:n], frr_plot[:n]
    # only keep finite entries
    mask = np.isfinite(thr_plot)
    thr_plot, fpr_plot, frr_plot = thr_plot[mask], fpr_plot[mask], frr_plot[mask]

    ax.plot(thr_plot, fpr_plot, color='blue', linewidth=2, label='FAR')
    ax.plot(thr_plot, frr_plot, color='red',  linewidth=2, label='FRR')
    eer_idx = np.argmin(np.abs(fpr_plot - frr_plot))
    ax.axvline(x=thr_plot[eer_idx], color='black', linestyle='--', linewidth=1.2,
               label=f'EER={m["EER"]:.4f} @ thr={thr_plot[eer_idx]:.3f}')
    ax.set_title(name, fontsize=11)
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Error Rate')
    xlim_min = float(np.nanmin(thr_plot)) - 0.05
    xlim_max = float(np.nanmax(thr_plot)) + 0.05
    ax.set_xlim([xlim_min, xlim_max])
    ax.set_ylim([-0.02, 1.02])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'far_frr_curves.png'), dpi=200, bbox_inches='tight')
plt.close(fig)
print('Saved: far_frr_curves.png')

# 4. Metrics bar chart
fig, axes = plt.subplots(1, 4, figsize=(16, 5))
fig.suptitle('Metrics Summary — cropped_img Dataset', fontsize=13, fontweight='bold')
metric_keys  = ['EER', 'VER@0.1%', 'VER@1%', 'AUC']
bar_colors   = ['#e74c3c', '#2ecc71', '#3498db', '#9b59b6']
method_names = list(results.keys())

for ax, metric, col in zip(axes, metric_keys, bar_colors):
    vals = [results[m][metric] for m in method_names]
    bars = ax.bar(range(len(method_names)), vals, color=col, alpha=0.85,
                  edgecolor='black', linewidth=0.7)
    ax.set_xticks(range(len(method_names)))
    ax.set_xticklabels([n.replace(' + ', '\n+\n') for n in method_names], fontsize=8)
    ax.set_title(metric, fontsize=11, fontweight='bold')
    ax.set_ylim([0, max(vals) * 1.2])
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'metrics_summary.png'), dpi=200, bbox_inches='tight')
plt.close(fig)
print('Saved: metrics_summary.png')

print('\nAll done!')
