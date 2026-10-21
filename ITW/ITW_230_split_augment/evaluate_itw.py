import cv2
import numpy as np
import torch
import torch.nn.functional as F
import os
import sys
import math

# uerc26_utils lives in legacy/uerc26_starter
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'legacy', 'uerc26_starter')))
from uerc26_utils import compute_metrics

from model import PrunedTinyViTWrapper

# ─── Configuration ───────────────────────────────────────────────────────────
WEIGHTS_PATH = 'best_pruned_35per_tinyvit.pt'
TEST_DIR     = 'test'
OUTPUT_FILE  = 'TinyViT_pruned_results.txt'
IMG_SIZE     = 224          # TinyViT expects 224×224
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─── Load model ──────────────────────────────────────────────────────────────
print(f'Loading pruned TinyViT model from {WEIGHTS_PATH} ...')
wrapper = PrunedTinyViTWrapper()
model   = wrapper.model                        # _PrunedTinyViT (auto-rebuilds on load)
state   = torch.load(WEIGHTS_PATH, map_location='cpu')
model.load_state_dict(state, strict=False)      # triggers rebuild_from_pruned_state_dict
model = model.to(DEVICE)
model.eval()
print('Model loaded.\n')

# ─── Discover subjects and their images ──────────────────────────────────────
subject_dirs = sorted([
    d for d in os.listdir(TEST_DIR)
    if os.path.isdir(os.path.join(TEST_DIR, d))
])

# Build a dict:  subject_name -> sorted list of image paths
subject_images = {}
for subj in subject_dirs:
    folder = os.path.join(TEST_DIR, subj)
    imgs   = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
    ])
    if imgs:
        subject_images[subj] = [os.path.join(folder, f) for f in imgs]

subjects = list(subject_images.keys())
num_subjects = len(subjects)
print(f'Found {num_subjects} subjects in {TEST_DIR}/\n')

# ─── Split each subject into gallery / probe ─────────────────────────────────
#     If n images: gallery = ceil(n/2), probe = floor(n/2)
#     (more in gallery when odd)
gallery_paths = {}   # subject -> [paths]
probe_paths   = {}   # subject -> [paths]

for subj in subjects:
    all_imgs = subject_images[subj]
    n = len(all_imgs)
    n_gallery = math.ceil(n / 2)
    gallery_paths[subj] = all_imgs[:n_gallery]
    probe_paths[subj]   = all_imgs[n_gallery:]
    print(f'  Subject {subj}: {n} images  ->  gallery={n_gallery}, probe={n - n_gallery}')

print()

# ─── Helper: load & preprocess a single image ────────────────────────────────
def load_image(path):
    """Read an image, resize to IMG_SIZE×IMG_SIZE, normalise to [0,1], return tensor [1,3,H,W]."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f'Cannot read image: {path}')
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img[:, :, ::-1].copy()          # BGR → RGB
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))    # HWC → CHW
    return torch.from_numpy(img).unsqueeze(0)   # [1,3,H,W]


# ─── Phase 1: Pre-compute ALL embeddings ─────────────────────────────────────
print('Pre-computing embeddings for all images ...')
embeddings = {}   # image_path -> numpy 1-D embedding

all_image_paths = []
for subj in subjects:
    all_image_paths.extend(gallery_paths[subj])
    all_image_paths.extend(probe_paths[subj])

with torch.no_grad():
    for idx, img_path in enumerate(all_image_paths):
        tensor = load_image(img_path).to(DEVICE)
        emb = model(tensor)                       # [1, 512]
        emb = F.normalize(emb, p=2, dim=1)        # L2-normalise
        embeddings[img_path] = emb.cpu().numpy().flatten()
        if (idx + 1) % 50 == 0 or (idx + 1) == len(all_image_paths):
            print(f'  [{idx+1}/{len(all_image_paths)}] embeddings computed')

print(f'All {len(embeddings)} embeddings ready.\n')

# Calculate the normalized global mean, then center and renormalize every
# embedding. The original normalized embeddings are kept for comparison.
global_mean = np.mean(np.stack(list(embeddings.values())), axis=0)
global_mean_norm = np.linalg.norm(global_mean)
if global_mean_norm == 0:
    raise ValueError('The global mean embedding has zero norm.')
global_mean = global_mean / global_mean_norm

centered_embeddings = {}
for image_path, embedding in embeddings.items():
    centered = embedding - global_mean
    centered_norm = np.linalg.norm(centered)
    if centered_norm == 0:
        raise ValueError(f'Centered embedding has zero norm: {image_path}')
    centered_embeddings[image_path] = centered / centered_norm

print('Global mean embedding calculated and centered embeddings normalized.\n')

# ─── Phase 2: Score every (probe, gallery) pair ──────────────────────────────
#     For each probe image compare against every gallery image across all subjects
#       • score_normalized = cosine similarity after per-image L2 normalization
#       • score_centered = cosine similarity after global-mean centering
#       • flag  = 1 if same subject, 0 otherwise
#     Output columns: probe_subject probe_image gallery_subject gallery_image
#                     flag score_normalized score_centered
print('Computing pairwise scores ...')
f = open(OUTPUT_FILE, 'w')

labels = []
scores_normalized = []
scores_centered = []

for s1 in subjects:
    probes = probe_paths[s1]
    if not probes:
        continue
    for p1_path in probes:
        p1_name = os.path.basename(p1_path)
        emb_p = embeddings[p1_path]
        centered_emb_p = centered_embeddings[p1_path]
        for s2 in subjects:
            gallery = gallery_paths[s2]
            for p2_path in gallery:
                p2_name = os.path.basename(p2_path)
                emb_g = embeddings[p2_path]
                centered_emb_g = centered_embeddings[p2_path]
                
                # Both embedding sets are L2 normalized, so dot products are cosine similarities.
                score_normalized = float(np.dot(emb_p, emb_g))
                score_centered = float(np.dot(centered_emb_p, centered_emb_g))
                flag  = 1 if s1 == s2 else 0
                
                labels.append(flag)
                scores_normalized.append(score_normalized)
                scores_centered.append(score_centered)
                
                line  = (f'{s1}\t{p1_name}\t'
                         f'{s2}\t{p2_name}\t'
                         f'{flag}\t{score_normalized:.6f}\t{score_centered:.6f}\n')
                f.write(line)

f.close()
print(f'Results written to {OUTPUT_FILE}')

# ─── Phase 3: Calculate Metrics ──────────────────────────────────────────────
print('\nCalculating evaluation metrics ...')
labels_arr = np.array(labels)
scores_normalized_arr = np.array(scores_normalized)
scores_centered_arr = np.array(scores_centered)

# Compute metrics for both normalized-only and centered embeddings.
metrics_normalized = compute_metrics(labels_arr, scores_normalized_arr)
metrics_centered = compute_metrics(labels_arr, scores_centered_arr)

print("Evaluation Results (per-image L2 normalization only):")
for metric_name, metric_value in metrics_normalized.items():
    if "inference_time" not in metric_name:
        print(f"  {metric_name}: {metric_value:.4f}")

print("\nEvaluation Results (global-mean centered + L2 normalization):")
for metric_name, metric_value in metrics_centered.items():
    if "inference_time" not in metric_name:
        print(f"  {metric_name}: {metric_value:.4f}")

print('\nDone.')
