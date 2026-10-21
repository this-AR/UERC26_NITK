import csv
import math
import os
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import auc, roc_curve, roc_auc_score

matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Earlier exhaustive-pair evaluator (L2-only and mean-centred, uerc26_utils metrics).
# Moved from ITW/ITW_230_split_augment/itw_full_pair_evaluation/; outputs now go to results/legacy/<dataset>/.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
MODEL_DIR = PROJECT_DIR / 'ITW' / 'ITW_230_split_augment'
EVALUATION_DIR = PROJECT_DIR / 'results' / 'legacy' / 'itw'
L2_DIR = EVALUATION_DIR / 'l2_normalized'
CENTERED_DIR = EVALUATION_DIR / 'raw_mean_centered_l2'
sys.path.insert(0, str(MODEL_DIR))
sys.path.insert(0, str(PROJECT_DIR / 'legacy' / 'uerc26_starter'))

from model import PrunedTinyViTWrapper
from uerc26_utils import compute_metrics


WEIGHTS_PATH = MODEL_DIR / 'best_pruned_35per_tinyvit.pt'
TEST_DIR = MODEL_DIR / 'test'
IMG_SIZE = 224
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp'}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_image(path):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f'Cannot read image: {path}')
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image[:, :, ::-1].copy()
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(image).unsqueeze(0)


def discover_images():
    records = []
    for subject_dir in sorted(TEST_DIR.iterdir()):
        if not subject_dir.is_dir():
            continue
        for image_path in sorted(subject_dir.iterdir()):
            if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                records.append((subject_dir.name, image_path))
    return records


def l2_normalize(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError('Cannot L2-normalize a zero vector.')
    return vector / norm


def compute_embeddings(model, records):
    embeddings = []
    with torch.no_grad():
        for index, (_, image_path) in enumerate(records, start=1):
            tensor = load_image(image_path).to(DEVICE)
            embedding = model(tensor)
            embedding = F.normalize(embedding, p=2, dim=1)
            embeddings.append(embedding.cpu().numpy().ravel())
            if index % 50 == 0 or index == len(records):
                print(f'  [{index}/{len(records)}] embeddings computed')
    return np.asarray(embeddings, dtype=np.float64)


def make_centered_embeddings(embeddings):
    # Keep the global mean in its raw form before centering.
    global_mean = np.mean(embeddings, axis=0)
    centered = embeddings - global_mean
    centered_norms = np.linalg.norm(centered, axis=1, keepdims=True)
    if np.any(centered_norms == 0):
        raise ValueError('At least one centered embedding has zero norm.')
    return centered / centered_norms, global_mean


def write_pair_scores(records, embeddings, output_path):
    labels = []
    scores = []
    with output_path.open('w', newline='') as output_file:
        for left_index in range(len(records) - 1):
            left_subject, left_path = records[left_index]
            left_name = left_path.name
            for right_index in range(left_index + 1, len(records)):
                right_subject, right_path = records[right_index]
                label = int(left_subject == right_subject)
                score = float(np.dot(embeddings[left_index], embeddings[right_index]))
                labels.append(label)
                scores.append(score)
                output_file.write(
                    f'{left_subject} {left_name} {right_subject} '
                    f'{right_path.name} {label} {score!r}\n'
                )
    return np.asarray(labels, dtype=np.int8), np.asarray(scores, dtype=np.float64)


def score_statistics(scores):
    mean = float(np.mean(scores))
    std = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
    return {
        'count': int(len(scores)),
        'min': float(np.min(scores)),
        'max': float(np.max(scores)),
        'mean': mean,
        'standard_deviation': std,
        'standard_error': float(std / math.sqrt(len(scores))) if scores.size else 0.0,
    }


def calculate_metrics(labels, scores):
    metrics = compute_metrics(labels, scores)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    eer_index = int(np.argmin(np.abs(fpr - fnr)))
    return {
        'VER@0.1%': float(metrics['VER@0.1%']),
        'VER@1%': float(metrics['VER@1%']),
        'EER': float(metrics['EER']),
        'AUC': float(metrics['AUC']),
        'EER_threshold': float(thresholds[eer_index]),
        'FAR_at_EER_threshold': float(fpr[eer_index]),
        'FRR_at_EER_threshold': float(fnr[eer_index]),
        'fpr': fpr,
        'tpr': tpr,
        'fnr': fnr,
        'thresholds': thresholds,
    }


def write_threshold_data(labels, scores, output_path):
    far, ver, thresholds = roc_curve(labels, scores)
    frr = 1.0 - ver
    with output_path.open('w', newline='') as output_file:
        writer = csv.writer(output_file, delimiter='\t')
        writer.writerow(['threshold', 'FAR_percent', 'FRR_percent', 'VER_percent'])
        for threshold, threshold_far, threshold_frr, threshold_ver in zip(
                thresholds, far, frr, ver):
            writer.writerow([f'{threshold:.12g}', f'{threshold_far * 100:.12g}',
                             f'{threshold_frr * 100:.12g}',
                             f'{threshold_ver * 100:.12g}'])


def write_distribution_data(scores, output_path):
    sorted_scores = np.sort(scores)
    cdf = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
    with output_path.open('w', newline='') as output_file:
        writer = csv.writer(output_file, delimiter='\t')
        writer.writerow(['score', 'cdf'])
        writer.writerows(zip(sorted_scores, cdf))


def write_histogram_data(genuine, impostor, output_path, bins=100):
    minimum = min(np.min(genuine), np.min(impostor))
    maximum = max(np.max(genuine), np.max(impostor))
    counts_genuine, edges = np.histogram(genuine, bins=bins, range=(minimum, maximum))
    counts_impostor, _ = np.histogram(impostor, bins=bins, range=(minimum, maximum))
    centers = (edges[:-1] + edges[1:]) / 2
    with output_path.open('w', newline='') as output_file:
        writer = csv.writer(output_file, delimiter='\t')
        writer.writerow(['bin_center', 'genuine_frequency', 'impostor_frequency'])
        writer.writerows(zip(centers, counts_genuine, counts_impostor))


def write_report(name, labels, scores, metrics, output_path):
    genuine = scores[labels == 1]
    impostor = scores[labels == 0]
    with output_path.open('w') as report:
        report.write(f'ITW exhaustive pair evaluation: {name}\n')
        report.write('=' * 60 + '\n\n')
        report.write('Pair protocol\n')
        report.write('Unique unordered pairs; self-comparisons excluded.\n')
        report.write('Higher cosine similarity means a stronger match.\n\n')
        report.write(f'Total pairs: {len(scores)}\n')
        report.write(f'Genuine pairs: {len(genuine)}\n')
        report.write(f'Impostor pairs: {len(impostor)}\n\n')
        report.write('Metrics from compute_metrics\n')
        for key in ('EER', 'VER@0.1%', 'VER@1%', 'AUC'):
            report.write(f'{key}: {metrics[key]:.12g}\n')
        report.write('\nAdditional FAR/FRR metrics\n')
        report.write(f"EER threshold: {metrics['EER_threshold']:.12g}\n")
        report.write(f"FAR at EER threshold: {metrics['FAR_at_EER_threshold']:.12g}\n")
        report.write(f"FRR at EER threshold: {metrics['FRR_at_EER_threshold']:.12g}\n\n")
        for label, values in (('Genuine', genuine), ('Impostor', impostor)):
            report.write(f'{label} score statistics\n')
            for key, value in score_statistics(values).items():
                report.write(f'{key}: {value:.12g}\n')
            report.write('\n')


def create_plots(name, labels, scores, metrics, output_dir):
    genuine = scores[labels == 1]
    impostor = scores[labels == 0]

    plt.figure(figsize=(8, 6))
    plt.plot(metrics['fpr'] * 100, metrics['fnr'] * 100, label='FRR')
    plt.plot(metrics['fpr'] * 100, metrics['fpr'] * 100, label='FAR')
    plt.scatter(metrics['FAR_at_EER_threshold'] * 100,
                metrics['FRR_at_EER_threshold'] * 100,
                color='red', label='EER')
    plt.xlabel('Rate (%)')
    plt.ylabel('Rate (%)')
    plt.title(f'FAR vs FRR - {name}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'far_vs_frr.png', dpi=160)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.hist(genuine, bins=100, alpha=0.65, density=True, label='Genuine')
    plt.hist(impostor, bins=100, alpha=0.65, density=True, label='Impostor')
    plt.xlabel('Cosine similarity score')
    plt.ylabel('Frequency density')
    plt.title(f'Genuine vs Impostor Scores - {name}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'genuine_vs_impostor_histogram.png', dpi=160)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(metrics['fpr'] * 100, metrics['tpr'] * 100)
    plt.xlabel('FAR (%)')
    plt.ylabel('VER / TPR (%)')
    plt.title(f'ROC Curve - {name} (AUC={metrics["AUC"]:.6f})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'roc.png', dpi=160)
    plt.close()

    for score_type, values in (('genuine', genuine), ('impostor', impostor)):
        sorted_values = np.sort(values)
        cdf = np.arange(1, len(values) + 1) / len(values)
        plt.figure(figsize=(8, 6))
        plt.plot(sorted_values, cdf)
        plt.xlabel('Cosine similarity score')
        plt.ylabel('Cumulative probability')
        plt.title(f'{score_type.title()} Score CDF - {name}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f'{score_type}_cdf.png', dpi=160)
        plt.close()


def evaluate_variant(name, records, embeddings, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    score_path = output_dir / 'pair_scores.txt'
    labels, scores = write_pair_scores(records, embeddings, score_path)
    metrics = calculate_metrics(labels, scores)
    genuine = scores[labels == 1]
    impostor = scores[labels == 0]
    write_report(name, labels, scores, metrics, output_dir / 'metrics.txt')
    write_threshold_data(labels, scores, output_dir / 'far_frr.tsv')
    write_distribution_data(genuine, output_dir / 'genuine_cdf.tsv')
    write_distribution_data(impostor, output_dir / 'impostor_cdf.tsv')
    write_histogram_data(genuine, impostor, output_dir / 'histogram.tsv')
    create_plots(name, labels, scores, metrics, output_dir)
    return metrics


def main():
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    L2_DIR.mkdir(parents=True, exist_ok=True)
    CENTERED_DIR.mkdir(parents=True, exist_ok=True)
    records = discover_images()
    if len(records) < 2:
        raise RuntimeError('At least two test images are required.')

    print(f'Found {len(records)} images across {len(set(s for s, _ in records))} subjects.')
    print(f'Unique pairs: {len(records) * (len(records) - 1) // 2}')
    print(f'Using device: {DEVICE}')
    print(f'Loading model from {WEIGHTS_PATH}')
    model = PrunedTinyViTWrapper().model
    state = torch.load(WEIGHTS_PATH, map_location='cpu')
    model.load_state_dict(state, strict=False)
    model = model.to(DEVICE)
    model.eval()

    print('Computing embeddings...')
    embeddings = compute_embeddings(model, records)
    centered_embeddings, global_mean = make_centered_embeddings(embeddings)
    np.savetxt(CENTERED_DIR / 'global_mean_embedding.txt', global_mean)

    print('Evaluating L2-only scores...')
    normalized_metrics = evaluate_variant('l2_normalized', records, embeddings, L2_DIR)
    print('Evaluating mean-centered scores...')
    centered_metrics = evaluate_variant('raw_mean_centered_l2', records, centered_embeddings, CENTERED_DIR)

    with (EVALUATION_DIR / 'summary.txt').open('w') as summary:
        summary.write('ITW exhaustive pair evaluation summary\n')
        summary.write('=' * 45 + '\n\n')
        summary.write('Both variants use unique unordered pairs and exclude self-comparisons.\n')
        summary.write('Scores use the cosine convention: higher means more genuine.\n\n')
        for name, metrics in (('l2_normalized', normalized_metrics),
                              ('mean_centered_l2', centered_metrics)):
            summary.write(f'{name}\n')
            for key in ('EER', 'VER@0.1%', 'VER@1%', 'AUC'):
                summary.write(f'  {key}: {metrics[key]:.12g}\n')
            summary.write('\n')
    print(f'All results written to {EVALUATION_DIR}')


if __name__ == '__main__':
    main()