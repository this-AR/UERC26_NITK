import sys
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
DATASET_DIR = PROJECT_DIR / 'awe'
MODEL_DIR = PROJECT_DIR / 'ITW' / 'ITW_230_split_augment'
RESULT_DIR = PROJECT_DIR / 'results' / 'legacy' / 'awe'

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(MODEL_DIR))

import evaluate_all_pairs as evaluator


if not torch.cuda.is_available():
    raise RuntimeError('CUDA is required for this evaluation, but CUDA is unavailable.')

evaluator.EVALUATION_DIR = RESULT_DIR
evaluator.L2_DIR = RESULT_DIR / 'l2_normalized'
evaluator.CENTERED_DIR = RESULT_DIR / 'raw_mean_centered_l2'
evaluator.TEST_DIR = DATASET_DIR
evaluator.IMAGE_EXTENSIONS = {'.png'}
evaluator.WEIGHTS_PATH = MODEL_DIR / 'best_pruned_35per_tinyvit.pt'
evaluator.DEVICE = torch.device('cuda')


def discover_images():
    records = []
    for subject_dir in sorted(DATASET_DIR.iterdir()):
        if not subject_dir.is_dir() or subject_dir.name == 'evaluation_results':
            continue
        for image_path in sorted(subject_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in evaluator.IMAGE_EXTENSIONS:
                records.append((subject_dir.name, image_path))
    return records


evaluator.discover_images = discover_images


if __name__ == '__main__':
    evaluator.main()