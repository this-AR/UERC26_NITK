import sys
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
DATASET_DIR = PROJECT_DIR / 'cropped_img'
MODEL_DIR = PROJECT_DIR / 'ITW' / 'ITW_230_split_augment'
RESULT_DIR = PROJECT_DIR / 'results' / 'legacy' / 'ubear'

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(MODEL_DIR))

import evaluate_all_pairs as evaluator


evaluator.EVALUATION_DIR = RESULT_DIR
evaluator.L2_DIR = RESULT_DIR / 'l2_normalized'
evaluator.CENTERED_DIR = RESULT_DIR / 'raw_mean_centered_l2'
evaluator.TEST_DIR = DATASET_DIR
evaluator.IMAGE_EXTENSIONS = {'.tiff', '.tif'}
evaluator.WEIGHTS_PATH = MODEL_DIR / 'best_pruned_35per_tinyvit.pt'
if not torch.cuda.is_available():
    raise RuntimeError('CUDA is required for this evaluation, but CUDA is unavailable.')
evaluator.DEVICE = torch.device('cuda')


def discover_images():
    records = []
    for image_path in sorted(DATASET_DIR.iterdir()):
        if image_path.is_file() and image_path.suffix.lower() in evaluator.IMAGE_EXTENSIONS:
            parts = image_path.stem.split('_')
            if len(parts) < 2 or parts[0] != 's':
                raise ValueError(f'Unexpected image filename: {image_path.name}')
            subject = f'{parts[0]}_{parts[1]}'
            records.append((subject, image_path))
    return records


evaluator.discover_images = discover_images


if __name__ == '__main__':
    evaluator.main()