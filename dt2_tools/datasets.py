"""Dataset discovery for the scripts_d_t_2 matching evaluation.

Every record is (subject_name, image_path). Left and right ears of one person
share the subject name (NITJ, UBEAR), as requested.
"""
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_DIR / 'ITW' / 'ITW_230_split_augment'
WEIGHTS_PATH = MODEL_DIR / 'best_pruned_35per_tinyvit.pt'
SCRIPTS_DIR = PROJECT_DIR / 'scripts_d_t_2 (Matching Results)' / 'scripts_d_t_2 (Matching Results)'


def _awe():
    root = PROJECT_DIR / 'awe'
    return [(d.name, p) for d in sorted(root.iterdir()) if d.is_dir() and d.name.isdigit()
            for p in sorted(d.glob('*.png'))]


def _itw():
    root = MODEL_DIR / 'test'
    return [(d.name, p) for d in sorted(root.iterdir()) if d.is_dir()
            for p in sorted(d.glob('*.png'))]


def _nitj():
    root = PROJECT_DIR / 'NITJ_Cropped' / 'Cropped'
    subjects = sorted((d for d in root.iterdir() if d.is_dir()), key=lambda d: int(d.name))
    return [(d.name, p) for d in subjects for p in sorted(d.rglob('*.tiff'))]


def _ubear():
    root = PROJECT_DIR / 'cropped_img'
    records = []
    for p in sorted(root.glob('*.tiff')):
        parts = p.stem.split('_')
        if parts[0] != 's':
            raise ValueError(f'Unexpected image filename: {p.name}')
        records.append((f'{parts[0]}_{parts[1]}', p))
    return sorted(records, key=lambda r: (r[0], r[1].name))


DATASETS = {
    'AWE': (_awe, PROJECT_DIR / 'awe'),
    'ITW': (_itw, MODEL_DIR),
    'NITJ': (_nitj, PROJECT_DIR / 'NITJ_Cropped'),
    'UBEAR': (_ubear, PROJECT_DIR / 'cropped_img'),
}


def output_dir(key):
    return PROJECT_DIR / 'dt2_results' / key


def discover(key):
    return DATASETS[key][0]()
