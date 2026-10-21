"""Every path used by the evaluation, plus the model selection.

Datasets and weights stay where they were delivered; only code and results live in the new layout.
Pick the model with environment variables:
    DT2_MODEL=tinyvit (default) | swinv2
    DT2_CLAHE=1        Swin-V2 only: LAB CLAHE on the input, as in the notebook's feature extraction
"""
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

# Data (unchanged locations)
AWE_DIR = PROJECT_DIR / 'awe'
ITW_DIR = PROJECT_DIR / 'ITW' / 'ITW_230_split_augment'
ITW_TEST_DIR = ITW_DIR / 'test'
UBEAR_DIR = PROJECT_DIR / 'cropped_img'
NITJ_DIR = PROJECT_DIR / 'NITJ_Cropped' / 'Cropped'

# Models
TINYVIT_DIR = ITW_DIR  # holds model.py (architecture) next to the weights
TINYVIT_WEIGHTS = TINYVIT_DIR / 'best_pruned_35per_tinyvit.pt'
SWINV2_WEIGHTS = PROJECT_DIR / 'best_model.pt'

# Code and tools
MATCHING_SCRIPTS_DIR = PROJECT_DIR / 'matching_scripts'
TOOLS_BIN = PROJECT_DIR / 'tools' / 'bin'
GNUPLOT_BIN = Path(r'C:\Program Files\gnuplot\bin')
BASH = Path(r'C:\Program Files\Git\bin\bash.exe')

# Outputs
RESULTS_DIR = PROJECT_DIR / 'results'
ANALYSIS_DIR = RESULTS_DIR / 'analysis'
CACHE_DIR = PROJECT_DIR / '.cache' / 'embeddings'

# Model selection
MODEL = os.environ.get('DT2_MODEL', 'tinyvit')
CLAHE = os.environ.get('DT2_CLAHE', '0') == '1'


def model_tag():
    """Name used for result folders and caches, e.g. 'swinv2' or 'swinv2_clahe'."""
    return MODEL + ('_clahe' if CLAHE else '')


def result_dir(key=None):
    """results/<model>[/<dataset>]"""
    root = RESULTS_DIR / model_tag()
    return root / key if key else root
