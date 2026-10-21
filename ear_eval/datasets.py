"""Image discovery for the four ear datasets.

Every record is (subject, image_path), sorted by subject then file name.
Left and right ears of one person share the subject (NITJ, UBEAR).
"""
from .config import AWE_DIR, ITW_TEST_DIR, NITJ_DIR, UBEAR_DIR

KEYS = ['AWE', 'ITW', 'UBEAR', 'NITJ']


def _awe():
    return [(d.name, p) for d in sorted(AWE_DIR.iterdir()) if d.is_dir() and d.name.isdigit()
            for p in sorted(d.glob('*.png'))]


def _itw():
    return [(d.name, p) for d in sorted(ITW_TEST_DIR.iterdir()) if d.is_dir()
            for p in sorted(d.glob('*.png'))]


def _nitj():
    subjects = sorted((d for d in NITJ_DIR.iterdir() if d.is_dir()), key=lambda d: int(d.name))
    return [(d.name, p) for d in subjects for p in sorted(d.rglob('*.tiff'))]


def _ubear():
    # s_<subject>_<session>_<side>_<age>_<gender>_<pose>_<frame>.tiff
    records = []
    for p in sorted(UBEAR_DIR.glob('*.tiff')):
        parts = p.stem.split('_')
        if parts[0] != 's':
            raise ValueError(f'Unexpected image filename: {p.name}')
        records.append((f'{parts[0]}_{parts[1]}', p))
    return sorted(records, key=lambda r: (r[0], r[1].name))


_DISCOVER = {'AWE': _awe, 'ITW': _itw, 'UBEAR': _ubear, 'NITJ': _nitj}


def discover(key):
    return _DISCOVER[key]()
