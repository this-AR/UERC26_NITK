"""Build the frozen, subject-disjoint train / val / test splits shared by every retraining track.

Rules (see results/analysis/findings_and_plan.md):
- Split by SUBJECT, never by image, so near-identical video frames can't leak into the test set.
- Training classes are subject + ear side wherever the side is known (UBEAR, NITJ).
- UBEAR: 30% of subjects for training (stratified by gender and by having a session-2 capture),
  6 of those held out as a subject-disjoint validation set, the other 70% are the test set.
  Training frames: at most 5 per video (subject+session+side+pose), evenly spaced.
- UERC26 public and NITJ keep their official splits (train / val share subjects, test is disjoint).
- ITW: test only. AWE: only the 11 subjects that are in UERC's test split (the rest is UERC training data).

Writes splits/manifest.csv (one row per image) and splits/README.md (counts).
Usage (from the project root): python -m training.make_splits
"""
import collections
import csv

import numpy as np

from ear_eval.config import AWE_DIR, ITW_TEST_DIR, NITJ_DIR, PROJECT_DIR, UBEAR_DIR, UERC_PUBLIC_DIR

SEED = 42
UBEAR_TRAIN_FRACTION = 0.30
UBEAR_VAL_SUBJECTS = 6
FRAMES_PER_VIDEO = 5
OUT = PROJECT_DIR / 'splits'
FIELDS = ['path', 'dataset', 'subject', 'side', 'session', 'pose', 'sequence', 'split', 'class_key']


def rel(path):
    return path.relative_to(PROJECT_DIR).as_posix()


def row(path, dataset, subject, split, side='', session='', pose='', sequence='', class_key=''):
    return dict(path=rel(path), dataset=dataset, subject=subject, side=side, session=session, pose=pose,
                sequence=sequence, split=split, class_key=class_key)


def uerc():
    rows, test_subjects = [], set()
    with open(UERC_PUBLIC_DIR / 'dataset_split.csv') as f:
        for r in csv.reader(f, delimiter='\t'):
            if len(r) < 2:
                continue
            subject = r[0].split('/')[0]
            if r[1] == 'test':
                test_subjects.add(subject)
            rows.append(row(UERC_PUBLIC_DIR / r[0], 'UERC', subject, r[1],
                            class_key=f'UERC:{subject}' if r[1] != 'test' else ''))
    return rows, test_subjects


def awe(uerc_test_subjects):
    # AWE subject NNN == UERC subject 0NNN (byte-identical images, results/analysis/data_overlap.txt).
    rows = []
    for d in sorted(AWE_DIR.iterdir()):
        if d.is_dir() and d.name.isdigit() and f'0{d.name}' in uerc_test_subjects:
            rows += [row(p, 'AWE', d.name, 'test') for p in sorted(d.glob('*.png'))]
    return rows


def itw():
    return [row(p, 'ITW', d.name, 'test') for d in sorted(ITW_TEST_DIR.iterdir()) if d.is_dir()
            for p in sorted(d.glob('*.png'))]


def nitj():
    rows = []
    with open(NITJ_DIR / 'dataset_split.csv') as f:
        for r in csv.reader(f, delimiter='\t'):
            if len(r) < 2:
                continue
            subject, side = r[0].split('/')[:2]
            rows.append(row(NITJ_DIR / r[0], 'NITJ', subject, r[1], side=side,
                            class_key=f'NITJ:{subject}:{side}' if r[1] != 'test' else ''))
    return rows


def ubear(rng):
    videos = collections.defaultdict(list)
    info = collections.defaultdict(lambda: {'sessions': set(), 'gender': ''})
    for p in sorted(UBEAR_DIR.glob('*.tiff')):
        _, sid, session, side, _, gender, pose, _ = p.stem.split('_')
        subject = f's_{sid}'
        videos[(subject, session, side, pose)].append(p)
        info[subject]['sessions'].add(session)
        info[subject]['gender'] = gender.upper()

    # Stratified subject sampling: (gender, has session 2) strata, 30% of each to training.
    strata = collections.defaultdict(list)
    for subject, v in sorted(info.items()):
        strata[(v['gender'], '2' in v['sessions'])].append(subject)
    train_subjects = []
    for key in sorted(strata):
        members = strata[key]
        k = int(round(UBEAR_TRAIN_FRACTION * len(members)))
        train_subjects += list(rng.choice(members, size=k, replace=False))
    train_subjects = sorted(train_subjects)
    val_subjects = set(rng.choice(train_subjects, size=UBEAR_VAL_SUBJECTS, replace=False))

    rows = []
    for (subject, session, side, pose), frames in sorted(videos.items()):
        sequence = f'{subject}|{session}|{side}|{pose}'
        if subject in train_subjects:
            split = 'val' if subject in val_subjects else 'train'
            # Evenly spaced frames, at most FRAMES_PER_VIDEO per video: consecutive frames are near-duplicates.
            keep = np.unique(np.linspace(0, len(frames) - 1, min(FRAMES_PER_VIDEO, len(frames))).round().astype(int))
            frames = [frames[i] for i in keep]
            class_key = f'UBEAR:{subject}:{side}'
        else:
            split, class_key = 'test', ''
        rows += [row(p, 'UBEAR', subject, split, side, session, pose, sequence, class_key) for p in frames]
    return rows


def summary(rows):
    lines = ['# Frozen splits for retraining', '',
             'Generated by `python -m training.make_splits` (seed 42). One row per image in `manifest.csv`.', '',
             'Columns:',
             '- `path`: relative to the project root.',
             '- `split`: `train`, `val` or `test`.',
             '- `class_key`: the training class (subject + ear side when known). Empty for test rows.',
             '- UBEAR only: `session`, `pose` and `sequence` (one video).', '',
             '**Test subjects are disjoint from training subjects in every dataset.** Never train on `split == test`.', '',
             '| Dataset | Split | Images | Subjects | Classes |', '|---|---|---|---|---|']
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r['dataset'], r['split'])].append(r)
    order = {'UERC': 0, 'NITJ': 1, 'UBEAR': 2, 'ITW': 3, 'AWE': 4}
    for (ds, sp), rs in sorted(groups.items(), key=lambda x: (order[x[0][0]], ['train', 'val', 'test'].index(x[0][1]))):
        classes = len({r['class_key'] for r in rs if r['class_key']}) or '-'
        lines.append(f'| {ds} | {sp} | {len(rs)} | {len({r["subject"] for r in rs})} | {classes} |')
    lines += ['', 'Notes:',
              '- **UERC / NITJ:** the official splits are kept. Their `val` shares subjects with `train` (a closed-set check). '
              'Their `test` subjects are held out.',
              '- **UBEAR:** 30% of subjects train, stratified by gender and session-2 capture. 6 of those subjects form a '
              'subject-disjoint `val`. The other 70% are `test`. Training and validation keep at most 5 evenly spaced frames per video; '
              'test keeps every frame.',
              '- **AWE:** only the 11 subjects in UERC\'s test split. The other 89 AWE subjects are UERC training data.',
              '- **ITW:** test only (no training part is available).',
              '- **Suggested batch mix for the joint fine-tune:** about 60% UERC, 20% NITJ, 20% UBEAR by sampling share '
              '(raw counts would leave UBEAR under 1%).']
    return '\n'.join(lines) + '\n'


def main():
    rng = np.random.default_rng(SEED)
    uerc_rows, uerc_test = uerc()
    rows = uerc_rows + nitj() + ubear(rng) + itw() + awe(uerc_test)

    # Sanity checks: no subject in both train/val and test within a dataset; every path exists.
    by_ds = collections.defaultdict(lambda: collections.defaultdict(set))
    for r in rows:
        by_ds[r['dataset']][r['split'] == 'test'].add(r['subject'])
    for ds, s in by_ds.items():
        assert not (s[True] & s[False]), f'{ds}: test subjects overlap training subjects'
    missing = [r['path'] for r in rows if not (PROJECT_DIR / r['path']).exists()]
    assert not missing, f'{len(missing)} missing files, e.g. {missing[:3]}'

    OUT.mkdir(exist_ok=True)
    with (OUT / 'manifest.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    text = summary(rows)
    (OUT / 'README.md').write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
