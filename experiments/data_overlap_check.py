"""Train/test leakage check: which evaluation images also appear in UERC26 public (the training data)?

Two tests per image: exact bytes (MD5) and near-identical content (16x16 average hash), so
re-encoded copies are caught too. Reports, per dataset, the matches and the UERC split they sit in.
Writes results/analysis/data_overlap.txt. Reads all 248k UERC images: about 10-15 minutes.

Usage (from the project root): python -m experiments.data_overlap_check
"""
import collections
import csv
import hashlib

import cv2
import numpy as np

from ear_eval.config import ANALYSIS_DIR, PROJECT_DIR, UERC_PUBLIC_DIR
from ear_eval.datasets import KEYS, discover


def md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def ahash(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    small = cv2.resize(image, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32)
    return (small > small.mean()).tobytes()


def main():
    uerc = [p for p in UERC_PUBLIC_DIR.rglob('*') if p.suffix.lower() in ('.png', '.jpg')]
    by_md5, by_hash = {}, collections.defaultdict(list)
    for p in uerc:
        by_md5[md5(p)] = p
        h = ahash(p)
        if h:
            by_hash[h].append(p)
    split = {r[0]: r[1] for r in csv.reader(open(UERC_PUBLIC_DIR / 'dataset_split.csv'), delimiter='\t') if len(r) > 1}

    lines = [f'UERC26 public images hashed: {len(uerc)}', '']
    for key in KEYS:
        files = [p for _, p in discover(key)]
        exact = [(p, by_md5[m]) for p in files if (m := md5(p)) in by_md5]
        near = [(p, by_hash[h][0]) for p in files if (h := ahash(p)) and h in by_hash]
        lines.append(f'{key}: {len(files)} images | exact byte matches: {len(exact)} | near-identical: {len(near)} '
                     f'| UERC subjects hit: {len({q.parent.name for _, q in near})}')
        for p, q in (exact or near)[:3]:
            lines.append(f'    {p.relative_to(PROJECT_DIR).as_posix()} == {q.relative_to(UERC_PUBLIC_DIR).as_posix()}')
        if near:
            where = collections.Counter(split.get(q.relative_to(UERC_PUBLIC_DIR).as_posix(), '?') for _, q in near)
            lines.append(f'    UERC split of the matches: {dict(where)}')
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    (ANALYSIS_DIR / 'data_overlap.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
