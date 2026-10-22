"""Convert every image in splits/manifest.csv to one common format, in parallel on all CPU cores.

Output format (matches what uerc26_Model_training_(single_phase).ipynb expects):
- 224 x 224, stretched (no padding: letterboxing made UBEAR worse, see results/analysis/README.md)
- 3-channel RGB (greyscale UBEAR / NITJ TIFFs are replicated to 3 channels)
- WebP, quality 95 (about 10-15 KB per image)
Augmentation (colour, CLAHE, geometric) is NOT baked in; it happens at training time.

Writes <out>/<original path with .webp> and <out>/manifest.csv (same rows, `path` pointing at the
new files). Already-converted files are skipped, so an interrupted run can simply be restarted.

Usage (from the directory where ear_data.tar was extracted, e.g. on the HPC):
    python -m training.preprocess --root . --out data_224 --workers 32
    python -m training.preprocess --root . --out data_224 --limit 200      # quick check
"""
import argparse
import csv
import os
from multiprocessing import Pool
from pathlib import Path

import cv2

SIZE = 224
WEBP_QUALITY = 95


def convert(job):
    src, dst = job
    if os.path.exists(dst):
        return dst, 'skipped'
    image = cv2.imread(src, cv2.IMREAD_COLOR)  # always 3-channel BGR; greyscale is replicated
    if image is None:
        return dst, 'unreadable'
    image = cv2.resize(image, (SIZE, SIZE), interpolation=cv2.INTER_AREA if min(image.shape[:2]) > SIZE
                       else cv2.INTER_LINEAR)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    ok = cv2.imwrite(dst, image, [cv2.IMWRITE_WEBP_QUALITY, WEBP_QUALITY])
    return dst, 'ok' if ok else 'write failed'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.', help='directory holding splits/ and the image folders')
    ap.add_argument('--out', default='data_224')
    ap.add_argument('--manifest', default=None, help='manifest to convert (default: <root>/splits/manifest.csv)')
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    ap.add_argument('--limit', type=int, default=0, help='only convert the first N rows (for a quick check)')
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)

    with open(args.manifest or root / 'splits' / 'manifest.csv') as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)
    if args.limit:
        rows = rows[:args.limit]
    jobs = [(str(root / r['path']), str(out / Path(r['path']).with_suffix('.webp'))) for r in rows]

    status = {}
    with Pool(args.workers) as pool:
        for i, (dst, s) in enumerate(pool.imap_unordered(convert, jobs, chunksize=256), 1):
            status[s] = status.get(s, 0) + 1
            if s not in ('ok', 'skipped'):
                print(f'  {s}: {dst}', flush=True)
            if i % 25000 == 0:
                print(f'  {i}/{len(jobs)} done {status}', flush=True)

    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'manifest.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({**r, 'path': (out / Path(r['path']).with_suffix('.webp')).as_posix()})
    size = sum(p.stat().st_size for p in out.rglob('*.webp'))
    print(f'{len(jobs)} images -> {out} ({size / 1e9:.2f} GB), status {status}, workers {args.workers}')
    if status.get('unreadable') or status.get('write failed'):
        raise SystemExit('Some images failed; see the lines above.')


if __name__ == '__main__':
    main()
