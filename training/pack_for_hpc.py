"""Bundle every image listed in splits/manifest.csv, plus the splits, into one tar for the HPC.

Paths inside the tar are the manifest paths (relative to the project root), so after
`tar -xf ear_data.tar` on the HPC the manifest resolves without changes.
Images are stored as-is (already compressed). Writes hpc_package/ear_data.tar (git-ignored).

Usage (from the project root): python -m training.pack_for_hpc
Copy:  scp hpc_package/ear_data.tar fcse_shwetasharma@172.16.0.77:~/friend-gpu/data/
"""
import csv
import tarfile

from ear_eval.config import PROJECT_DIR

OUT = PROJECT_DIR / 'hpc_package' / 'ear_data.tar'


def main():
    with open(PROJECT_DIR / 'splits' / 'manifest.csv') as f:
        paths = [r['path'] for r in csv.DictReader(f)]
    OUT.parent.mkdir(exist_ok=True)
    with tarfile.open(OUT, 'w') as tar:
        tar.add(PROJECT_DIR / 'splits', arcname='splits')
        for i, p in enumerate(paths, 1):
            tar.add(PROJECT_DIR / p, arcname=p)
            if i % 50000 == 0:
                print(f'  {i}/{len(paths)} images added', flush=True)
    print(f'{OUT}: {len(paths)} images, {OUT.stat().st_size / 1e9:.2f} GB')


if __name__ == '__main__':
    main()
