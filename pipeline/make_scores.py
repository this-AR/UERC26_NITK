"""Step 1: embed every image, mean-centre, and write the all-vs-all data file for main.sh.

Pipeline: model -> L2-normalize -> subtract global mean -> re-L2-normalize.
Scores come from the full N x N cosine matrix (S = E E^T, computed on the GPU).
The data file follows main.sh's 6-field format, one probe at a time (all j != i):
    sub1  pose1  sub2  pose2  genuine(1/0)  dissimilarity(= 1 - cosine)
max_min.sh then min-max normalizes the 6th field.

Writes results/<model>/<KEY>/run_info.txt and, in results/<model>/<KEY>/_work/:
<KEY>.txt (data file), id_map.tsv and embeddings_final.npy for postprocess.py.

Usage (from the project root): python -m pipeline.make_scores <AWE|ITW|UBEAR|NITJ>
"""
import sys
import time

import numpy as np
import torch

from ear_eval.config import CLAHE, MODEL, result_dir
from ear_eval.datasets import discover
from ear_eval.embeddings import embed, mean_centre, model

ROW_BLOCK = 256


def write_data_file(path, emb, sub_ids, pose_ids):
    """Write all ordered pairs i != j, probe by probe; returns (genuine, impostor) counts."""
    n = len(emb)
    e = torch.from_numpy(emb).to('cuda')
    subj = np.asarray(sub_ids)
    left = [f'{s}\t{p}\t' for s, p in zip(sub_ids, pose_ids)]
    right_g = [f'{s}\t{p}\t1\t' for s, p in zip(sub_ids, pose_ids)]
    right_i = [f'{s}\t{p}\t0\t' for s, p in zip(sub_ids, pose_ids)]
    n_gen = n_imp = 0
    t0 = time.time()
    with path.open('w', newline='\n') as f:
        for start in range(0, n, ROW_BLOCK):
            stop = min(start + ROW_BLOCK, n)
            dist = (1.0 - e[start:stop] @ e.T).cpu().numpy()
            lines = []
            for i in range(start, stop):
                row = dist[i - start]
                genuine = subj == subj[i]
                prefix = left[i]
                for j in range(n):
                    if j != i:
                        lines.append(f'{prefix}{right_g[j] if genuine[j] else right_i[j]}{row[j]:.10f}')
                g = int(genuine.sum()) - 1
                n_gen += g
                n_imp += n - 1 - g
            f.write('\n'.join(lines))
            f.write('\n')
            print(f'  rows [{stop}/{n}] written ({time.time() - t0:.0f}s)', flush=True)
    return n_gen, n_imp


def main(key):
    out = result_dir(key)
    work = out / '_work'
    work.mkdir(parents=True, exist_ok=True)
    records = discover(key)
    n = len(records)
    subjects = sorted({s for s, _ in records}, key=lambda s: (len(s), s))
    subject_index = {s: i + 1 for i, s in enumerate(subjects)}
    print(f'{key}: {n} images, {len(subjects)} subjects, {n * (n - 1)} ordered pairs')

    # Integer subject / pose ids (identification.awk prints them with %d).
    sub_ids, pose_ids, counter = [], [], {}
    for subject, _ in records:
        counter[subject] = counter.get(subject, 0) + 1
        sub_ids.append(subject_index[subject])
        pose_ids.append(counter[subject])
    singletons = [s for s, c in counter.items() if c == 1]
    with (work / 'id_map.tsv').open('w') as f:
        f.write('subject_id\tpose_id\tsubject\timage_path\n')
        for (subject, path), sid, pid in zip(records, sub_ids, pose_ids):
            f.write(f'{sid}\t{pid}\t{subject}\t{path.as_posix()}\n')

    params = sum(p.numel() for p in model().parameters())
    t0 = time.time()
    final, _ = mean_centre(embed([p for _, p in records], progress=True))
    print(f'Embeddings done in {time.time() - t0:.1f}s')
    np.save(work / 'embeddings_final.npy', final.astype(np.float32))

    t0 = time.time()
    n_gen, n_imp = write_data_file(work / f'{key}.txt', final, sub_ids, pose_ids)
    print(f'Wrote {key}.txt ({n_gen} genuine, {n_imp} impostor) in {time.time() - t0:.0f}s')

    with (out / 'run_info.txt').open('w') as f:
        f.write(f'dataset: {key}\n')
        f.write(f'model: {MODEL}{" + CLAHE" if CLAHE else ""} ({params / 1e6:.4f} M parameters)\n')
        f.write('pipeline: embed -> L2-normalize -> subtract global mean -> re-L2-normalize\n')
        f.write('score: dissimilarity = 1 - cosine (then max_min.sh min-max normalization)\n')
        f.write('pairs: all ordered pairs i != j (each unordered pair appears twice)\n')
        f.write(f'images: {n}\nsubjects: {len(subjects)}\n')
        f.write(f'ordered pairs: {n * (n - 1)}\ngenuine pairs: {n_gen}\nimpostor pairs: {n_imp}\n')
        f.write(f'subjects with a single image (no genuine match possible): {len(singletons)}'
                f'{" " + ", ".join(singletons) if singletons else ""}\n')


if __name__ == '__main__':
    main(sys.argv[1])
