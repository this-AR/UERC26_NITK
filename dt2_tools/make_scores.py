"""Embed every image, mean-centre, and write the all-vs-all data file for main.sh.

Pipeline: model -> L2-normalize -> subtract global mean -> re-L2-normalize.
Scores come from the full N x N cosine matrix (S = E E^T, computed on the GPU).
The data file follows main.sh's 6-field format, one probe at a time (all j != i):
    sub1  pose1  sub2  pose2  genuine(1/0)  dissimilarity(= 1 - cosine)
max_min.sh from the scripts folder then min-max normalizes the 6th field.

Usage: python make_scores.py <AWE|ITW|NITJ|UBEAR>
"""
import sys
import time

import numpy as np
import torch

import models
from datasets import discover, output_dir
from models import load_image

BATCH_SIZE = 64
ROW_BLOCK = 256
DEVICE = torch.device('cuda')


def compute_embeddings(records):
    model, weights, params = models.build()
    print(f'Model: {weights.name} ({models.NAME}{", CLAHE" if models.CLAHE else ""}), '
          f'parameters: {params / 1e6:.4f} M')

    chunks = []
    with torch.no_grad():
        for start in range(0, len(records), BATCH_SIZE):
            batch = np.stack([load_image(p) for _, p in records[start:start + BATCH_SIZE]])
            chunks.append(model(torch.from_numpy(batch).to(DEVICE)).float().cpu())
            done = min(start + BATCH_SIZE, len(records))
            print(f'  [{done}/{len(records)}] embeddings computed', flush=True)
    return torch.cat(chunks).numpy().astype(np.float64), params


def main(key):
    out = output_dir(key)
    out.mkdir(parents=True, exist_ok=True)
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
    with (out / 'id_map.tsv').open('w') as f:
        f.write('subject_id\tpose_id\tsubject\timage_path\n')
        for (subject, path), sid, pid in zip(records, sub_ids, pose_ids):
            f.write(f'{sid}\t{pid}\t{subject}\t{path.as_posix()}\n')

    t0 = time.time()
    raw, params = compute_embeddings(records)
    print(f'Embeddings done in {time.time() - t0:.1f}s')
    l2 = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    global_mean = l2.mean(axis=0)
    centered = l2 - global_mean
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError('A centred embedding has zero norm.')
    final = centered / norms
    np.save(out / 'embeddings_raw.npy', raw.astype(np.float32))
    np.save(out / 'embeddings_final.npy', final.astype(np.float32))
    np.savetxt(out / 'global_mean_embedding.txt', global_mean)

    # Full cosine matrix in row blocks on the GPU -> data file, one probe per row.
    t0 = time.time()
    emb = torch.from_numpy(final).to(DEVICE)
    subj = np.asarray(sub_ids)
    left = [f'{s}\t{p}\t' for s, p in zip(sub_ids, pose_ids)]
    right_g = [f'{s}\t{p}\t1\t' for s, p in zip(sub_ids, pose_ids)]
    right_i = [f'{s}\t{p}\t0\t' for s, p in zip(sub_ids, pose_ids)]
    n_gen = n_imp = 0
    data_path = out / f'{key}.txt'
    with data_path.open('w', newline='\n') as f:
        for start in range(0, n, ROW_BLOCK):
            stop = min(start + ROW_BLOCK, n)
            dist = (1.0 - emb[start:stop] @ emb.T).cpu().numpy()
            lines = []
            for i in range(start, stop):
                row = dist[i - start]
                genuine = subj == subj[i]
                prefix = left[i]
                for j in range(n):
                    if j == i:
                        continue
                    r = right_g[j] if genuine[j] else right_i[j]
                    lines.append(f'{prefix}{r}{row[j]:.10f}')
                g = int(genuine.sum()) - 1
                n_gen += g
                n_imp += n - 1 - g
            f.write('\n'.join(lines))
            f.write('\n')
            print(f'  rows [{stop}/{n}] written ({time.time() - t0:.0f}s)', flush=True)

    with (out / 'run_info.txt').open('w') as f:
        f.write(f'dataset: {key}\n')
        f.write(f'model: {models.NAME}{" + CLAHE" if models.CLAHE else ""} '
                f'({params / 1e6:.4f} M parameters)\n')
        f.write('pipeline: embed -> L2-normalize -> subtract global mean -> re-L2-normalize\n')
        f.write('score: dissimilarity = 1 - cosine (then max_min.sh min-max normalization)\n')
        f.write('pairs: all ordered pairs i != j (each unordered pair appears twice)\n')
        f.write(f'images: {n}\nsubjects: {len(subjects)}\n')
        f.write(f'ordered pairs: {n * (n - 1)}\ngenuine pairs: {n_gen}\nimpostor pairs: {n_imp}\n')
        f.write(f'subjects with a single image (no genuine match possible): {len(singletons)}'
                f'{" " + ", ".join(singletons) if singletons else ""}\n')
    print(f'Wrote {data_path} ({n_gen} genuine, {n_imp} impostor) in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main(sys.argv[1])
