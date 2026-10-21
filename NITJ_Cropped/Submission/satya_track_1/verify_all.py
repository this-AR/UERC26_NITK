"""
Cross-verification script for NITJ_Cropped dataset + solution pipeline.
Run from: NITJ_Cropped/Submission/satya_track_1/
"""
import sys, os, csv, configparser
sys.path.append("../../")

ROOT = "../../Cropped"
SUBMISSION = "."

SEP = "=" * 65


# ── CHECK 1: CSV files exist ────────────────────────────────────────────────
print(SEP)
print("CHECK 1: CSV files exist in Cropped/")
print(SEP)
expected_csvs = [
    "image_list.csv", "dataset_split.csv",
    "pairs_train.csv", "pairs_val.csv", "pairs_test.csv",
]
all_csvs_ok = True
for f in expected_csvs:
    path = os.path.join(ROOT, f)
    exists = os.path.exists(path)
    size   = os.path.getsize(path) if exists else 0
    lines  = sum(1 for _ in open(path)) if exists else 0
    status = "OK     " if exists else "MISSING"
    print(f"  [{status}] {f:<30s}  {lines:>7} rows  {size:>12,} bytes")
    if not exists:
        all_csvs_ok = False
if not all_csvs_ok:
    print("  FAIL: One or more CSV files are missing!")
    sys.exit(1)
print()


# ── CHECK 2: image_list.csv path format & L/R counts ───────────────────────
print(SEP)
print("CHECK 2: image_list.csv  -- path format, subject IDs, L/R balance")
print(SEP)
subjects_seen = set()
lr_counts = {"L": 0, "R": 0, "other": 0}
missing_files = 0
bad_depth = 0

with open(os.path.join(ROOT, "image_list.csv")) as f:
    for row in csv.reader(f, delimiter="\t"):
        p = row[0]
        parts = p.split("/")
        if len(parts) != 4:
            bad_depth += 1
            continue
        subj, lr, angle_var, fname = parts
        subjects_seen.add(subj)
        lr_counts[lr if lr in ("L", "R") else "other"] += 1
        if not os.path.exists(os.path.join(ROOT, p)):
            missing_files += 1

l_count = lr_counts["L"]
r_count = lr_counts["R"]
total_imgs = l_count + r_count
print(f"  Unique subjects  : {len(subjects_seen)}  (expected 510)")
print(f"  L images         : {l_count}  (expected 5100)")
print(f"  R images         : {r_count}  (expected 5100)")
print(f"  Total images     : {total_imgs}  (expected 10200)")
print(f"  Wrong path depth : {bad_depth}  (expected 0)")
print(f"  Missing on disk  : {missing_files}  (expected 0)")
assert len(subjects_seen) == 510,  "Subject count mismatch!"
assert l_count == 5100,            "L image count mismatch!"
assert r_count == 5100,            "R image count mismatch!"
assert missing_files == 0,         "Some images are missing from disk!"
print("  PASS")
print()


# ── CHECK 3: dataset_split.csv counts & no leakage ─────────────────────────
print(SEP)
print("CHECK 3: dataset_split.csv  -- split counts and subject-level test isolation")
print(SEP)
split_counts = {"train": 0, "val": 0, "test": 0}
split_subjects = {"train": set(), "val": set(), "test": set()}

with open(os.path.join(ROOT, "dataset_split.csv")) as f:
    for row in csv.reader(f, delimiter="\t"):
        img_path, split = row[0], row[1]
        split_counts[split] = split_counts.get(split, 0) + 1
        subj = img_path.split("/")[0]
        split_subjects[split].add(subj)

total = sum(split_counts.values())
train_test_overlap = split_subjects["train"] & split_subjects["test"]
val_test_overlap   = split_subjects["val"]   & split_subjects["test"]

print(f"  train={split_counts['train']}  val={split_counts['val']}  test={split_counts['test']}  total={total}")
print(f"  Train subjects : {len(split_subjects['train'])}")
print(f"  Val subjects   : {len(split_subjects['val'])}")
print(f"  Test subjects  : {len(split_subjects['test'])}  (expected ~51, whole subjects held out)")
print(f"  Train/test subject overlap : {len(train_test_overlap)}  (expected 0)")
print(f"  Val/test   subject overlap : {len(val_test_overlap)}   (expected 0)")
assert total == 10200,                     "Total image count mismatch!"
assert len(train_test_overlap) == 0,       "Train/test subject leakage!"
assert len(val_test_overlap) == 0,         "Val/test subject leakage!"
print("  PASS")
print()


# ── CHECK 4: pairs_test.csv label consistency ───────────────────────────────
print(SEP)
print("CHECK 4: pairs_test.csv  -- label consistency (same/diff subject)")
print(SEP)
pos, neg, bad = 0, 0, 0
with open(os.path.join(ROOT, "pairs_test.csv")) as f:
    for row in csv.reader(f, delimiter="\t"):
        if len(row) != 3:
            bad += 1
            continue
        img1, img2, lbl = row
        s1 = img1.split("/")[0]
        s2 = img2.split("/")[0]
        if lbl == "1":
            pos += 1
            if s1 != s2:
                bad += 1   # positive pair but different subjects!
        else:
            neg += 1
            if s1 == s2:
                bad += 1   # negative pair but same subject!

print(f"  Positive pairs (same subject): {pos}")
print(f"  Negative pairs (diff subject): {neg}")
print(f"  Total                        : {pos + neg}")
print(f"  Mislabeled / malformed rows  : {bad}  (expected 0)")
assert bad == 0, "Pairs CSV has mislabeled rows!"
print("  PASS")
print()


# ── CHECK 5: Dataset class loading ─────────────────────────────────────────
print(SEP)
print("CHECK 5: UERCDataset & UERCPairwiseDataset  -- load and count")
print(SEP)
from uerc26_dataset import UERCDataset, UERCPairwiseDataset

for split in ("train", "val", "test"):
    ds = UERCDataset(split, root_dir=ROOT,
                     full_image_list_csv="image_list.csv",
                     data_split_csv="dataset_split.csv")
    print(f"  UERCDataset [{split:5s}]: {len(ds):>6} images,  {ds.num_of_classes} classes")

for split in ("train", "val", "test"):
    ds = UERCPairwiseDataset(split, root_dir=ROOT,
                              full_image_list_csv="image_list.csv",
                              data_split_csv="dataset_split.csv",
                              pairs_csv=f"pairs_{split}.csv")
    print(f"  UERCPairwiseDataset [{split:5s}]: {len(ds):>7} pairs")
print("  PASS")
print()


# ── CHECK 6: config.ini ──────────────────────────────────────────────────────
print(SEP)
print("CHECK 6: config.ini  -- paths, track, weights file")
print(SEP)
cfg = configparser.ConfigParser()
cfg.read(os.path.join(SUBMISSION, "config.ini"))

track      = cfg.get("SUBMISSION", "track")
name       = cfg.get("SUBMISSION", "name")
model_cls  = cfg.get("MODEL", "model_class")
weights    = cfg.get("MODEL", "weights")
data_path  = cfg.get("DATA", "data_path")
data_split = cfg.get("DATA", "data_split")

print(f"  track       = {track}   (expected T1)")
print(f"  name        = {name}")
print(f"  model_class = {model_cls}")
print(f"  weights     = {weights}")
print(f"  data_path   = {data_path}   (expected ../../Cropped)")
print(f"  data_split  = {data_split}  (expected test)")

weights_ok   = os.path.exists(os.path.join(SUBMISSION, weights))
data_path_ok = os.path.exists(os.path.join(SUBMISSION, data_path))
pairs_ok     = os.path.exists(os.path.join(SUBMISSION, data_path, f"pairs_{data_split}.csv"))

print(f"  weights file exists     : {weights_ok}")
print(f"  data_path exists        : {data_path_ok}")
print(f"  pairs_test.csv exists   : {pairs_ok}")
assert track == "T1",                    "Track should be T1!"
assert data_path == "../../Cropped",     "data_path must point to ../../Cropped!"
assert weights_ok,                       "Weights file not found!"
assert data_path_ok,                     "data_path directory not found!"
assert pairs_ok,                         "pairs_test.csv not found at data_path!"
print("  PASS")
print()


# ── CHECK 7: CustomUERCPairwiseDataset tensor shape & value range ───────────
print(SEP)
print("CHECK 7: CustomUERCPairwiseDataset  -- __getitem__ shapes & normalization")
print(SEP)
from solution import CustomUERCPairwiseDataset

custom_ds = CustomUERCPairwiseDataset(
    "test", ROOT,
    "image_list.csv", "dataset_split.csv",
    pairs_csv="pairs_test.csv"
)

# Sample 5 items to check consistency
for i in [0, 100, 500, 1000, 5000]:
    (t1, t2), lbl = custom_ds[i]
    assert tuple(t1.shape) == (3, 224, 224), f"img1 shape wrong at idx {i}: {t1.shape}"
    assert tuple(t2.shape) == (3, 224, 224), f"img2 shape wrong at idx {i}: {t2.shape}"
    assert str(t1.dtype) == "torch.float32",  f"Wrong dtype at idx {i}"

(t1, t2), lbl = custom_ds[0]
print(f"  img1 shape : {tuple(t1.shape)}  (expected (3, 224, 224))")
print(f"  img2 shape : {tuple(t2.shape)}  (expected (3, 224, 224))")
print(f"  dtype      : {t1.dtype}  (expected torch.float32)")
print(f"  label      : {lbl}")
print(f"  img1 range : [{float(t1.min()):.3f}, {float(t1.max()):.3f}]  (ImageNet-normalized, not clipped)")
print(f"  Sampled 5 items consistently -- PASS")
print()


# ── SUMMARY ──────────────────────────────────────────────────────────────────
print(SEP)
print("ALL 7 CHECKS PASSED -- pipeline is ready to run")
print(SEP)
