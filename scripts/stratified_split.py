"""
scripts/stratified_split.py

Build a patient-level, multi-label-stratified 80/10/10 train/val/test split
using iterstrat.MultilabelStratifiedKFold operating at the *subject* level.

Steps:
  1. Concatenate all rows from existing more_{train,val,test}.npy (49,076 patients;
     each subject_id appears exactly once in our cohort because MIMIC-IV-ECG +
     MIMIC-CXR-JPG were paired one-to-one in scripts/convert_manifest_to_more_format.py).
  2. Build a per-subject 10-class ECG label vector (any positive across that subject's
     records, but in our case there is one record per subject so it equals item[4]).
  3. Multi-label stratified split:
        - First split: 80% train  vs  20% temp
        - Second split: 50/50 of temp  →  10% val + 10% test
     Each split preserves multi-label co-occurrence patterns.
  4. Write data/processed/more_strat_{train,val,test}.npy in the same 7-element
     row format as the legacy splits (only item[6] 'split' is rewritten).
  5. Print per-class distributions across the three splits + patient-overlap audit.

Usage:
    cd /scratch/user/dshekar/more-clinical-distill
    python scripts/stratified_split.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

REPO = Path(__file__).resolve().parent.parent
PROC = REPO / "data" / "processed"

ECG_NAMES = ["Normal", "SinusBrady", "SinusTachy", "AFib", "LBBB", "RBBB",
             "STEMI", "STIschemia", "AVBlock", "LVH"]


def load_all_rows():
    rows = []
    for split in ("train", "val", "test"):
        a = np.load(PROC / f"more_{split}.npy", allow_pickle=True)
        for r in a:
            rows.append(list(r))
    return rows


def build_subject_table(rows, sid_to_subject):
    """One row per (study, subject). Subject IDs from combined_reports.csv."""
    out = []
    for r in rows:
        ecg_stem = r[1]
        sid = ecg_stem.split("/")[-1]
        sub = sid_to_subject.get(sid, sid)
        out.append((sub, r))
    return out


def main():
    print("Loading legacy splits...")
    rows = load_all_rows()
    print(f"  total: {len(rows)} rows")

    df = pd.read_csv(REPO / "data" / "combined_reports.csv")
    sid_to_subject = dict(zip(df["ecg_study_id"].astype(str),
                              df["subject_id"].astype(str)))

    pairs = build_subject_table(rows, sid_to_subject)
    n_unique_subjects = len(set(p[0] for p in pairs))
    print(f"  unique subjects: {n_unique_subjects}")

    labels = np.stack([p[1][4].astype(int) for p in pairs])
    print(f"  label matrix: {labels.shape}, positives/class: {labels.sum(0)}")

    rng = np.random.RandomState(42)
    indices = np.arange(len(pairs))

    print("\nFirst split: 80% train | 20% temp")
    msss1 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, temp_idx = next(msss1.split(indices, labels))

    print("Second split: 50/50 of temp -> 10% val | 10% test")
    msss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    sub_labels = labels[temp_idx]
    val_local, test_local = next(msss2.split(np.arange(len(temp_idx)), sub_labels))
    val_idx  = temp_idx[val_local]
    test_idx = temp_idx[test_local]

    splits = {"train": train_idx, "validate": val_idx, "test": test_idx}

    train_subj = set(pairs[i][0] for i in train_idx)
    val_subj   = set(pairs[i][0] for i in val_idx)
    test_subj  = set(pairs[i][0] for i in test_idx)
    print(f"\nSubject overlaps  (must all be 0):")
    print(f"  train ∩ val:  {len(train_subj & val_subj)}")
    print(f"  train ∩ test: {len(train_subj & test_subj)}")
    print(f"  val   ∩ test: {len(val_subj & test_subj)}")

    print("\nPer-class distribution per split:")
    header = "split  | " + "  ".join(f"{n:<10}" for n in ECG_NAMES)
    print(header)
    print("-" * len(header))
    for name in ("train", "validate", "test"):
        idx = splits[name]
        labs = labels[idx]
        n = len(idx)
        cells = [f"{int(c)}({100*c/n:.1f}%)" for c in labs.sum(0)]
        print(f"{name:<6} ({n:5d}) | " + "  ".join(f"{c:<10}" for c in cells))

    print("\nWriting more_strat_{train,val,test}.npy...")
    for name, idx in splits.items():
        out_rows = []
        for i in idx:
            row = list(pairs[i][1])
            row[6] = name
            out_rows.append(row)
        arr = np.empty(len(out_rows), dtype=object)
        for k, r in enumerate(out_rows):
            arr[k] = r
        out_path = PROC / f"more_strat_{'val' if name == 'validate' else name}.npy"
        np.save(out_path, arr, allow_pickle=True)
        print(f"  {out_path}: {len(out_rows)} rows")

    print("\nDone.")


if __name__ == "__main__":
    main()
