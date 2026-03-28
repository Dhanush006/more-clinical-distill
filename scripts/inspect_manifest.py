#!/usr/bin/env python3
"""
scripts/inspect_manifest.py

Quick sanity-check of the matched_patients_v1.csv manifest.
Prints column stats, path format examples, day_diff distribution,
and flags any unexpected nulls or duplicate subjects.

Usage:
    python scripts/inspect_manifest.py [--manifest PATH]
"""

import argparse
import pandas as pd
import yaml
from pathlib import Path


def load_paths_config(config_path: str = "configs/paths.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def inspect(manifest_path: str) -> None:
    print(f"Loading manifest: {manifest_path}")
    df = pd.read_csv(manifest_path)

    print(f"\n{'='*60}")
    print(f"Shape: {df.shape}  ({df.shape[0]} rows, {df.shape[1]} cols)")
    print(f"{'='*60}")

    print("\n--- Columns ---")
    print(df.dtypes.to_string())

    print("\n--- Null counts ---")
    nulls = df.isnull().sum()
    print(nulls.to_string())

    print("\n--- Duplicate subject_id ---")
    dups = df.duplicated(subset="subject_id").sum()
    print(f"  {dups} duplicate subject_id rows")

    print("\n--- Sample ECG paths (first 3) ---")
    for p in df["ecg_path"].head(3):
        print(f"  {p}")

    print("\n--- Sample CXR paths (first 3) ---")
    for p in df["cxr_path"].head(3):
        print(f"  {p}")

    print("\n--- day_diff distribution ---")
    dd = df["day_diff"].value_counts().sort_index()
    same_day = dd.get(0, 0)
    pct = same_day / len(df) * 100
    print(f"  Same-day (0):  {same_day:,}  ({pct:.1f}%)")
    print(f"  min={df['day_diff'].min()}, max={df['day_diff'].max()}, "
          f"mean={df['day_diff'].mean():.2f}")

    print("\n--- ECG path prefix distribution (top 5) ---")
    ecg_prefix = df["ecg_path"].str.extract(r'^(files/p\d+)/')[0]
    print(ecg_prefix.value_counts().head(5).to_string())

    print("\n--- CXR path extension check ---")
    ext = df["cxr_path"].str.extract(r'(\.\w+)$')[0].value_counts()
    print(ext.to_string())

    # Validate ECG path structure: should be stem (no .hea/.dat extension)
    has_ext = df["ecg_path"].str.endswith((".hea", ".dat")).sum()
    if has_ext > 0:
        print(f"\n  WARNING: {has_ext} ECG paths have a file extension "
              "(expected bare WFDB stems).")
    else:
        print("\n  ECG paths look correct (bare WFDB stems, no extensions).")

    print(f"\n{'='*60}")
    print("Inspection complete.")


def main():
    parser = argparse.ArgumentParser(description="Inspect matched_patients manifest")
    parser.add_argument("--manifest", default=None,
                        help="Path to manifest CSV (default: from configs/paths.yaml)")
    parser.add_argument("--config", default="configs/paths.yaml")
    args = parser.parse_args()

    if args.manifest:
        manifest_path = args.manifest
    else:
        cfg = load_paths_config(args.config)
        manifest_path = cfg["manifest_v1"]

    inspect(manifest_path)


if __name__ == "__main__":
    main()
