#!/usr/bin/env python3
"""
scripts/build_smoke_test_data.py

Builds a small smoke-test .npy file (default 50 subjects) from the
manifest using the same logic as convert_manifest_to_more_format.py.

Output: data/processed/smoke_test.npy

Usage:
    python scripts/build_smoke_test_data.py [--n 50] [--config configs/paths.yaml]
"""

import argparse
import sys
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description="Build smoke-test .npy subset")
    parser.add_argument("--n", type=int, default=50,
                        help="Number of subjects to include (default: 50)")
    parser.add_argument("--config", default="configs/paths.yaml")
    parser.add_argument("--out", default=None,
                        help="Output path (default: data/processed/smoke_test.npy)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_path = args.out or str(Path(cfg["processed_root"]) / "smoke_test.npy")

    # Reuse convert script logic with --subset
    sys.argv = [
        "convert_manifest_to_more_format.py",
        "--config", args.config,
        "--subset", str(args.n),
    ]

    # Temporarily override npy output paths so we write to smoke_test.npy
    # and don't overwrite real train/val/test splits.
    import numpy as np
    import pandas as pd
    import re

    # Inline the conversion for the smoke subset, writing a single file
    from scripts.convert_manifest_to_more_format import (
        load_config, CHEXPERT_COLS, clean_text,
        extract_radiology_note, label_fallback,
    )

    cfg = load_config(args.config)
    ecg_root      = Path(cfg["mimic_ecg_root"])
    cxr_root      = Path(cfg["mimic_cxr_root"])
    processed_root = Path(cfg["processed_root"])
    processed_root.mkdir(parents=True, exist_ok=True)

    ecg_files_root = ecg_root / "physionet.org" / "files" / "mimic-iv-ecg" / "1.0"
    cxr_files_root = cxr_root / "physionet.org" / "files" / "mimic-cxr-jpg" / "2.0.0"
    cxr_reports_root = cxr_files_root / "files"

    df = pd.read_csv(cfg["manifest_v1"]).head(args.n)
    print(f"Building smoke-test data from first {len(df)} subjects...")

    # Load optional metadata
    split_csv = Path(cfg["mimic_cxr_split"])
    split_map = {}
    if split_csv.exists():
        split_df = pd.read_csv(split_csv)
        split_map = dict(zip(split_df["study_id"].astype(str),
                             split_df["split"].str.strip()))

    chexpert_csv = Path(cfg["mimic_cxr_chexpert"])
    chexpert_df = None
    if chexpert_csv.exists():
        chexpert_df = pd.read_csv(chexpert_csv)
        chexpert_df["study_id"] = chexpert_df["study_id"].astype(str)
        chexpert_df = chexpert_df.set_index("study_id")

    ecg_meas_csv = Path(cfg["mimic_ecg_machine_reports"])
    ecg_note_map = {}
    if ecg_meas_csv.exists():
        meas_df = pd.read_csv(ecg_meas_csv, low_memory=False)
        if "study_id" in meas_df.columns and "report_0" in meas_df.columns:
            report_cols = [c for c in meas_df.columns if c.startswith("report_")]
            meas_df["ecg_note_raw"] = meas_df[report_cols].fillna("").agg(
                lambda r: " ".join(v for v in r if v), axis=1
            )
            ecg_note_map = dict(zip(meas_df["study_id"].astype(str),
                                    meas_df["ecg_note_raw"]))

    items = []
    for _, row in df.iterrows():
        ecg_stem_rel = row["ecg_path"].rstrip("/")
        cxr_jpg_rel  = row["cxr_path"].lstrip("/")
        cxr_study_id = str(int(row["cxr_study_id"])) if not pd.isna(row["cxr_study_id"]) else ""
        ecg_study_id = str(int(row["ecg_study_id"])) if not pd.isna(row["ecg_study_id"]) else ""

        xray_path = str(cxr_files_root / cxr_jpg_rel)
        ecg_stem  = str(ecg_files_root / ecg_stem_rel)

        if chexpert_df is not None and cxr_study_id in chexpert_df.index:
            row_labels = chexpert_df.loc[cxr_study_id, CHEXPERT_COLS].values.astype(float)
            row_labels = np.where(np.isnan(row_labels), 0.0, row_labels)
        else:
            row_labels = np.zeros(4, dtype=float)

        parts = cxr_jpg_rel.split("/")
        if len(parts) >= 4:
            txt_dir = cxr_reports_root / parts[2] / parts[3]
            txt_files = list(txt_dir.glob("*.txt")) if txt_dir.exists() else []
            xray_note = extract_radiology_note(txt_files[0]) if txt_files else label_fallback(row_labels)
        else:
            xray_note = label_fallback(row_labels)

        xray_note = "The report from Xray is: " + xray_note
        raw_ecg   = ecg_note_map.get(ecg_study_id, "")
        ecg_note  = "The report from ECG is: " + (clean_text(raw_ecg) if raw_ecg else "ECG note not available.")
        split_val = split_map.get(cxr_study_id, "train")
        if split_val == "validation":
            split_val = "validate"

        items.append([xray_path, ecg_stem, xray_note, ecg_note, row_labels, split_val])

    np.save(out_path, np.array(items, dtype=object), allow_pickle=True)
    print(f"Saved {len(items)} items → {out_path}")
    print("Split distribution:", {s: sum(1 for it in items if it[5] == s)
                                   for s in ["train", "validate", "test"]})


if __name__ == "__main__":
    main()
