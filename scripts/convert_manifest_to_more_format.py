#!/usr/bin/env python3
"""
scripts/convert_manifest_to_more_format.py

Converts matched_patients_v1.csv + MIMIC metadata into the .npy format
expected by MoRE's MultiModalData dataset class.

MoRE item format (list of 7 elements per row):
  [0] xray_path    str  — absolute path to .jpg
  [1] ecg_stem     str  — absolute path stem (no extension) for wfdb.rdsamp()
  [2] xray_note    str  — radiology report (FINDINGS + IMPRESSION)
  [3] ecg_note     str  — ECG machine report / measurement text
  [4] ecg_labels   np.ndarray shape (10,) — ECG rhythm classes (multi-label):
                          [Normal, Sinus bradycardia, Sinus tachycardia,
                           Atrial fibrillation, LBBB, RBBB,
                           ST elevation MI, ST ischemia, AV block, LVH]
                          values: 1.0 (present), 0.0 (absent)
  [5] pulm_labels  np.ndarray shape (4,) — CheXpert pulmonary subset:
                          [Atelectasis, Cardiomegaly, Edema, Pleural Effusion]
                          values: 1.0 (present), 0.0 (absent), -1.0 (uncertain)
  [6] split        str  — 'train', 'validate', or 'test'

Outputs:
  data/processed/more_train.npy
  data/processed/more_val.npy
  data/processed/more_test.npy

Prerequisites:
  - Download lists built + data downloaded (Phase 2)
  - MIMIC-CXR-JPG metadata CSVs present:
      mimic-cxr-2.0.0-split.csv
      mimic-cxr-2.0.0-chexpert.csv
  - MIMIC-IV-ECG machine_measurements.csv present
  - Radiology .txt reports present under data/mimic-cxr-jpg/files/

Usage:
    python scripts/convert_manifest_to_more_format.py [--config configs/paths.yaml]
    python scripts/convert_manifest_to_more_format.py --subset 500  # smoke-test
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# CheXpert label columns we use (indices match MoRE paper: 0,1,3,9)
CHEXPERT_COLS = ["Atelectasis", "Cardiomegaly", "Edema", "Pleural Effusion"]

# ── ECG rhythm class definitions (multi-label, from machine_measurements) ────
# Each entry: (class_name, compiled_regex)
# Keyword matching is case-insensitive on the concatenated report string.
# "Normal ECG" uses a negative lookbehind so "Abnormal ECG" is not matched.
ECG_RHYTHM_CLASSES = [
    "Normal",
    "Sinus bradycardia",
    "Sinus tachycardia",
    "Atrial fibrillation",
    "LBBB",
    "RBBB",
    "ST elevation MI",
    "ST ischemia",
    "AV block",
    "LVH",
]

_ECG_PATTERNS = [
    re.compile(r'(?<![a-z])normal ecg\b', re.I),                          # Normal ECG (not Abnormal ECG)
    re.compile(r'\bsinus bradycardia\b', re.I),                           # Sinus bradycardia
    re.compile(r'\bsinus tachycardia\b', re.I),                           # Sinus tachycardia
    re.compile(r'\batrial fibrillation\b', re.I),                         # Atrial fibrillation
    re.compile(r'\bleft bundle branch block\b', re.I),                    # LBBB
    re.compile(r'\bright bundle branch block\b|\brbbb\b', re.I),          # RBBB
    re.compile(r'\bacute st elevation\b|\bst elevation mi\b', re.I),      # ST elevation MI
    re.compile(r'\bmyocardial ischemia\b', re.I),                         # ST ischemia
    re.compile(r'\ba-v block\b|\bav block\b|\bdegree a-v\b', re.I),       # AV block
    re.compile(r'\bleft ventricular hypertrophy\b|\blvh\b', re.I),        # LVH
]


def parse_ecg_rhythm_labels(report_text: str) -> np.ndarray:
    """
    Parse ECG machine report text into a 10-dim multi-label binary vector.
    Args:
        report_text: concatenated report_0..report_N strings (space-joined)
    Returns:
        np.ndarray shape (10,) with 0.0 / 1.0 values
    """
    text = report_text.lower()
    return np.array([1.0 if pat.search(text) else 0.0 for pat in _ECG_PATTERNS],
                    dtype=np.float32)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Text helpers ──────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = re.sub(r'[_\u2013\u2014\u2015\u2212\uFE58\uFE63\uFF0D]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_radiology_note(txt_path: Path) -> str:
    """Extract FINDINGS + IMPRESSION from a radiology .txt report."""
    try:
        content = txt_path.read_text(encoding="utf-8", errors="ignore")
    except FileNotFoundError:
        return ""
    fi = content.find("FINDINGS:")
    ii = content.find("IMPRESSION:")
    findings, impression = "", ""
    if fi != -1:
        end = ii if ii != -1 else len(content)
        findings = content[fi:end].strip()
    if ii != -1:
        impression = content[ii:].strip()
    note = (findings + "\n" + impression).strip()
    return clean_text(note) if note else clean_text(content)


def label_fallback(labels_arr: np.ndarray) -> str:
    """Generate a text note from CheXpert labels when no report is available."""
    present = [CHEXPERT_COLS[i] for i, v in enumerate(labels_arr) if v == 1.0]
    uncertain = [CHEXPERT_COLS[i] for i, v in enumerate(labels_arr) if v == -1.0]
    parts = []
    if present:
        parts.append(", ".join(present) + " is present")
    if uncertain:
        parts.append("uncertain finding of " + ", ".join(uncertain))
    return ". ".join(parts) if parts else "No significant finding."


# ── Main conversion ───────────────────────────────────────────────────

def convert(cfg: dict, subset: int | None) -> None:
    manifest_path = cfg["manifest_v1"]
    ecg_root = Path(cfg["mimic_ecg_root"])
    cxr_root = Path(cfg["mimic_cxr_root"])
    processed_root = Path(cfg["processed_root"])
    processed_root.mkdir(parents=True, exist_ok=True)

    # Physionet wget mirrors URL structure under dest root
    ecg_files_root = ecg_root / "physionet.org" / "files" / "mimic-iv-ecg" / "1.0"
    cxr_files_root = cxr_root / "physionet.org" / "files" / "mimic-cxr-jpg" / "2.0.0"
    cxr_reports_root = cxr_files_root / "files"  # radiology .txt files

    print("Loading manifest...")
    df = pd.read_csv(manifest_path)
    if subset:
        df = df.head(subset)
        print(f"  Subset: {subset} rows")

    # ── Load MIMIC-CXR split ──────────────────────────────────────────
    split_csv = Path(cfg["mimic_cxr_split"])
    if not split_csv.exists():
        print(f"WARNING: split CSV not found: {split_csv}")
        print("  All rows will be assigned split='train'. Download metadata first.")
        split_df = None
    else:
        split_df = pd.read_csv(split_csv)
        # columns: subject_id, study_id, dicom_id, split
        split_map = dict(zip(split_df["study_id"].astype(str),
                             split_df["split"].str.strip()))

    # ── Load CheXpert labels ──────────────────────────────────────────
    chexpert_csv = Path(cfg["mimic_cxr_chexpert"])
    if not chexpert_csv.exists():
        print(f"WARNING: CheXpert CSV not found: {chexpert_csv}")
        print("  All labels will be set to 0.0. Download metadata first.")
        chexpert_df = None
    else:
        chexpert_df = pd.read_csv(chexpert_csv)
        # columns: subject_id, study_id, Atelectasis, Cardiomegaly, ...
        chexpert_df["study_id"] = chexpert_df["study_id"].astype(str)
        chexpert_df = chexpert_df.set_index("study_id")

    # ── Load ECG machine measurements (for ecg_note) ─────────────────
    ecg_meas_csv = Path(cfg["mimic_ecg_machine_reports"])
    if not ecg_meas_csv.exists():
        print(f"WARNING: ECG machine measurements not found: {ecg_meas_csv}")
        ecg_note_map = {}
    else:
        meas_df = pd.read_csv(ecg_meas_csv, low_memory=False)
        # Expect column 'study_id' and text cols; adapt as needed
        if "study_id" in meas_df.columns and "report_0" in meas_df.columns:
            # Concatenate report_0..report_7 columns that exist
            report_cols = [c for c in meas_df.columns if c.startswith("report_")]
            meas_df["ecg_note_raw"] = meas_df[report_cols].fillna("").agg(
                lambda r: " ".join(v for v in r if v), axis=1
            )
            ecg_note_map = dict(zip(meas_df["study_id"].astype(str),
                                    meas_df["ecg_note_raw"]))
        else:
            ecg_note_map = {}

    # ── Build items ───────────────────────────────────────────────────
    items = []
    n_missing_cxr = 0
    n_missing_ecg = 0
    n_missing_note = 0

    for _, row in df.iterrows():
        ecg_stem_rel = row["ecg_path"].rstrip("/")
        cxr_jpg_rel  = row["cxr_path"].lstrip("/")
        cxr_study_id = str(int(row["cxr_study_id"])) if not pd.isna(row["cxr_study_id"]) else ""
        ecg_study_id = str(int(row["ecg_study_id"])) if not pd.isna(row["ecg_study_id"]) else ""

        # Absolute paths
        xray_path = str(cxr_files_root / cxr_jpg_rel)
        ecg_stem  = str(ecg_files_root / ecg_stem_rel)

        # Track missing files
        if not Path(xray_path).exists():
            n_missing_cxr += 1
        if not Path(ecg_stem + ".hea").exists():
            n_missing_ecg += 1

        # Labels
        if chexpert_df is not None and cxr_study_id in chexpert_df.index:
            row_labels = chexpert_df.loc[cxr_study_id, CHEXPERT_COLS].values.astype(float)
            # Replace NaN with 0.0
            row_labels = np.where(np.isnan(row_labels), 0.0, row_labels)
        else:
            row_labels = np.zeros(4, dtype=float)

        # Xray note: look for {p_dir}/{s_dir}/*.txt
        # CXR path format: files/p19/p19128402/s53334082/abc.jpg
        parts = cxr_jpg_rel.split("/")
        if len(parts) >= 4:
            # parts: ['files', 'p19', 'p19128402', 's53334082', 'abc.jpg']
            p2_dir = parts[2]  # e.g. p19128402
            s_dir  = parts[3]  # e.g. s53334082
            txt_dir = cxr_reports_root / p2_dir / s_dir
            txt_files = list(txt_dir.glob("*.txt")) if txt_dir.exists() else []
            if txt_files:
                xray_note = extract_radiology_note(txt_files[0])
            else:
                xray_note = label_fallback(row_labels)
                n_missing_note += 1
        else:
            xray_note = label_fallback(row_labels)
            n_missing_note += 1

        xray_note = "The report from Xray is: " + xray_note

        # ECG note
        raw_ecg = ecg_note_map.get(ecg_study_id, "")
        ecg_note = "The report from ECG is: " + (clean_text(raw_ecg) if raw_ecg else "ECG note not available.")

        # ECG rhythm labels (multi-label, 10 classes)
        ecg_labels = parse_ecg_rhythm_labels(raw_ecg)

        # Split
        if split_df is not None:
            split_val = split_map.get(cxr_study_id, "train")
            # MoRE uses 'validate' (not 'val')
            if split_val == "validate":
                split_val = "validate"
            elif split_val == "validation":
                split_val = "validate"
        else:
            split_val = "train"

        # item[4]=ecg_labels(10,)  item[5]=pulm_labels(4,)  item[6]=split
        items.append([xray_path, ecg_stem, xray_note, ecg_note, ecg_labels, row_labels, split_val])

    # ── Split and save ────────────────────────────────────────────────
    train_items    = [it for it in items if it[6] == "train"]
    val_items      = [it for it in items if it[6] == "validate"]
    test_items     = [it for it in items if it[6] == "test"]

    for out_path, data in [
        (cfg["more_npy_train"], train_items),
        (cfg["more_npy_val"],   val_items),
        (cfg["more_npy_test"],  test_items),
    ]:
        np.save(out_path, np.array(data, dtype=object), allow_pickle=True)
        print(f"Saved {len(data):,} items → {out_path}")

    print(f"\nSummary:")
    print(f"  Total rows:         {len(items):,}")
    print(f"  Train:              {len(train_items):,}")
    print(f"  Validate:           {len(val_items):,}")
    print(f"  Test:               {len(test_items):,}")
    print(f"  Missing CXR files:  {n_missing_cxr:,}")
    print(f"  Missing ECG files:  {n_missing_ecg:,}")
    print(f"  Missing xray notes: {n_missing_note:,}  (label fallback used)")


def main():
    parser = argparse.ArgumentParser(description="Convert manifest to MoRE .npy format")
    parser.add_argument("--config", default="configs/paths.yaml")
    parser.add_argument("--subset", type=int, default=None,
                        help="Limit to first N subjects (smoke-test)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    convert(cfg, args.subset)


if __name__ == "__main__":
    main()
