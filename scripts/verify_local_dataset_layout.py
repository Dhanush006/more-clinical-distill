#!/usr/bin/env python3
"""
scripts/verify_local_dataset_layout.py

After running slurm/download_subset.slurm, verify that the downloaded
files match what the manifest expects.  Prints a pass/fail summary and
writes a report to logs/layout_verification.txt.

Usage:
    python scripts/verify_local_dataset_layout.py [--config configs/paths.yaml]
    python scripts/verify_local_dataset_layout.py --subset 500
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def verify(cfg: dict, subset: int | None, report_path: str) -> bool:
    manifest_path = cfg["manifest_v1"]
    ecg_root = Path(cfg["mimic_ecg_root"])
    cxr_root = Path(cfg["mimic_cxr_root"])

    print(f"Manifest:  {manifest_path}")
    print(f"ECG root:  {ecg_root}")
    print(f"CXR root:  {cxr_root}")

    df = pd.read_csv(manifest_path)
    if subset:
        df = df.head(subset)
        print(f"Subset:    first {subset} rows")

    n = len(df)
    ecg_missing_hea, ecg_missing_dat, cxr_missing = [], [], []

    for _, row in df.iterrows():
        stem = row["ecg_path"].rstrip("/")
        # PhysioNet wget mirrors the URL path under the destination root
        # URL: https://physionet.org/files/mimic-iv-ecg/1.0/{stem}.hea
        # Stored at: {ecg_root}/physionet.org/files/mimic-iv-ecg/1.0/{stem}.hea
        hea = ecg_root / "physionet.org" / "files" / "mimic-iv-ecg" / "1.0" / (stem + ".hea")
        dat = ecg_root / "physionet.org" / "files" / "mimic-iv-ecg" / "1.0" / (stem + ".dat")
        if not hea.exists():
            ecg_missing_hea.append(str(hea))
        if not dat.exists():
            ecg_missing_dat.append(str(dat))

        jpg_rel = row["cxr_path"].lstrip("/")
        jpg = cxr_root / "physionet.org" / "files" / "mimic-cxr-jpg" / "2.0.0" / jpg_rel
        if not jpg.exists():
            cxr_missing.append(str(jpg))

    # ── Report ────────────────────────────────────────────────────────
    lines = []
    lines.append(f"=== Layout Verification Report ===")
    lines.append(f"Manifest rows checked: {n}")
    lines.append(f"ECG .hea missing:      {len(ecg_missing_hea)}")
    lines.append(f"ECG .dat missing:      {len(ecg_missing_dat)}")
    lines.append(f"CXR .jpg missing:      {len(cxr_missing)}")

    ok = (len(ecg_missing_hea) == 0 and
          len(ecg_missing_dat) == 0 and
          len(cxr_missing) == 0)

    lines.append(f"\nResult: {'PASS' if ok else 'FAIL'}")

    if not ok:
        if ecg_missing_hea:
            lines.append(f"\nFirst 10 missing .hea:")
            lines.extend(f"  {p}" for p in ecg_missing_hea[:10])
        if ecg_missing_dat:
            lines.append(f"\nFirst 10 missing .dat:")
            lines.extend(f"  {p}" for p in ecg_missing_dat[:10])
        if cxr_missing:
            lines.append(f"\nFirst 10 missing .jpg:")
            lines.extend(f"  {p}" for p in cxr_missing[:10])

    report = "\n".join(lines)
    print("\n" + report)

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report + "\n")
    print(f"\nReport written to: {report_path}")
    return ok


def main():
    parser = argparse.ArgumentParser(description="Verify downloaded dataset layout")
    parser.add_argument("--config", default="configs/paths.yaml")
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--report", default="logs/layout_verification.txt")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ok = verify(cfg, args.subset, args.report)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
