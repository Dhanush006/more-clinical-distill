#!/usr/bin/env python3
"""
scripts/build_download_lists.py

Reads matched_patients_v1.csv and produces two text files:
  - manifests/ecg_download_list.txt   (one PhysioNet URL per line, .hea + .dat)
  - manifests/cxr_download_list.txt   (one PhysioNet URL per line, .jpg)

URLs use the standard PhysioNet files endpoint and are suitable for
wget/curl with --user / --password.

Usage:
    python scripts/build_download_lists.py [--config configs/paths.yaml]
    python scripts/build_download_lists.py --subset 500   # first N subjects
"""

import argparse
from pathlib import Path

import pandas as pd
import yaml


MIMIC_ECG_BASE = "https://physionet.org/files/mimic-iv-ecg/1.0/"
MIMIC_CXR_BASE = "https://physionet.org/files/mimic-cxr-jpg/2.1.0/"


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_lists(manifest_path: str,
                ecg_out: str,
                cxr_out: str,
                subset: int | None = None) -> None:
    print(f"Reading manifest: {manifest_path}")
    df = pd.read_csv(manifest_path)
    if subset:
        df = df.head(subset)
        print(f"  Subset mode: first {subset} rows")

    n = len(df)

    # ── ECG: each row needs .hea + .dat ────────────────────────────────
    ecg_urls = []
    for stem in df["ecg_path"]:
        stem = stem.rstrip("/")
        ecg_urls.append(f"{MIMIC_ECG_BASE}{stem}.hea")
        ecg_urls.append(f"{MIMIC_ECG_BASE}{stem}.dat")

    Path(ecg_out).parent.mkdir(parents=True, exist_ok=True)
    with open(ecg_out, "w") as f:
        f.write("\n".join(ecg_urls) + "\n")
    print(f"ECG download list: {ecg_out}  ({len(ecg_urls):,} URLs, {n} subjects)")

    # ── CXR: one .jpg per row ───────────────────────────────────────────
    cxr_urls = []
    for jpg_path in df["cxr_path"]:
        jpg_path = jpg_path.lstrip("/")
        cxr_urls.append(f"{MIMIC_CXR_BASE}{jpg_path}")

    with open(cxr_out, "w") as f:
        f.write("\n".join(cxr_urls) + "\n")
    print(f"CXR download list: {cxr_out}  ({len(cxr_urls):,} URLs, {n} subjects)")

    print("\nDone. Review a few lines before submitting download job:")
    print("  head -4", ecg_out)
    print("  head -2", cxr_out)


def main():
    parser = argparse.ArgumentParser(description="Build PhysioNet download URL lists")
    parser.add_argument("--config", default="configs/paths.yaml")
    parser.add_argument("--subset", type=int, default=None,
                        help="Limit to first N subjects (smoke-test)")
    parser.add_argument("--ecg-out", default=None,
                        help="Override output path for ECG list")
    parser.add_argument("--cxr-out", default=None,
                        help="Override output path for CXR list")
    args = parser.parse_args()

    cfg = load_config(args.config)
    manifest_path = cfg["manifest_v1"]
    ecg_out = args.ecg_out or cfg["ecg_download_list"]
    cxr_out = args.cxr_out or cfg["cxr_download_list"]

    build_lists(manifest_path, ecg_out, cxr_out, subset=args.subset)


if __name__ == "__main__":
    main()
