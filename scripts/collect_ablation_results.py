"""
scripts/collect_ablation_results.py

Scans ablation checkpoint directories, extracts best ECG + Pulm AUC,
and writes outputs/ablation_results.csv for comparison.

Usage:
    cd /scratch/user/dshekar/more-clinical-distill
    python scripts/collect_ablation_results.py
    # or with a different base dir:
    python scripts/collect_ablation_results.py --base outputs/ablations
"""

import argparse
import csv
import glob
import os
import re
import sys
from pathlib import Path

import torch


ABLATION_NAMES = ["A0_baseline", "A0_efficientnet", "A1_uniform", "A2_lossgate",
                  "A3_ecgonly", "B1_lead2", "B2_v2"]

ABLATION_DESCRIPTIONS = {
    "A0_baseline":     "MobileNetV3-Small, static weights, ECG-only align, Lead I",
    "A0_efficientnet": "EfficientNet-B0, static weights, ECG-only align, Lead I",
    "A1_uniform":      "MobileNetV3-Small, uniform 1/n weights, all modalities, Lead I",
    "A2_lossgate":     "MobileNetV3-Small, ResidualLossGate, all modalities, Lead I",
    "A3_ecgonly":      "MobileNetV3-Small, ResidualLossGate, ECG-only align, Lead I",
    "B1_lead2":        "MobileNetV3-Small, ResidualLossGate, all modalities, Lead II",
    "B2_v2":           "MobileNetV3-Small, ResidualLossGate, all modalities, V2",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="outputs/ablations")
    p.add_argument("--out",  default="outputs/ablation_results.csv")
    return p.parse_args()


def best_checkpoint(run_dir: Path):
    """Find the best checkpoint .pth in a run directory."""
    ckpts = sorted(run_dir.glob("student_best_*.pth"))
    if not ckpts:
        return None
    # Return the one with highest ecgauc in filename
    def auc_from_name(p):
        m = re.search(r"ecgauc(\d+\.\d+)", p.name)
        return float(m.group(1)) if m else 0.0
    return max(ckpts, key=auc_from_name)


def extract_metrics(ckpt_path: Path) -> dict:
    """Load checkpoint and extract stored AUC metrics."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    return {
        "epoch":    ckpt.get("epoch", -1),
        "ecg_auc":  ckpt.get("ecg_auc", float("nan")),
        "pulm_auc": ckpt.get("pulm_auc", float("nan")),
    }


def main():
    args = parse_args()
    base  = Path(args.base)
    out   = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for name in ABLATION_NAMES:
        run_dir = base / name
        row = {
            "name":        name,
            "description": ABLATION_DESCRIPTIONS.get(name, ""),
            "status":      "missing",
            "ecg_auc":     "",
            "pulm_auc":    "",
            "epoch":       "",
            "ckpt":        "",
        }

        if not run_dir.exists():
            rows.append(row)
            continue

        ckpt = best_checkpoint(run_dir)
        if ckpt is None:
            row["status"] = "no_checkpoint"
            rows.append(row)
            continue

        try:
            metrics = extract_metrics(ckpt)
            row.update({
                "status":   "done",
                "ecg_auc":  f"{metrics['ecg_auc']:.4f}",
                "pulm_auc": f"{metrics['pulm_auc']:.4f}",
                "epoch":    metrics["epoch"],
                "ckpt":     ckpt.name,
            })
        except Exception as e:
            row["status"] = f"error: {e}"

        rows.append(row)

    # Write CSV
    fieldnames = ["name", "description", "status", "ecg_auc", "pulm_auc", "epoch", "ckpt"]
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Pretty-print
    print(f"\n{'Run':<16}  {'ECG AUC':>8}  {'Pulm AUC':>9}  {'Ep':>4}  Status")
    print("-" * 60)
    for row in rows:
        print(f"{row['name']:<16}  {row['ecg_auc']:>8}  {row['pulm_auc']:>9}  "
              f"{str(row['epoch']):>4}  {row['status']}")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
