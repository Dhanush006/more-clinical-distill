#!/usr/bin/env python3
"""
scripts/flatten_data.py

Flattens the deep PhysioNet wget mirror directory tree into two flat directories:
  data/ecg_flat/{study_id}.hea
  data/ecg_flat/{study_id}.dat
  data/cxr_flat/{dicom_id}.jpg

This eliminates ~200k intermediate directory inodes created by wget -x,
freeing significant inode quota without changing disk block usage.

After flattening, the original deep tree is removed (empty dirs deleted).
`os.rename()` is used throughout — O(1) on the same filesystem, no data copied.

Usage:
    python scripts/flatten_data.py [--dry-run] [--config configs/paths.yaml]

    --dry-run: print what would happen without moving anything
"""

import argparse
import os
import sys
from pathlib import Path

import yaml


def flatten_ecg(ecg_root: Path, ecg_flat: Path, dry_run: bool) -> tuple[int, int]:
    """Move all .hea/.dat files from deep tree into ecg_flat/. Returns (moved, skipped)."""
    moved = 0
    skipped = 0
    conflicts = 0

    hea_files = list(ecg_root.rglob("*.hea"))
    dat_files = list(ecg_root.rglob("*.dat"))
    all_files = hea_files + dat_files
    total = len(all_files)
    print(f"\n[ECG] Found {len(hea_files):,} .hea + {len(dat_files):,} .dat = {total:,} files")

    if not dry_run:
        ecg_flat.mkdir(parents=True, exist_ok=True)

    for i, src in enumerate(all_files):
        dst = ecg_flat / src.name
        if dst.exists():
            skipped += 1
            continue
        if dry_run:
            print(f"  [DRY] {src} → {dst}")
            moved += 1
        else:
            try:
                os.rename(src, dst)
                moved += 1
            except OSError as e:
                print(f"  ERROR renaming {src}: {e}")
                conflicts += 1

        if (i + 1) % 10000 == 0:
            print(f"  ... {i+1:,}/{total:,} processed")

    print(f"[ECG] Moved: {moved:,}  Skipped (already flat): {skipped:,}  Errors: {conflicts}")
    return moved, skipped


def flatten_cxr(cxr_root: Path, cxr_flat: Path, dry_run: bool) -> tuple[int, int]:
    """Move all .jpg files from deep tree into cxr_flat/. Returns (moved, skipped)."""
    moved = 0
    skipped = 0
    conflicts = 0

    jpg_files = list(cxr_root.rglob("*.jpg"))
    total = len(jpg_files)
    print(f"\n[CXR] Found {total:,} .jpg files")

    if not dry_run:
        cxr_flat.mkdir(parents=True, exist_ok=True)

    for i, src in enumerate(jpg_files):
        dst = cxr_flat / src.name
        if dst.exists():
            skipped += 1
            continue
        if dry_run:
            print(f"  [DRY] {src} → {dst}")
            moved += 1
        else:
            try:
                os.rename(src, dst)
                moved += 1
            except OSError as e:
                print(f"  ERROR renaming {src}: {e}")
                conflicts += 1

        if (i + 1) % 10000 == 0:
            print(f"  ... {i+1:,}/{total:,} processed")

    print(f"[CXR] Moved: {moved:,}  Skipped (already flat): {skipped:,}  Errors: {conflicts}")
    return moved, skipped


def remove_empty_dirs(root: Path, dry_run: bool) -> int:
    """Remove empty directories bottom-up under root. Returns count removed."""
    removed = 0
    # Walk bottom-up so children are processed before parents
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        p = Path(dirpath)
        if p == root:
            continue
        try:
            children = list(p.iterdir())
            if not children:
                if dry_run:
                    print(f"  [DRY] rmdir {p}")
                else:
                    p.rmdir()
                removed += 1
        except OSError:
            pass
    return removed


def main():
    parser = argparse.ArgumentParser(description="Flatten ECG + CXR data directory trees")
    parser.add_argument("--config", default="configs/paths.yaml")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print actions without moving files")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_root = Path(cfg["data_root"])
    ecg_flat  = data_root / "ecg_flat"
    cxr_flat  = data_root / "cxr_flat"

    # Old deep roots (wget mirror paths)
    ecg_deep = data_root / "mimic-iv-ecg"
    cxr_deep = data_root / "mimic-cxr-jpg" / "physionet.org"

    print("=" * 60)
    print("Flatten Data Migration")
    print("=" * 60)
    print(f"ECG source: {ecg_deep}")
    print(f"ECG dest:   {ecg_flat}")
    print(f"CXR source: {cxr_deep}")
    print(f"CXR dest:   {cxr_flat}")
    if args.dry_run:
        print("\n*** DRY RUN — no files will be moved ***")

    # ── ECG ──────────────────────────────────────────────────────────
    if not ecg_deep.exists():
        print(f"\n[ECG] Source not found: {ecg_deep} — skipping")
    else:
        flatten_ecg(ecg_deep, ecg_flat, args.dry_run)
        if not args.dry_run:
            print(f"\n[ECG] Removing empty dirs under {ecg_deep} ...")
            n = remove_empty_dirs(ecg_deep, dry_run=False)
            print(f"[ECG] Removed {n:,} empty directories")

    # ── CXR ──────────────────────────────────────────────────────────
    if not cxr_deep.exists():
        print(f"\n[CXR] Source not found: {cxr_deep} — skipping")
    else:
        flatten_cxr(cxr_deep, cxr_flat, args.dry_run)
        if not args.dry_run:
            print(f"\n[CXR] Removing empty dirs under {cxr_deep} ...")
            n = remove_empty_dirs(cxr_deep, dry_run=False)
            print(f"[CXR] Removed {n:,} empty directories")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Migration complete.")
    print("Next steps:")
    print("  1. Re-run: python scripts/convert_manifest_to_more_format.py")
    print("     (regenerates .npy files with flat paths)")
    print("  2. Re-run: sbatch slurm/cache_embeddings.slurm")
    print("     (re-caches teacher embeddings from flat ECG dir)")
    print("=" * 60)


if __name__ == "__main__":
    main()
