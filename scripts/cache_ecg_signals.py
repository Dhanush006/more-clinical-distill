"""
scripts/cache_ecg_signals.py

Pre-cache all 12-lead ECG signals to a single memory-mappable .npy file
+ a parallel study_id array. Eliminates per-batch wfdb.rdsamp + scipy.resample
overhead during training (the dominant bottleneck — 30 min/epoch with on-the-fly
loading vs. < 2 min/epoch with cached signals).

Output:
    data/processed/ecg_signals_100hz.npy   shape (N, 1000, 12) float32 — all 12 leads at 100 Hz
    data/processed/ecg_signal_ids.npy      shape (N,)         str       — aligned study IDs

Usage:
    cd /scratch/user/dshekar/more-clinical-distill
    python scripts/cache_ecg_signals.py [--workers 16] [--ecg-dir data/ecg_flat]
"""

import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
import wfdb
from scipy.signal import resample as scipy_resample
from tqdm import tqdm


def _process_one(stem: str) -> tuple[str, np.ndarray] | None:
    """Load one ECG, resample to 100 Hz, return (study_id, (1000, 12) float32)."""
    try:
        sig, fields = wfdb.rdsamp(stem)
        sig = np.nan_to_num(sig.astype(np.float32), nan=0.0)
        if fields["fs"] != 100:
            n_out = int(sig.shape[0] * 100 / fields["fs"])
            sig = scipy_resample(sig, n_out, axis=0).astype(np.float32)
        # Trim/pad to exactly 1000 samples
        if sig.shape[0] >= 1000:
            sig = sig[:1000]
        else:
            pad = np.zeros((1000 - sig.shape[0], sig.shape[1]), dtype=np.float32)
            sig = np.vstack([sig, pad])
        return Path(stem).name, sig
    except Exception as e:
        sys.stderr.write(f"[FAIL] {stem}: {e}\n")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ecg-dir", default="data/ecg_flat")
    ap.add_argument("--out-signals", default="data/processed/ecg_signals_100hz.npy")
    ap.add_argument("--out-ids",     default="data/processed/ecg_signal_ids.npy")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    ecg_dir = Path(args.ecg_dir)
    hea_files = sorted(ecg_dir.glob("*.hea"))
    stems = [str(p.with_suffix("")) for p in hea_files]
    print(f"Found {len(stems)} ECG records under {ecg_dir}")

    if not stems:
        sys.exit("No ECG records — check --ecg-dir")

    Path(args.out_signals).parent.mkdir(parents=True, exist_ok=True)

    n = len(stems)
    signals = np.zeros((n, 1000, 12), dtype=np.float32)
    ids     = np.empty(n, dtype=object)
    n_ok = 0

    if args.workers > 1:
        with mp.Pool(args.workers) as pool:
            results = pool.imap_unordered(_process_one, stems, chunksize=64)
            for r in tqdm(results, total=n, desc="Caching ECGs"):
                if r is None:
                    continue
                sid, sig = r
                signals[n_ok] = sig
                ids[n_ok]     = sid
                n_ok += 1
    else:
        for stem in tqdm(stems, desc="Caching ECGs"):
            r = _process_one(stem)
            if r is None:
                continue
            sid, sig = r
            signals[n_ok] = sig
            ids[n_ok]     = sid
            n_ok += 1

    # Trim to actual successes (in case of failures)
    signals = signals[:n_ok]
    ids     = ids[:n_ok]

    np.save(args.out_signals, signals)
    np.save(args.out_ids,     ids)

    sz_gb = os.path.getsize(args.out_signals) / 1e9
    print(f"Saved {n_ok}/{n} ECGs → {args.out_signals} ({sz_gb:.2f} GB)")
    print(f"Saved IDs       → {args.out_ids}")


if __name__ == "__main__":
    main()
