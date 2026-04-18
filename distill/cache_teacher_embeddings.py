"""
distill/cache_teacher_embeddings.py

Cache MoRE teacher ECG embeddings to disk for student KD training.

Pipeline:
  raw .dat/.hea files → wfdb → 12-lead ECG → teacher.ecg_model → 768-dim
                       → teacher.projector_ecg_text → 128-dim embedding

CXR images and the RoBERTa text model are NOT needed.

Output:
  data/processed/teacher_ecg_embeddings.npy  — float32 (N, 128)
  data/processed/teacher_ecg_study_ids.npy   — str array (N,)  for alignment

Usage:
    python distill/cache_teacher_embeddings.py --config configs/distill_config.yaml
    python distill/cache_teacher_embeddings.py --config configs/distill_config.yaml --subset 500
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.build_model import ViTModelEcg, ProjectionHead


# ── ECG helpers ───────────────────────────────────────────────────────────────

def load_ecg_wfdb(hea_path: str, target_fs: int = 100, n_samples: int = 1000):
    """
    Load a 12-lead ECG record with wfdb, resample to target_fs,
    and return a (12, n_samples) float32 array.
    Returns None if record cannot be loaded.
    """
    import wfdb

    record_path = str(hea_path).removesuffix(".hea")
    try:
        rec = wfdb.rdrecord(record_path)
    except Exception:
        return None

    signal = rec.p_signal  # (T, 12)
    if signal is None or signal.shape[1] < 12:
        return None

    signal = signal.T.astype(np.float32)  # (12, T)

    # Resample if needed
    orig_fs = rec.fs
    if orig_fs != target_fs:
        from scipy.signal import resample
        n_out = int(signal.shape[1] * target_fs / orig_fs)
        signal = resample(signal, n_out, axis=1)

    # Trim or pad to n_samples
    T = signal.shape[1]
    if T >= n_samples:
        signal = signal[:, :n_samples]
    else:
        pad = np.zeros((12, n_samples - T), dtype=np.float32)
        signal = np.concatenate([signal, pad], axis=1)

    # Normalize per lead to [-1, 1]
    for i in range(12):
        m = np.max(np.abs(signal[i]))
        if m > 1e-6:
            signal[i] /= m

    return signal  # (12, n_samples)


# ── ECG-only teacher ──────────────────────────────────────────────────────────

class TeacherECGEncoder(torch.nn.Module):
    """Minimal teacher: just the ECG encoder + ECG-text projector from MoRE."""

    def __init__(self):
        super().__init__()
        self.ecg_model = ViTModelEcg(projector=False)      # 768-dim output
        self.projector_ecg_text = ProjectionHead(768, 128, 768)  # 128-dim output

    @torch.no_grad()
    def forward(self, ecg: torch.Tensor) -> torch.Tensor:
        """ecg: (B, 12, T) → (B, 128)"""
        feat = self.ecg_model(ecg)
        return self.projector_ecg_text(feat)


def load_teacher_ecg(checkpoint_path: str, device: str) -> TeacherECGEncoder:
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and not any(isinstance(v, torch.Tensor) for v in ckpt.values()):
        state_dict = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    else:
        state_dict = ckpt

    cleaned = {k.removeprefix("module."): v for k, v in state_dict.items()}

    # Extract only ECG-relevant keys, stripping the component prefix
    ecg_sd  = {k.removeprefix("ecg_model."): v
               for k, v in cleaned.items() if k.startswith("ecg_model.")}
    proj_sd = {k.removeprefix("projector_ecg_text."): v
               for k, v in cleaned.items() if k.startswith("projector_ecg_text.")}

    teacher = TeacherECGEncoder()
    missing_e, _ = teacher.ecg_model.load_state_dict(ecg_sd, strict=False)
    missing_p, _ = teacher.projector_ecg_text.load_state_dict(proj_sd, strict=False)
    if missing_e:
        print(f"  ecg_model  missing: {missing_e[:5]}")
    if missing_p:
        print(f"  projector  missing: {missing_p[:5]}")

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/distill_config.yaml")
    parser.add_argument("--subset", type=int, default=0,
                        help="Process only first N studies (0 = all)")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    ckpt_path = cfg["teacher"]["checkpoint"]
    print(f"Loading teacher ECG encoder from {ckpt_path} ...")
    teacher = load_teacher_ecg(ckpt_path, device).to(device)
    print("Teacher loaded and frozen.")

    # ── Locate ECG .hea files ─────────────────────────────────────────
    # Load paths config
    import yaml as _yaml
    with open("configs/paths.yaml") as fp:
        paths_cfg = _yaml.safe_load(fp)

    # Prefer flat dir (post flatten_data.py) — uses glob instead of rglob
    ecg_flat = Path(paths_cfg.get("ecg_flat_root", "data/ecg_flat"))
    ecg_root = Path(paths_cfg.get("mimic_ecg_root", "data/mimic-iv-ecg"))

    if ecg_flat.exists() and any(ecg_flat.glob("*.hea")):
        hea_files = sorted(ecg_flat.glob("*.hea"))
        print(f"Using flat ECG dir: {ecg_flat}")
    else:
        hea_files = sorted(ecg_root.rglob("*.hea"))
        print(f"Using deep ECG dir: {ecg_root}")

    if not hea_files:
        print(f"No .hea files found under {ecg_root}. Run ECG download first.")
        sys.exit(1)

    print(f"Found {len(hea_files):,} .hea files")

    if args.subset > 0:
        hea_files = hea_files[: args.subset]
        print(f"Using subset of {len(hea_files)} studies")

    # ── Batch and embed ───────────────────────────────────────────────
    signal_length = cfg["student"].get("signal_length", 1000)
    batch_size = args.batch_size

    all_embeddings: list[np.ndarray] = []
    all_study_ids: list[str] = []
    failed = 0

    buffer_ecg: list[np.ndarray] = []
    buffer_ids: list[str] = []

    def flush_buffer():
        nonlocal buffer_ecg, buffer_ids
        if not buffer_ecg:
            return
        batch = torch.tensor(np.stack(buffer_ecg), dtype=torch.float32).to(device)
        with torch.no_grad():
            emb = teacher(batch).cpu().numpy()
        all_embeddings.append(emb)
        all_study_ids.extend(buffer_ids)
        buffer_ecg, buffer_ids = [], []

    for hea in tqdm(hea_files, desc="Caching teacher embeddings"):
        signal = load_ecg_wfdb(str(hea), target_fs=100, n_samples=signal_length)
        if signal is None:
            failed += 1
            continue
        study_id = hea.stem
        buffer_ecg.append(signal)
        buffer_ids.append(study_id)
        if len(buffer_ecg) >= batch_size:
            flush_buffer()

    flush_buffer()

    print(f"\nProcessed {len(all_study_ids):,} studies  ({failed} failed/skipped)")

    all_embeddings_arr = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    all_study_ids_arr  = np.array(all_study_ids)

    out_dir = Path(cfg["teacher"]["embedding_cache"]).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    emb_path = cfg["teacher"]["embedding_cache"]
    ids_path = str(Path(emb_path).parent / "teacher_ecg_study_ids.npy")

    np.save(emb_path, all_embeddings_arr)
    np.save(ids_path, all_study_ids_arr)

    print(f"Saved embeddings → {emb_path}  {all_embeddings_arr.shape}  "
          f"({all_embeddings_arr.nbytes / 1e6:.1f} MB)")
    print(f"Saved study IDs  → {ids_path}")


if __name__ == "__main__":
    main()
