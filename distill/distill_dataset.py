"""
distill/distill_dataset.py

Dataset for the distillation phase.

Returns per item:
  ecg_tensor:        (1, 1000) float32  — single lead (default Lead I, index 0)
  teacher_embedding: (128,)   float32  — pre-cached teacher ECG embedding
  ecg_labels:        (10,)    float32  — ECG rhythm classes (multi-label)
  pulm_labels:       (4,)     float32  — CheXpert pulmonary labels

Teacher embeddings are pre-computed by distill/cache_teacher_embeddings.py.
Alignment is done via study-ID lookup (not index-to-index), so the .npy
embedding array and the manifest rows can be in any order.

Rows where the ECG .hea file is absent or no matching embedding exists are
silently dropped at __init__ time, so __len__ reflects the usable subset.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.append("./utils")


class DistillDataset(Dataset):
    """
    Args:
        data_path:       path to MoRE-format .npy (train/val/test split)
        embed_cache:     path to teacher_ecg_embeddings.npy (N×128), or None
        study_id_cache:  path to teacher_ecg_study_ids.npy (N,), or None
                         Required when embed_cache is provided for alignment.
        lead_idx:        which ECG lead to use (0=Lead I, 1=Lead II, 7=V2)
        signal_length:   expected signal length after preprocessing (1000)
    """

    def __init__(
        self,
        data_path: str,
        embed_cache: str | None = None,
        study_id_cache: str | None = None,
        lead_idx: int = 0,
        signal_length: int = 1000,
    ):
        self.data = np.load(data_path, allow_pickle=True)
        self.lead_idx = lead_idx
        self.signal_length = signal_length

        # ── Load embedding cache and build study-ID lookup ────────────
        self.embeddings: np.ndarray | None = None
        self.emb_lookup: dict[str, int] = {}

        if embed_cache is not None:
            raw = np.load(embed_cache, allow_pickle=True).astype(np.float32)
            # Replace NaN/Inf in teacher embeddings (some ViT outputs are NaN for bad records)
            n_nan_rows = int(np.isnan(raw).any(axis=1).sum())
            if n_nan_rows:
                print(f"[DistillDataset] WARNING: {n_nan_rows} teacher embeddings contain NaN "
                      f"— replacing with zeros.")
            self.embeddings = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
            # Derive study_id_cache path from embed_cache path if not given
            if study_id_cache is None:
                study_id_cache = str(Path(embed_cache).parent / "teacher_ecg_study_ids.npy")
            if os.path.exists(study_id_cache):
                study_ids = np.load(study_id_cache, allow_pickle=True)
                self.emb_lookup = {str(sid): i for i, sid in enumerate(study_ids)}
            else:
                # Fallback: index-to-index (only safe if arrays are aligned)
                print(f"[DistillDataset] WARNING: study_id_cache not found at "
                      f"{study_id_cache}. Falling back to index alignment.")

        # ── Pre-filter: keep only rows with present ECG and embedding ──
        valid = []
        for i, item in enumerate(self.data):
            ecg_stem = item[1]
            if not os.path.exists(ecg_stem + ".hea"):
                continue
            if self.emb_lookup:
                study_id = Path(ecg_stem).name
                if study_id not in self.emb_lookup:
                    continue
            valid.append(i)
        self.valid_indices = valid

        n_total = len(self.data)
        n_valid = len(self.valid_indices)
        print(f"[DistillDataset] {data_path}: {n_valid}/{n_total} rows usable "
              f"({100*n_valid/max(n_total,1):.1f}%)")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        item      = self.data[self.valid_indices[idx]]
        ecg_stem  = item[1]
        ecg_labels  = item[4].astype(np.float32)   # (10,) ECG rhythm labels
        pulm_labels = item[5].astype(np.float32)   # (4,)  CheXpert pulmonary

        # ── Load ECG via wfdb ─────────────────────────────────────────
        import wfdb
        from scipy.signal import resample as scipy_resample

        sig, fields = wfdb.rdsamp(ecg_stem)          # (T, 12), fields dict
        sig = np.nan_to_num(sig.astype(np.float32), nan=0.0)

        # Resample to 100 Hz using scipy.signal.resample (matches teacher cache)
        orig_fs = fields["fs"]
        if orig_fs != 100:
            n_out = int(sig.shape[0] * 100 / orig_fs)
            sig = scipy_resample(sig, n_out, axis=0)  # (1000, 12)

        # Select single lead
        lead = sig[:, self.lead_idx]                 # (T,)

        # Trim / pad to signal_length
        if len(lead) >= self.signal_length:
            lead = lead[:self.signal_length]
        else:
            lead = np.pad(lead, (0, self.signal_length - len(lead)))

        # Per-lead abs-max normalisation (matches teacher's cache_teacher_embeddings.py)
        m = np.max(np.abs(lead))
        if m > 1e-6:
            lead = lead / m

        ecg_tensor = torch.FloatTensor(lead).unsqueeze(0)   # (1, 1000)

        # ── Teacher embedding ─────────────────────────────────────────
        if self.embeddings is not None:
            if self.emb_lookup:
                study_id  = Path(ecg_stem).name
                emb_idx   = self.emb_lookup[study_id]
            else:
                emb_idx   = self.valid_indices[idx]
            teacher_emb = torch.FloatTensor(self.embeddings[emb_idx])
        else:
            teacher_emb = torch.zeros(128)

        return (ecg_tensor,
                teacher_emb,
                torch.FloatTensor(ecg_labels),
                torch.FloatTensor(pulm_labels))
