"""
distill/distill_dataset.py

Dataset for the distillation phase.

Returns per item:
  ecg_tensor:        (1, 1000) float32  — single lead (default Lead I, index 0)
  teacher_embedding: (128,)   float32  — pre-cached teacher ECG embedding
  labels:            (4,)     float32  — CheXpert subset labels

Teacher embeddings are pre-computed by distill/cache_teacher_embeddings.py
and stored as a .npy array aligned row-for-row with the .npy data file.

Alternatively, if no cache is provided, embeddings can be computed on-the-fly
(slower, requires teacher model loaded in the same process).
"""

import numpy as np
import torch
from torch.utils.data import Dataset

import sys
sys.path.append("./utils")


class DistillDataset(Dataset):
    """
    Args:
        data_path:       path to MoRE-format .npy (train split)
        embed_cache:     path to teacher_embeddings.npy (shape N×128), or None
        lead_idx:        which ECG lead to use (0=Lead I, 1=Lead II, 7=V2)
        signal_length:   expected signal length after preprocessing (1000)
    """

    def __init__(
        self,
        data_path: str,
        embed_cache: str | None = None,
        lead_idx: int = 0,
        signal_length: int = 1000,
    ):
        self.data = np.load(data_path, allow_pickle=True)
        self.lead_idx = lead_idx
        self.signal_length = signal_length

        self.embeddings = None
        if embed_cache is not None:
            self.embeddings = np.load(embed_cache, allow_pickle=True).astype(np.float32)
            assert len(self.embeddings) == len(self.data), (
                f"Embedding cache length {len(self.embeddings)} != "
                f"data length {len(self.data)}"
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        ecg_stem = item[1]
        labels   = item[4].astype(np.float32)

        # ── Load ECG ──────────────────────────────────────────────────
        import wfdb
        from scipy.signal import resample_poly

        sig, _ = wfdb.rdsamp(ecg_stem)          # (T, 12)
        sig = np.nan_to_num(sig, nan=0.0)
        sig = resample_poly(sig, up=1, down=5)   # (200, 12) if 500→100 Hz

        # Select single lead and truncate/pad to signal_length
        lead = sig[:, self.lead_idx]             # (T,)
        if len(lead) >= self.signal_length:
            lead = lead[:self.signal_length]
        else:
            lead = np.pad(lead, (0, self.signal_length - len(lead)))

        # Normalise to [-1, 1]
        lo, hi = lead.min(), lead.max()
        if hi - lo > 1e-8:
            lead = 2.0 * (lead - lo) / (hi - lo) - 1.0

        ecg_tensor = torch.FloatTensor(lead).unsqueeze(0)  # (1, 1000)

        # ── Teacher embedding ─────────────────────────────────────────
        if self.embeddings is not None:
            teacher_emb = torch.FloatTensor(self.embeddings[idx])
        else:
            teacher_emb = torch.zeros(128)   # placeholder when no cache

        return ecg_tensor, teacher_emb, torch.FloatTensor(labels)
