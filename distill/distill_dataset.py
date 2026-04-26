"""
distill/distill_dataset.py

Dataset for the distillation phase with all 3 teacher modality embeddings.

Returns per item:
  ecg_tensor:       (1, 1000) float32  — single lead (default Lead I, index 0)
  teacher_ecg_emb:  (128,)   float32  — teacher ECG embedding
  teacher_cxr_emb:  (128,)   float32  — teacher CXR embedding
  teacher_text_emb: (128,)   float32  — teacher text embedding
  ecg_labels:       (10,)    float32  — ECG rhythm classes (multi-label)
  pulm_labels:      (4,)     float32  — CheXpert pulmonary labels

All teacher embeddings come from teacher_triplet.npz.
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

_ZERO_EMB = np.zeros(128, dtype=np.float32)


class DistillDataset(Dataset):
    """
    Args:
        data_path:        path to MoRE-format .npy (train/val/test split)
        triplet_cache:    path to teacher_triplet.npz (all 3 modalities)
        embed_cache:      legacy: path to teacher_ecg_embeddings.npy (N×128).
                          Ignored when triplet_cache is provided.
        study_id_cache:   legacy: path to teacher_ecg_study_ids.npy (N,).
                          Ignored when triplet_cache is provided.
        lead_idx:         which ECG lead to use (0=Lead I, 1=Lead II, 7=V2)
        signal_length:    expected signal length after preprocessing (1000)
        split:            which split key to load from triplet_cache
                          ('train', 'validate'/'val', or 'test')
    """

    def __init__(
        self,
        data_path:      str,
        triplet_cache:  str | None = None,
        embed_cache:    str | None = None,    # legacy fallback
        study_id_cache: str | None = None,   # legacy fallback
        signal_cache:   str | None = None,   # (N, 1000, 12) float32 cache of all leads
        signal_id_cache: str | None = None,  # parallel study-id array
        lead_idx:       int = 0,
        signal_length:  int = 1000,
        split:          str = "train",
    ):
        self.data         = np.load(data_path, allow_pickle=True)
        self.lead_idx     = lead_idx
        self.signal_length = signal_length

        # Normalise split key: .npy stores 'validate', npz stores 'val'
        split_key = "val" if split in ("validate", "val") else split

        # ── Pre-cached ECG signals (fast path) ────────────────────────
        self.signals: np.ndarray | None = None
        self.signal_lookup: dict[str, int] = {}
        if signal_cache is not None and os.path.exists(signal_cache):
            self.signals = np.load(signal_cache, mmap_mode="r")  # (N, 1000, 12) float32
            if signal_id_cache and os.path.exists(signal_id_cache):
                ids = np.load(signal_id_cache, allow_pickle=True)
                self.signal_lookup = {str(s): i for i, s in enumerate(ids)}
            print(f"[DistillDataset] Using ECG signal cache: {self.signals.shape} from {signal_cache}")

        # ── Load teacher embeddings from triplet_cache (preferred) ────
        self.ecg_embs:  np.ndarray | None = None
        self.cxr_embs:  np.ndarray | None = None
        self.text_embs: np.ndarray | None = None
        self.emb_lookup: dict[str, int]   = {}

        if triplet_cache is not None and os.path.exists(triplet_cache):
            npz = np.load(triplet_cache, allow_pickle=True)

            def _clean(arr: np.ndarray) -> np.ndarray:
                n_bad = int(np.isnan(arr).any(axis=1).sum())
                if n_bad:
                    print(f"[DistillDataset] WARNING: {n_bad} teacher embeddings contain NaN "
                          f"— replacing with zeros.")
                return np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

            # Build a GLOBAL lookup that spans all three npz partitions
            # (train+val+test). The npz was originally partitioned per the
            # legacy CXR-derived split, but the underlying embeddings cover
            # the full 49,076-record cohort. Supporting any external split
            # (e.g. patient-stratified 80/10/10) requires a global lookup.
            ecg_blocks, cxr_blocks, text_blocks, id_blocks = [], [], [], []
            for part in ("train", "val", "test"):
                ecg_blocks.append(_clean(npz[f"{part}_ecg"]))
                cxr_blocks.append(_clean(npz[f"{part}_cxr"]))
                text_blocks.append(_clean(npz[f"{part}_text"]))
                id_blocks.append(npz[f"{part}_ids"])
            self.ecg_embs  = np.concatenate(ecg_blocks,  axis=0)
            self.cxr_embs  = np.concatenate(cxr_blocks,  axis=0)
            self.text_embs = np.concatenate(text_blocks, axis=0)
            ids_all = np.concatenate(id_blocks, axis=0)
            self.emb_lookup = {str(sid): i for i, sid in enumerate(ids_all)}

        elif embed_cache is not None and os.path.exists(embed_cache):
            # Legacy: ECG-only embeddings (for backwards compatibility)
            raw = np.load(embed_cache, allow_pickle=True).astype(np.float32)
            n_nan = int(np.isnan(raw).any(axis=1).sum())
            if n_nan:
                print(f"[DistillDataset] WARNING: {n_nan} teacher embeddings contain NaN.")
            self.ecg_embs = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
            self.cxr_embs  = None
            self.text_embs = None

            if study_id_cache is None:
                study_id_cache = str(Path(embed_cache).parent / "teacher_ecg_study_ids.npy")
            if os.path.exists(study_id_cache):
                ids = np.load(study_id_cache, allow_pickle=True)
                self.emb_lookup = {str(sid): i for i, sid in enumerate(ids)}
            else:
                print(f"[DistillDataset] WARNING: study_id_cache not found at "
                      f"{study_id_cache}. Falling back to index alignment.")

        # ── Pre-filter: keep only rows with present ECG and embedding ──
        valid = []
        for i, item in enumerate(self.data):
            ecg_stem = item[1]
            study_id = Path(ecg_stem).name
            if self.signals is not None and self.signal_lookup:
                # Fast path: only require presence in the signal cache; .hea stat skipped
                if study_id not in self.signal_lookup:
                    continue
            else:
                if not os.path.exists(ecg_stem + ".hea"):
                    continue
            if self.emb_lookup and study_id not in self.emb_lookup:
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
        item        = self.data[self.valid_indices[idx]]
        ecg_stem    = item[1]
        ecg_labels  = item[4].astype(np.float32)   # (10,) ECG rhythm labels
        pulm_labels = item[5].astype(np.float32)   # (4,)  CheXpert pulmonary
        study_id    = Path(ecg_stem).name

        # ── Load ECG: cache fast path or wfdb fallback ────────────────
        if self.signals is not None and study_id in self.signal_lookup:
            # mmap read of the (1000, 12) slice for this study, then pick the lead
            sig_idx = self.signal_lookup[study_id]
            lead    = np.asarray(self.signals[sig_idx, :, self.lead_idx], dtype=np.float32)
        else:
            import wfdb
            from scipy.signal import resample as scipy_resample
            sig, fields = wfdb.rdsamp(ecg_stem)
            sig = np.nan_to_num(sig.astype(np.float32), nan=0.0)
            if fields["fs"] != 100:
                n_out = int(sig.shape[0] * 100 / fields["fs"])
                sig = scipy_resample(sig, n_out, axis=0)
            lead = sig[:, self.lead_idx]

        if len(lead) >= self.signal_length:
            lead = lead[:self.signal_length]
        else:
            lead = np.pad(lead, (0, self.signal_length - len(lead)))

        m = np.max(np.abs(lead))
        if m > 1e-6:
            lead = lead / m

        ecg_tensor = torch.FloatTensor(lead).unsqueeze(0)   # (1, 1000)

        # ── Teacher embeddings ────────────────────────────────────────
        if self.emb_lookup:
            study_id = Path(ecg_stem).name
            emb_idx  = self.emb_lookup[study_id]
        else:
            emb_idx  = self.valid_indices[idx]

        def _get_emb(arr: np.ndarray | None) -> torch.Tensor:
            if arr is not None:
                return torch.FloatTensor(arr[emb_idx])
            return torch.zeros(128)

        teacher_ecg_emb  = _get_emb(self.ecg_embs)
        teacher_cxr_emb  = _get_emb(self.cxr_embs)
        teacher_text_emb = _get_emb(self.text_embs)

        return (ecg_tensor,
                teacher_ecg_emb,
                teacher_cxr_emb,
                teacher_text_emb,
                torch.FloatTensor(ecg_labels),
                torch.FloatTensor(pulm_labels))
