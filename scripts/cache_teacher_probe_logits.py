"""
scripts/cache_teacher_probe_logits.py

Fit a multi-label logistic regression probe on teacher ECG embeddings + ECG labels
from the training split. Saves probe weights for use as frozen KL teacher targets.

Output: data/processed/teacher_probe_weights.npz
    W: (10, 128) float32 — probe weight matrix (one row per ECG class)
    b: (10,)     float32 — probe bias vector

At training time, teacher logits are computed on-the-fly via:
    teacher_ecg_logits = t_ecg @ W.T + b   (B, 10)

Usage:
    cd /scratch/user/dshekar/more-clinical-distill
    python scripts/cache_teacher_probe_logits.py
"""

from pathlib import Path

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression

REPO = Path(__file__).resolve().parent.parent

with open(REPO / "configs" / "distill_config.yaml") as f:
    cfg = yaml.safe_load(f)

# Load training teacher ECG embeddings from triplet cache
triplet_path = REPO / cfg["teacher"]["triplet_cache"]
print(f"Loading triplet cache: {triplet_path}")
npz = np.load(triplet_path, allow_pickle=True)

# Concatenate all partitions into a global lookup (same logic as DistillDataset)
ecg_blocks, id_blocks = [], []
for part in ("train", "val", "test"):
    ecg_blocks.append(npz[f"{part}_ecg"].astype(np.float32))
    id_blocks.append(npz[f"{part}_ids"])
all_embs = np.concatenate(ecg_blocks, axis=0)
all_ids  = np.concatenate(id_blocks,  axis=0)
emb_lookup = {str(sid): i for i, sid in enumerate(all_ids)}
print(f"Total embeddings available: {len(all_embs)}")

# Load training labels from stratified train split
train_npy = REPO / cfg["data"]["train_npy"]
print(f"Loading training labels: {train_npy}")
train_data = np.load(train_npy, allow_pickle=True)

X, Y = [], []
for item in train_data:
    stem     = item[1]
    study_id = Path(stem).name
    if study_id not in emb_lookup:
        continue
    X.append(all_embs[emb_lookup[study_id]])
    Y.append(item[4].astype(np.float32))  # (10,) ECG labels

X = np.array(X, dtype=np.float32)  # (N, 128)
Y = np.array(Y, dtype=np.float32)  # (N, 10)

# Only fit on labelled rows (all-zero rows contribute nothing to a probe)
has_label = Y.sum(axis=1) > 0
print(f"Labelled training samples: {has_label.sum()} / {len(X)}")
X_fit = X[has_label]
Y_fit = Y[has_label]

# Fit one binary LR per class (multi-label → independent per-class classifiers)
n_classes = Y_fit.shape[1]
class_names = cfg["student"].get("ecg_class_names", [str(c) for c in range(n_classes)])

W = np.zeros((n_classes, 128), dtype=np.float32)
b = np.zeros(n_classes,        dtype=np.float32)

for c in range(n_classes):
    y_c = Y_fit[:, c]
    n_pos = int(y_c.sum())
    if n_pos < 2:
        print(f"  [{class_names[c]}] too few positives ({n_pos}), probe weight stays 0")
        continue
    lr = LogisticRegression(class_weight="balanced", max_iter=500, C=1.0, solver="lbfgs")
    lr.fit(X_fit, y_c)
    W[c] = lr.coef_[0]
    b[c] = lr.intercept_[0]
    print(f"  [{class_names[c]}] n_pos={n_pos} — probe fit done")

out = REPO / "data" / "processed" / "teacher_probe_weights.npz"
np.savez(out, W=W, b=b)
print(f"\nSaved probe weights to {out}  (W={W.shape}, b={b.shape})")
print("Run ablation A4_correctgate_strat to use these in training.")
