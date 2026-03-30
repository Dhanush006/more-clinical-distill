"""
distill/evaluate_student.py

Benchmark: Student model vs Teacher embedding baseline on the test set.

Computes:
  1. Student ECG head AUC-ROC (per-class + macro) — primary task
  2. Student Pulm head AUC-ROC (per-class + macro) — secondary task
  3. Teacher linear-probe AUC-ROC for both tasks
     (train logistic regression on teacher ECG embeddings → baseline)
  4. Mean cosine similarity: student embedding vs teacher embedding
  5. Summary table comparing student vs teacher-probe on every metric

Usage:
    python distill/evaluate_student.py \\
        --config configs/distill_config.yaml \\
        --checkpoint outputs/distill/student_best_ep42_ecgauc0.7830.pth
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

sys.path.append("./distill")
sys.path.append("./utils")

from student_model import build_student
from distill_dataset import DistillDataset


# ── Helpers ───────────────────────────────────────────────────────────

def compute_auc_per_class(probs: np.ndarray, labels: np.ndarray,
                           class_names: list[str]) -> dict:
    """Return {class_name: auc, 'macro': macro_auc}."""
    from sklearn.metrics import roc_auc_score
    out = {}
    for i, name in enumerate(class_names):
        y = labels[:, i]
        # Skip classes with only one unique label value
        if len(np.unique(y)) < 2:
            out[name] = float("nan")
            continue
        try:
            out[name] = roc_auc_score(y, probs[:, i])
        except Exception:
            out[name] = float("nan")
    valid = [v for v in out.values() if not np.isnan(v)]
    out["macro"] = float(np.mean(valid)) if valid else float("nan")
    return out


def print_auc_table(title: str, student_auc: dict, probe_auc: dict,
                    class_names: list[str]) -> None:
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")
    print(f"  {'Class':<25}  {'Student':>8}  {'Teacher probe':>13}  {'Δ':>6}")
    print(f"  {'-'*25}  {'-'*8}  {'-'*13}  {'-'*6}")
    for name in class_names:
        s = student_auc.get(name, float("nan"))
        t = probe_auc.get(name, float("nan"))
        delta = s - t if not (np.isnan(s) or np.isnan(t)) else float("nan")
        s_str = f"{s:.4f}" if not np.isnan(s) else "  n/a  "
        t_str = f"{t:.4f}" if not np.isnan(t) else "  n/a  "
        d_str = f"{delta:+.4f}" if not np.isnan(delta) else "  n/a  "
        print(f"  {name:<25}  {s_str:>8}  {t_str:>13}  {d_str:>6}")
    s_m = student_auc.get("macro", float("nan"))
    t_m = probe_auc.get("macro", float("nan"))
    d_m = s_m - t_m if not (np.isnan(s_m) or np.isnan(t_m)) else float("nan")
    print(f"  {'─'*25}  {'─'*8}  {'─'*13}  {'─'*6}")
    s_str = f"{s_m:.4f}" if not np.isnan(s_m) else "  n/a  "
    t_str = f"{t_m:.4f}" if not np.isnan(t_m) else "  n/a  "
    d_str = f"{d_m:+.4f}" if not np.isnan(d_m) else "  n/a  "
    print(f"  {'MACRO':<25}  {s_str:>8}  {t_str:>13}  {d_str:>6}")
    print(f"{'─'*60}")


# ── Main ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate student model")
    p.add_argument("--config",     default="configs/distill_config.yaml")
    p.add_argument("--checkpoint", required=True,
                   help="Path to student checkpoint .pth")
    p.add_argument("--split",      default="test",
                   choices=["test", "val"],
                   help="Which split to evaluate on")
    p.add_argument("--probe_train_split", default="train",
                   help="Split used to fit linear probe on teacher embeddings")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    ecg_class_names  = cfg["student"].get("ecg_class_names", [
        "Normal", "Sinus brady", "Sinus tachy", "AFib",
        "LBBB", "RBBB", "ST elev MI", "ST ischemia", "AV block", "LVH"
    ])
    pulm_class_names = cfg["student"].get("pulm_class_names", [
        "Atelectasis", "Cardiomegaly", "Edema", "Pleural Effusion"
    ])

    # ── Load student ─────────────────────────────────────────────────
    student = build_student(cfg["student"]).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        student.load_state_dict(ckpt["state_dict"])
        print(f"Checkpoint metadata: epoch={ckpt.get('epoch')}, "
              f"ecg_auc={ckpt.get('ecg_auc', 'n/a'):.4f}, "
              f"pulm_auc={ckpt.get('pulm_auc', 'n/a'):.4f}")
    else:
        student.load_state_dict(ckpt)
    student.eval()
    n_params = sum(p.numel() for p in student.parameters())
    print(f"Student parameters: {n_params:,}")

    # ── Eval dataset ─────────────────────────────────────────────────
    split_key = "test_npy" if args.split == "test" else "val_npy"
    eval_ds = DistillDataset(
        data_path      = cfg["data"][split_key],
        embed_cache    = cfg["teacher"].get("embedding_cache"),
        study_id_cache = cfg["teacher"].get("study_id_cache"),
        lead_idx       = cfg["student"].get("input_lead", 0),
        signal_length  = cfg["student"].get("signal_length", 1000),
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size  = cfg["training"]["batch_size"],
        num_workers = cfg["training"]["num_workers"],
        shuffle     = False,
        pin_memory  = True,
    )
    print(f"\nEvaluating on {args.split} set: {len(eval_ds)} samples")

    # ── Run student inference ─────────────────────────────────────────
    all_ecg_logits, all_ecg_labels   = [], []
    all_pulm_logits, all_pulm_labels = [], []
    all_student_emb, all_teacher_emb = [], []

    with torch.no_grad():
        for ecg, teacher_emb, ecg_labels, pulm_labels in eval_loader:
            ecg         = ecg.to(device)
            ecg_logits, pulm_logits, student_emb = student(ecg)

            all_ecg_logits.append(ecg_logits.cpu())
            all_ecg_labels.append(ecg_labels)
            all_pulm_logits.append(pulm_logits.cpu())
            all_pulm_labels.append(pulm_labels)
            all_student_emb.append(student_emb.cpu())
            all_teacher_emb.append(teacher_emb)

    ecg_logits_np  = torch.sigmoid(torch.cat(all_ecg_logits)).numpy()
    ecg_labels_np  = torch.cat(all_ecg_labels).clamp(min=0.0).numpy()
    pulm_logits_np = torch.sigmoid(torch.cat(all_pulm_logits)).numpy()
    pulm_labels_np = torch.cat(all_pulm_labels).clamp(min=0.0).numpy()
    student_emb_np = torch.cat(all_student_emb).numpy()
    teacher_emb_np = torch.cat(all_teacher_emb).numpy()

    # ── Student AUC ──────────────────────────────────────────────────
    student_ecg_auc  = compute_auc_per_class(ecg_logits_np,  ecg_labels_np,  ecg_class_names)
    student_pulm_auc = compute_auc_per_class(pulm_logits_np, pulm_labels_np, pulm_class_names)

    # ── Embedding cosine similarity ───────────────────────────────────
    # Normalize
    s_norm = student_emb_np / (np.linalg.norm(student_emb_np, axis=1, keepdims=True) + 1e-8)
    t_norm = teacher_emb_np / (np.linalg.norm(teacher_emb_np, axis=1, keepdims=True) + 1e-8)
    cos_sim = (s_norm * t_norm).sum(axis=1)
    mean_cos = float(np.mean(cos_sim))
    print(f"\nEmbedding alignment — mean cosine similarity: {mean_cos:.4f}  "
          f"(std={float(np.std(cos_sim)):.4f})")

    # ── Teacher linear probe ──────────────────────────────────────────
    # Load training set teacher embeddings to fit probes
    probe_ecg_auc  = {"macro": float("nan")}
    probe_pulm_auc = {"macro": float("nan")}

    emb_cache = cfg["teacher"].get("embedding_cache")
    if emb_cache and Path(emb_cache).exists():
        print("\nFitting linear probes on teacher embeddings (train split)...")
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.multioutput import MultiOutputClassifier
            from sklearn.preprocessing import StandardScaler

            train_ds = DistillDataset(
                data_path      = cfg["data"]["train_npy"],
                embed_cache    = emb_cache,
                study_id_cache = cfg["teacher"].get("study_id_cache"),
                lead_idx       = cfg["student"].get("input_lead", 0),
                signal_length  = cfg["student"].get("signal_length", 1000),
            )
            # Collect only embeddings + labels (no ECG signal needed for probe)
            probe_emb, probe_ecg_lab, probe_pulm_lab = [], [], []
            probe_loader = DataLoader(train_ds, batch_size=512, num_workers=4,
                                      shuffle=False, pin_memory=False)
            with torch.no_grad():
                for _, t_emb, e_lab, p_lab in probe_loader:
                    probe_emb.append(t_emb.numpy())
                    probe_ecg_lab.append(e_lab.clamp(min=0.0).numpy())
                    probe_pulm_lab.append(p_lab.clamp(min=0.0).numpy())

            probe_emb_np      = np.concatenate(probe_emb)
            probe_ecg_lab_np  = np.concatenate(probe_ecg_lab)
            probe_pulm_lab_np = np.concatenate(probe_pulm_lab)

            scaler = StandardScaler()
            probe_emb_np  = scaler.fit_transform(probe_emb_np)
            teacher_eval  = scaler.transform(teacher_emb_np)  # eval set teacher emb

            # ECG probe
            clf_ecg = MultiOutputClassifier(
                LogisticRegression(max_iter=300, C=1.0, solver="lbfgs"), n_jobs=-1)
            clf_ecg.fit(probe_emb_np, probe_ecg_lab_np)
            ecg_probe_probs = np.stack([e.predict_proba(teacher_eval)[:, 1]
                                         for e in clf_ecg.estimators_], axis=1)
            probe_ecg_auc = compute_auc_per_class(
                ecg_probe_probs, ecg_labels_np, ecg_class_names)

            # Pulm probe
            clf_pulm = MultiOutputClassifier(
                LogisticRegression(max_iter=300, C=1.0, solver="lbfgs"), n_jobs=-1)
            clf_pulm.fit(probe_emb_np, probe_pulm_lab_np)
            pulm_probe_probs = np.stack([e.predict_proba(teacher_eval)[:, 1]
                                          for e in clf_pulm.estimators_], axis=1)
            probe_pulm_auc = compute_auc_per_class(
                pulm_probe_probs, pulm_labels_np, pulm_class_names)

            print("Probes fitted successfully.")
        except Exception as e:
            print(f"  Linear probe failed: {e}")
    else:
        print("\nNo teacher embedding cache found — skipping linear probe baseline.")

    # ── Print results ─────────────────────────────────────────────────
    print_auc_table(
        f"ECG Rhythm Classification  (primary head)  — {args.split} set",
        student_ecg_auc, probe_ecg_auc, ecg_class_names,
    )
    print_auc_table(
        f"Pulmonary Pathology (secondary head, via distillation)  — {args.split} set",
        student_pulm_auc, probe_pulm_auc, pulm_class_names,
    )

    print(f"\n{'═'*60}")
    print(f"  SUMMARY")
    print(f"{'═'*60}")
    print(f"  Student ECG  macro AUC : {student_ecg_auc['macro']:.4f}  "
          f"(teacher probe: {probe_ecg_auc['macro']:.4f})")
    print(f"  Student Pulm macro AUC : {student_pulm_auc['macro']:.4f}  "
          f"(teacher probe: {probe_pulm_auc['macro']:.4f})")
    print(f"  Embedding cosine sim   : {mean_cos:.4f}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
