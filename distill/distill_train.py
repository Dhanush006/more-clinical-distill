"""
distill/distill_train.py

Distillation training loop — dual-head student.

Teacher:  MoRE MultiModal (frozen) — multimodal ECG+CXR+Text
Student:  MobileNetV3-Small        — single-lead ECG only

Heads:
  Primary   (ecg_classifier):  10-class ECG rhythm  (multi-label BCE)
  Secondary (pulm_classifier):  4-class pulmonary    (multi-label BCE via distillation)

Loss:
  L = alpha_ecg  * L_ecg_task    (ECG rhythm BCE — primary head)
    + alpha_pulm * L_pulm_task   (pulmonary BCE  — secondary head)
    + gamma      * L_align       (embedding cosine similarity to teacher)

Early stopping on validation macro-AUC for ECG head (primary task).

Usage:
    python distill/distill_train.py --config configs/distill_config.yaml
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

sys.path.append("./utils")
sys.path.append("./distill")

from student_model import build_student
from distill_dataset import DistillDataset


# ── Loss functions ────────────────────────────────────────────────────

def task_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy for multi-label classification.
    Uncertain labels (-1) are replaced with 0 for BCE."""
    return F.binary_cross_entropy_with_logits(logits, labels.clamp(min=0.0))


def embedding_align_loss(
    student_emb: torch.Tensor,
    teacher_emb: torch.Tensor,
) -> torch.Tensor:
    """1 - mean cosine similarity between student and teacher embeddings."""
    cos = F.cosine_similarity(student_emb, teacher_emb, dim=-1)
    return 1.0 - cos.mean()


# ── Training utilities ────────────────────────────────────────────────

def train_one_epoch(
    student, loader, optimizer, scaler, device, cfg, epoch, total_epochs
):
    student.train()
    total_loss = 0.0
    alpha_ecg  = cfg["loss_weights"]["alpha_ecg"]
    alpha_pulm = cfg["loss_weights"]["alpha_pulm"]
    gamma      = cfg["loss_weights"]["gamma"]

    bar = tqdm(loader, desc=f"Distill Epoch {epoch+1}/{total_epochs}")
    for ecg, teacher_emb, ecg_labels, pulm_labels in bar:
        ecg         = ecg.to(device)
        teacher_emb = teacher_emb.to(device)
        ecg_labels  = ecg_labels.to(device)
        pulm_labels = pulm_labels.to(device)

        with autocast():
            ecg_logits, pulm_logits, student_emb = student(ecg)
            l_ecg_task  = task_loss(ecg_logits, ecg_labels)
            l_pulm_task = task_loss(pulm_logits, pulm_labels)
            l_align     = embedding_align_loss(student_emb, teacher_emb)
            loss = (alpha_ecg  * l_ecg_task
                  + alpha_pulm * l_pulm_task
                  + gamma      * l_align)

        # Skip batch if loss is NaN (e.g., from all-zero teacher emb in batch)
        if not torch.isfinite(loss):
            optimizer.zero_grad()
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        total_loss += loss.item()
        bar.set_postfix({
            "loss":      f"{loss.item():.4f}",
            "ecg_task":  f"{l_ecg_task.item():.4f}",
            "pulm_task": f"{l_pulm_task.item():.4f}",
            "align":     f"{l_align.item():.4f}",
        })

    return total_loss / len(loader)


def compute_auc(logits_list, labels_list, label_name: str) -> tuple[float, np.ndarray]:
    """Compute per-class and macro AUC-ROC. Returns (macro_auc, per_class_auc)."""
    from sklearn.metrics import roc_auc_score
    probs = torch.sigmoid(torch.cat(logits_list)).numpy()
    labs  = torch.cat(labels_list).clamp(min=0.0).numpy()
    try:
        per_class = roc_auc_score(labs, probs, average=None)
        macro     = float(np.mean(per_class))
    except Exception as e:
        print(f"  [{label_name}] AUC error: {e}")
        n = probs.shape[1]
        per_class = np.full(n, float("nan"))
        macro = float("nan")
    return macro, per_class


def validate(student, loader, device, ecg_class_names, pulm_class_names):
    student.eval()
    ecg_logits_all, ecg_labels_all   = [], []
    pulm_logits_all, pulm_labels_all = [], []

    with torch.no_grad():
        for ecg, _teacher_emb, ecg_labels, pulm_labels in loader:
            ecg         = ecg.to(device)
            ecg_labels  = ecg_labels.to(device)
            pulm_labels = pulm_labels.to(device)
            ecg_logits, pulm_logits, _ = student(ecg)
            ecg_logits_all.append(ecg_logits.cpu())
            ecg_labels_all.append(ecg_labels.cpu())
            pulm_logits_all.append(pulm_logits.cpu())
            pulm_labels_all.append(pulm_labels.cpu())

    ecg_macro,  ecg_per  = compute_auc(ecg_logits_all,  ecg_labels_all,  "ECG")
    pulm_macro, pulm_per = compute_auc(pulm_logits_all, pulm_labels_all, "Pulm")

    ecg_bce  = F.binary_cross_entropy_with_logits(
        torch.cat(ecg_logits_all), torch.cat(ecg_labels_all).clamp(min=0.0)).item()
    pulm_bce = F.binary_cross_entropy_with_logits(
        torch.cat(pulm_logits_all), torch.cat(pulm_labels_all).clamp(min=0.0)).item()

    return {
        "ecg_bce":   ecg_bce,
        "ecg_auc":   ecg_macro,
        "ecg_per":   ecg_per,
        "pulm_bce":  pulm_bce,
        "pulm_auc":  pulm_macro,
        "pulm_per":  pulm_per,
    }


# ── Main ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Distillation training")
    p.add_argument("--config", default="configs/distill_config.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Class name lists for readable per-class AUC printing
    ecg_class_names  = cfg["student"].get("ecg_class_names", [
        "Normal", "Sinus brady", "Sinus tachy", "AFib",
        "LBBB", "RBBB", "ST elev MI", "ST ischemia", "AV block", "LVH"
    ])
    pulm_class_names = cfg["student"].get("pulm_class_names", [
        "Atelectasis", "Cardiomegaly", "Edema", "Pleural Effusion"
    ])

    # ── Datasets ──────────────────────────────────────────────────────
    train_ds = DistillDataset(
        data_path      = cfg["data"]["train_npy"],
        embed_cache    = cfg["teacher"].get("embedding_cache"),
        study_id_cache = cfg["teacher"].get("study_id_cache"),
        lead_idx       = cfg["student"].get("input_lead", 0),
        signal_length  = cfg["student"].get("signal_length", 1000),
    )
    val_ds = DistillDataset(
        data_path      = cfg["data"]["val_npy"],
        embed_cache    = None,
        lead_idx       = cfg["student"].get("input_lead", 0),
        signal_length  = cfg["student"].get("signal_length", 1000),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size  = cfg["training"]["batch_size"],
        num_workers = cfg["training"]["num_workers"],
        shuffle     = True,
        pin_memory  = True,
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = cfg["training"]["batch_size"],
        num_workers = cfg["training"]["num_workers"],
        shuffle     = False,
        pin_memory  = True,
    )

    # ── Model ─────────────────────────────────────────────────────────
    student = build_student(cfg["student"]).to(device)
    n_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"Student parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr           = cfg["training"]["learning_rate"],
        weight_decay = cfg["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0     = cfg["training"]["t_0"],
        eta_min = cfg["training"]["eta_min"],
    )
    scaler = GradScaler()

    out_dir = Path(cfg["outputs"]["checkpoint_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    best_ecg_auc  = 0.0
    patience      = cfg["training"].get("early_stopping_patience", 15)
    no_improve    = 0
    epochs        = cfg["training"]["epochs"]

    for epoch in range(epochs):
        t0 = time.time()
        train_loss = train_one_epoch(
            student, train_loader, optimizer, scaler, device, cfg, epoch, epochs
        )
        metrics = validate(student, val_loader, device, ecg_class_names, pulm_class_names)
        scheduler.step()

        elapsed = (time.time() - t0) / 60
        print(f"\nEpoch {epoch+1}/{epochs} | "
              f"train={train_loss:.4f}  "
              f"ecg_bce={metrics['ecg_bce']:.4f}  ecg_auc={metrics['ecg_auc']:.4f}  "
              f"pulm_bce={metrics['pulm_bce']:.4f}  pulm_auc={metrics['pulm_auc']:.4f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}  time={elapsed:.1f}min")

        # Per-class AUC
        ecg_per_str  = "  ".join(f"{n}={v:.3f}" for n, v in
                                  zip(ecg_class_names, metrics["ecg_per"]))
        pulm_per_str = "  ".join(f"{n}={v:.3f}" for n, v in
                                  zip(pulm_class_names, metrics["pulm_per"]))
        print(f"  ECG  per-class: {ecg_per_str}")
        print(f"  Pulm per-class: {pulm_per_str}")

        # Early stopping on primary (ECG) AUC
        if metrics["ecg_auc"] > best_ecg_auc:
            best_ecg_auc = metrics["ecg_auc"]
            no_improve   = 0
            ckpt = out_dir / f"student_best_ep{epoch+1}_ecgauc{metrics['ecg_auc']:.4f}.pth"
            torch.save({
                "epoch":       epoch + 1,
                "state_dict":  student.state_dict(),
                "ecg_auc":     metrics["ecg_auc"],
                "pulm_auc":    metrics["pulm_auc"],
                "ecg_per":     metrics["ecg_per"].tolist(),
                "pulm_per":    metrics["pulm_per"].tolist(),
            }, ckpt)
            print(f"  Saved best checkpoint: {ckpt.name}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1} "
                      f"(no ECG AUC improvement for {patience} epochs)")
                break

        if epoch % 10 == 0:
            ckpt = out_dir / f"student_ep{epoch}.pth"
            torch.save(student.state_dict(), ckpt)


if __name__ == "__main__":
    main()
