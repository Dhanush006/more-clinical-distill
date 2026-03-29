"""
distill/distill_train.py

Distillation training loop scaffold.

Teacher:  MoRE MultiModal (frozen) — multimodal ECG+CXR+Text
Student:  MobileNetV3-Small        — single-lead ECG only

Loss:
  L = alpha * L_task
    + beta  * L_kl        (soft logits KL divergence)
    + gamma * L_align     (embedding cosine similarity)
    + delta * L_xmodal    (preserve teacher ECG-CXR similarity matrix)

Usage:
    python distill/distill_train.py --config configs/distill_config.yaml

NOTE: This is a scaffold. The teacher forward pass and
      cross-modal similarity loss (L_xmodal) are stubbed out
      pending teacher weight availability (Phase 4 smoke test).
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
    """Binary cross-entropy for multi-label CheXpert classification."""
    # Replace -1 (uncertain) with 0 for BCE
    labels = labels.clamp(min=0.0)
    return F.binary_cross_entropy_with_logits(logits, labels)


def kl_soft_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 4.0,
) -> torch.Tensor:
    """KL divergence between softened teacher and student distributions."""
    p_teacher = F.softmax(teacher_logits / temperature, dim=-1)
    log_p_student = F.log_softmax(student_logits / temperature, dim=-1)
    return F.kl_div(log_p_student, p_teacher, reduction="batchmean") * (temperature ** 2)


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
    alpha = cfg["loss_weights"]["alpha"]
    gamma = cfg["loss_weights"]["gamma"]

    bar = tqdm(loader, desc=f"Distill Epoch {epoch+1}/{total_epochs}")
    for ecg, teacher_emb, labels in bar:
        ecg         = ecg.to(device)
        teacher_emb = teacher_emb.to(device)
        labels      = labels.to(device)

        with autocast():
            logits, student_emb = student(ecg)
            l_task  = task_loss(logits, labels)
            l_align = embedding_align_loss(student_emb, teacher_emb)
            # L_kl and L_xmodal require teacher logits / CXR embeddings;
            # stubbed as zero until teacher inference is wired up
            loss = alpha * l_task + gamma * l_align

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        total_loss += loss.item()
        bar.set_postfix({"loss": f"{loss.item():.4f}",
                         "l_task": f"{l_task.item():.4f}",
                         "l_align": f"{l_align.item():.4f}"})

    return total_loss / len(loader)


def validate(student, loader, device):
    student.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for ecg, teacher_emb, labels in loader:
            ecg    = ecg.to(device)
            labels = labels.to(device)
            logits, _ = student(ecg)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels).clamp(min=0.0)
    val_loss = F.binary_cross_entropy_with_logits(all_logits, all_labels).item()

    # AUC-ROC per class (more informative than BCE given class imbalance)
    probs = torch.sigmoid(all_logits).numpy()
    labs  = all_labels.numpy()
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(labs, probs, average="macro")
    except Exception:
        auc = float("nan")
    return val_loss, auc


# ── Main ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Distillation training")
    p.add_argument("--config", default="configs/distill_config.yaml")
    return p.parse_args()


def main():
    args  = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

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

    best_val_auc   = 0.0
    patience       = cfg["training"].get("early_stopping_patience", 15)
    no_improve     = 0
    epochs         = cfg["training"]["epochs"]

    for epoch in range(epochs):
        t0 = time.time()
        train_loss = train_one_epoch(
            student, train_loader, optimizer, scaler, device, cfg, epoch, epochs
        )
        val_loss, val_auc = validate(student, val_loader, device)
        scheduler.step()

        elapsed = (time.time() - t0) / 60
        print(f"\nEpoch {epoch+1}/{epochs} | "
              f"train={train_loss:.4f}  val_bce={val_loss:.4f}  "
              f"val_auc={val_auc:.4f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}  "
              f"time={elapsed:.1f}min")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            no_improve   = 0
            ckpt = out_dir / f"student_best_ep{epoch+1}_auc{val_auc:.4f}.pth"
            torch.save(student.state_dict(), ckpt)
            print(f"  Saved best checkpoint: {ckpt}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1} "
                      f"(no AUC improvement for {patience} epochs)")
                break

        if epoch % 10 == 0:
            ckpt = out_dir / f"student_ep{epoch}.pth"
            torch.save(student.state_dict(), ckpt)


if __name__ == "__main__":
    main()
