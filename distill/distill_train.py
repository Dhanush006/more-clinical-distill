"""
distill/distill_train.py

Distillation training loop — dual-head student with ResidualLossGate.

Teacher:  MoRE MultiModal (frozen) — multimodal ECG+CXR+Text
Student:  MobileNetV3-Small        — single-lead ECG only

Heads:
  Primary   (ecg_classifier):  10-class ECG rhythm  (multi-label BCE)
  Secondary (pulm_classifier):  4-class pulmonary    (multi-label BCE via distillation)

Loss: DistillationLoss — 6 per-sample terms gated by ResidualLossGate
  L_ecg, L_pulm, L_align_ecg, L_align_cxr, L_align_text, L_kl

Early stopping on validation macro-AUC for ECG head (primary task).

Usage:
    python distill/distill_train.py --config configs/distill_config.yaml
    python distill/distill_train.py --config configs/ablations/A2_lossgate.yaml
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

sys.path.append("./utils")
sys.path.append("./distill")

from student_model import build_student
from distill_dataset import DistillDataset
from distill_loss import DistillationLoss


# ── Training utilities ────────────────────────────────────────────────

def train_one_epoch(
    student, loss_fn, loader, optimizer, scaler, device, epoch, total_epochs
):
    student.train()
    total_loss       = 0.0
    term_totals      = {"L_ecg": 0., "L_pulm": 0., "L_align_ecg": 0.,
                        "L_align_cxr": 0., "L_align_text": 0.,
                        "L_kl": 0., "gate_entropy": 0.}
    n_batches        = 0

    bar = tqdm(loader, desc=f"Distill Epoch {epoch+1}/{total_epochs}")
    for ecg, t_ecg, t_cxr, t_text, ecg_labels, pulm_labels in bar:
        ecg         = ecg.to(device)
        t_ecg       = t_ecg.to(device)
        t_cxr       = t_cxr.to(device)
        t_text      = t_text.to(device)
        ecg_labels  = ecg_labels.to(device)
        pulm_labels = pulm_labels.to(device)

        with autocast():
            ecg_logits, pulm_logits, student_emb = student(ecg)
            result = loss_fn(
                student_ecg_logits  = ecg_logits,
                student_pulm_logits = pulm_logits,
                student_emb         = student_emb,
                teacher_ecg_logits  = ecg_logits.detach(),  # soft targets from student (no teacher logits cached)
                teacher_ecg_emb     = t_ecg,
                teacher_cxr_emb     = t_cxr,
                teacher_text_emb    = t_text,
                ecg_labels          = ecg_labels,
                pulm_labels         = pulm_labels,
            )
            loss = result["loss"]

        if not torch.isfinite(loss):
            optimizer.zero_grad()
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(student.parameters()) + list(loss_fn.gate.parameters()),
            max_norm=1.0,
        )
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        total_loss += loss.item()
        for k in term_totals:
            term_totals[k] += result.get(k, 0.0)
        n_batches += 1

        bar.set_postfix({
            "loss":    f"{loss.item():.4f}",
            "ecg":     f"{result['L_ecg']:.3f}",
            "align_e": f"{result['L_align_ecg']:.3f}",
            "H_gate":  f"{result['gate_entropy']:.3f}",
        })

    avg = total_loss / max(n_batches, 1)
    avg_terms = {k: v / max(n_batches, 1) for k, v in term_totals.items()}
    return avg, avg_terms


def compute_auc(logits_list, labels_list, label_name: str) -> tuple[float, np.ndarray]:
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
        for ecg, _t_ecg, _t_cxr, _t_text, ecg_labels, pulm_labels in loader:
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
        "ecg_bce":  ecg_bce,
        "ecg_auc":  ecg_macro,
        "ecg_per":  ecg_per,
        "pulm_bce": pulm_bce,
        "pulm_auc": pulm_macro,
        "pulm_per": pulm_per,
    }


# ── Main ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Distillation training with ResidualLossGate")
    p.add_argument("--config", default="configs/distill_config.yaml")
    return p.parse_args()


def _load_config(config_path: str) -> dict:
    """Load YAML config with optional _extends inheritance."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    base_path = cfg.pop("_extends", None)
    if base_path:
        with open(base_path) as f:
            base = yaml.safe_load(f)
        base.pop("_extends", None)
        # Deep-merge: cfg overrides base at the section level
        for section, overrides in cfg.items():
            if section in base and isinstance(base[section], dict) and isinstance(overrides, dict):
                base[section].update(overrides)
            else:
                base[section] = overrides
        return base
    return cfg


def main():
    args = parse_args()
    cfg = _load_config(args.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    ecg_class_names  = cfg["student"].get("ecg_class_names", [
        "Normal", "Sinus brady", "Sinus tachy", "AFib",
        "LBBB", "RBBB", "ST elev MI", "ST ischemia", "AV block", "LVH"
    ])
    pulm_class_names = cfg["student"].get("pulm_class_names", [
        "Atelectasis", "Cardiomegaly", "Edema", "Pleural Effusion"
    ])

    gate_cfg    = cfg.get("loss_gate", {})
    gate_enabled = gate_cfg.get("enabled", True)
    lambda_h     = gate_cfg.get("lambda_h", 0.1)
    modalities   = tuple(gate_cfg.get("modalities", ["ecg", "cxr", "text"]))

    triplet_cache = cfg["teacher"].get("triplet_cache")

    # ── Datasets ──────────────────────────────────────────────────────
    train_ds = DistillDataset(
        data_path      = cfg["data"]["train_npy"],
        triplet_cache  = triplet_cache,
        embed_cache    = cfg["teacher"].get("embedding_cache"),
        study_id_cache = cfg["teacher"].get("study_id_cache"),
        lead_idx       = cfg["student"].get("input_lead", 0),
        signal_length  = cfg["student"].get("signal_length", 1000),
        split          = "train",
    )
    val_ds = DistillDataset(
        data_path      = cfg["data"]["val_npy"],
        triplet_cache  = triplet_cache,
        lead_idx       = cfg["student"].get("input_lead", 0),
        signal_length  = cfg["student"].get("signal_length", 1000),
        split          = "val",
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

    # ── Model + Loss ──────────────────────────────────────────────────
    student = build_student(cfg["student"]).to(device)
    n_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"Student parameters: {n_params:,}")

    loss_fn = DistillationLoss(
        gate_enabled    = gate_enabled,
        lambda_h        = lambda_h,
        kl_temperature  = cfg["training"].get("kl_temperature", 4.0),
        modalities      = modalities,
        embedding_dim   = cfg["student"].get("embedding_dim", 128),
        gate_hidden_dim = gate_cfg.get("hidden_dim", 64),
    ).to(device)

    gate_params = list(loss_fn.gate.parameters())
    print(f"LossGate parameters: {sum(p.numel() for p in gate_params):,} "
          f"(enabled={gate_enabled})")

    optimizer = torch.optim.AdamW(
        list(student.parameters()) + gate_params,
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

    best_ecg_auc = 0.0
    patience     = cfg["training"].get("early_stopping_patience", 15)
    no_improve   = 0
    epochs       = cfg["training"]["epochs"]

    for epoch in range(epochs):
        t0 = time.time()
        train_loss, term_avgs = train_one_epoch(
            student, loss_fn, train_loader, optimizer, scaler, device, epoch, epochs
        )
        metrics = validate(student, val_loader, device, ecg_class_names, pulm_class_names)
        scheduler.step()

        elapsed = (time.time() - t0) / 60
        print(f"\nEpoch {epoch+1}/{epochs} | "
              f"train={train_loss:.4f}  "
              f"ecg_auc={metrics['ecg_auc']:.4f}  pulm_auc={metrics['pulm_auc']:.4f}  "
              f"H_gate={term_avgs['gate_entropy']:.3f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}  time={elapsed:.1f}min")
        print(f"  Terms: ecg={term_avgs['L_ecg']:.4f}  pulm={term_avgs['L_pulm']:.4f}  "
              f"align_ecg={term_avgs['L_align_ecg']:.4f}  "
              f"align_cxr={term_avgs['L_align_cxr']:.4f}  "
              f"align_text={term_avgs['L_align_text']:.4f}  "
              f"kl={term_avgs['L_kl']:.4f}")

        ecg_per_str  = "  ".join(f"{n}={v:.3f}" for n, v in
                                  zip(ecg_class_names, metrics["ecg_per"]))
        pulm_per_str = "  ".join(f"{n}={v:.3f}" for n, v in
                                  zip(pulm_class_names, metrics["pulm_per"]))
        print(f"  ECG  per-class: {ecg_per_str}")
        print(f"  Pulm per-class: {pulm_per_str}")

        if metrics["ecg_auc"] > best_ecg_auc:
            best_ecg_auc = metrics["ecg_auc"]
            no_improve   = 0
            ckpt = out_dir / f"student_best_ep{epoch+1}_ecgauc{metrics['ecg_auc']:.4f}.pth"
            torch.save({
                "epoch":          epoch + 1,
                "state_dict":     student.state_dict(),
                "gate_state":     loss_fn.gate.state_dict(),
                "ecg_auc":        metrics["ecg_auc"],
                "pulm_auc":       metrics["pulm_auc"],
                "ecg_per":        metrics["ecg_per"].tolist(),
                "pulm_per":       metrics["pulm_per"].tolist(),
                "gate_enabled":   gate_enabled,
                "modalities":     list(modalities),
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
