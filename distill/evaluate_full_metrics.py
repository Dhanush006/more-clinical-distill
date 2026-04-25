"""
distill/evaluate_full_metrics.py

Full evaluation suite for a trained student checkpoint.

Reports for both ECG (10-class) and Pulm (4-class) heads:
  - Macro / micro AUROC
  - Macro / micro AUPRC (Average Precision)
  - Per-class AUROC + AUPRC
  - Best-F1 threshold per class + sensitivity/specificity at that threshold
  - Confusion matrix per class @ best-F1 threshold
  - Macro F1, balanced accuracy
  - Inference latency (ms/sample on CPU and GPU)
  - Model size on disk (MB), parameter count

Output: outputs/eval/{ckpt_basename}_full_metrics.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append("./distill")
sys.path.append("./utils")

import yaml
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_recall_curve, confusion_matrix, balanced_accuracy_score,
)

from student_model import build_student
from distill_dataset import DistillDataset


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    return ap.parse_args()


def _load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    base = cfg.pop("_extends", None)
    if base:
        with open(base) as f:
            full = yaml.safe_load(f)
        full.pop("_extends", None)
        for k, v in cfg.items():
            if k in full and isinstance(full[k], dict) and isinstance(v, dict):
                full[k].update(v)
            else:
                full[k] = v
        return full
    return cfg


def best_f1_threshold(y_true, y_score):
    """Find threshold maximizing F1; return (threshold, f1, prec, rec)."""
    if y_true.sum() == 0:
        return 0.5, 0.0, 0.0, 0.0
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    idx = int(np.argmax(f1[:-1])) if len(thr) > 0 else 0
    t = float(thr[idx]) if len(thr) > 0 else 0.5
    return t, float(f1[idx]), float(prec[idx]), float(rec[idx])


def evaluate_head(probs: np.ndarray, labels: np.ndarray, class_names: list[str]) -> dict:
    """Compute full per-class metrics, masking uncertain (-1) labels."""
    n_classes = probs.shape[1]
    per_class = []
    aurocs, auprcs = [], []
    for c in range(n_classes):
        m = labels[:, c] != -1
        yt = labels[m, c].astype(int)
        ys = probs[m, c]
        if yt.sum() == 0 or yt.sum() == len(yt):
            auroc = float("nan"); auprc = float("nan")
            t = 0.5; f1 = 0.0; prec = 0.0; rec = 0.0; spec = 0.0; tn = fp = fn = tp = 0
        else:
            auroc = float(roc_auc_score(yt, ys))
            auprc = float(average_precision_score(yt, ys))
            t, f1, prec, rec = best_f1_threshold(yt, ys)
            yp = (ys >= t).astype(int)
            cm = confusion_matrix(yt, yp, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            spec = float(tn) / max(tn + fp, 1)
        per_class.append({
            "class":       class_names[c] if c < len(class_names) else f"class_{c}",
            "n_pos":       int(yt.sum()) if hasattr(yt, "sum") else 0,
            "n_total":     int(len(yt)),
            "auroc":       auroc,
            "auprc":       auprc,
            "best_f1":     f1,
            "threshold":   t,
            "precision":   prec,
            "sensitivity": rec,
            "specificity": spec,
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        })
        if not np.isnan(auroc):
            aurocs.append(auroc); auprcs.append(auprc)

    flat_yt, flat_ys = [], []
    for c in range(n_classes):
        m = labels[:, c] != -1
        flat_yt.extend(labels[m, c].astype(int).tolist())
        flat_ys.extend(probs[m, c].tolist())
    flat_yt = np.array(flat_yt); flat_ys = np.array(flat_ys)
    micro_auroc = float(roc_auc_score(flat_yt, flat_ys)) if flat_yt.sum() > 0 else float("nan")
    micro_auprc = float(average_precision_score(flat_yt, flat_ys)) if flat_yt.sum() > 0 else float("nan")

    yp_macro = np.stack([(probs[:, c] >= per_class[c]["threshold"]).astype(int)
                         for c in range(n_classes)], axis=1)
    valid = (labels != -1).all(axis=1)
    macro_f1 = float(f1_score(labels[valid], yp_macro[valid], average="macro", zero_division=0)) \
        if valid.sum() > 0 else float("nan")
    bal_acc_per_class = []
    for c in range(n_classes):
        m = labels[:, c] != -1
        yt = labels[m, c].astype(int)
        if yt.sum() in (0, len(yt)):
            continue
        yp = (probs[m, c] >= per_class[c]["threshold"]).astype(int)
        bal_acc_per_class.append(balanced_accuracy_score(yt, yp))
    macro_bal_acc = float(np.mean(bal_acc_per_class)) if bal_acc_per_class else float("nan")

    return {
        "macro_auroc":     float(np.mean(aurocs)) if aurocs else float("nan"),
        "macro_auprc":     float(np.mean(auprcs)) if auprcs else float("nan"),
        "micro_auroc":     micro_auroc,
        "micro_auprc":     micro_auprc,
        "macro_f1":        macro_f1,
        "macro_bal_acc":   macro_bal_acc,
        "per_class":       per_class,
    }


def measure_latency(model, device, n_warm=20, n_iters=200) -> dict:
    """Mean ± std inference latency for a single ECG."""
    x = torch.randn(1, 1, 1000, device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(n_warm):
            _ = model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        ts = []
        for _ in range(n_iters):
            t0 = time.perf_counter()
            _ = model(x)
            if device == "cuda":
                torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000.0)
    return {"mean_ms": float(np.mean(ts)), "std_ms": float(np.std(ts)),
            "p50_ms": float(np.percentile(ts, 50)),
            "p95_ms": float(np.percentile(ts, 95)),
            "p99_ms": float(np.percentile(ts, 99))}


def main():
    args = parse_args()
    cfg = _load_config(args.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    student = build_student(cfg["student"]).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    student.load_state_dict(ckpt["state_dict"])
    student.eval()

    n_params = sum(p.numel() for p in student.parameters())
    ckpt_size_mb = os.path.getsize(args.checkpoint) / (1024 * 1024)

    split_key = "val" if args.split == "val" else "test"
    npy_key = "val_npy" if args.split == "val" else "test_npy"
    ds = DistillDataset(
        data_path       = cfg["data"][npy_key],
        triplet_cache   = cfg["teacher"].get("triplet_cache"),
        signal_cache    = cfg["data"].get("signal_cache"),
        signal_id_cache = cfg["data"].get("signal_id_cache"),
        lead_idx        = cfg["student"].get("input_lead", 0),
        signal_length   = cfg["student"].get("signal_length", 1000),
        split           = split_key,
    )
    loader = DataLoader(ds, batch_size=256, num_workers=4, shuffle=False, pin_memory=True)

    ecg_logits_all, ecg_labels_all = [], []
    pulm_logits_all, pulm_labels_all = [], []
    with torch.no_grad():
        for ecg, *_, ecg_labels, pulm_labels in loader:
            ecg = ecg.to(device)
            ecg_logits, pulm_logits, _ = student(ecg)
            ecg_logits_all.append(ecg_logits.cpu())
            ecg_labels_all.append(ecg_labels.cpu())
            pulm_logits_all.append(pulm_logits.cpu())
            pulm_labels_all.append(pulm_labels.cpu())

    ecg_probs  = torch.sigmoid(torch.cat(ecg_logits_all)).numpy()
    pulm_probs = torch.sigmoid(torch.cat(pulm_logits_all)).numpy()
    ecg_labels  = torch.cat(ecg_labels_all).numpy()
    pulm_labels = torch.cat(pulm_labels_all).numpy()

    ecg_class_names = cfg["student"].get("ecg_class_names", [f"ecg_{i}" for i in range(10)])
    pulm_class_names = cfg["student"].get("pulm_class_names", [f"pulm_{i}" for i in range(4)])

    ecg_metrics  = evaluate_head(ecg_probs,  ecg_labels,  ecg_class_names)
    pulm_metrics = evaluate_head(pulm_probs, pulm_labels, pulm_class_names)

    lat_gpu = measure_latency(student, device) if device == "cuda" else None
    student_cpu = student.to("cpu")
    lat_cpu = measure_latency(student_cpu, "cpu")

    out = {
        "checkpoint":      str(args.checkpoint),
        "config":          str(args.config),
        "split":           args.split,
        "n_eval_samples":  int(len(ds)),
        "model": {
            "n_params":        int(n_params),
            "ckpt_size_mb":    round(ckpt_size_mb, 3),
            "backbone":        cfg["student"].get("backbone", "mobilenetv3_small_100"),
            "embedding_dim":   cfg["student"].get("embedding_dim", 128),
            "input_lead":      cfg["student"].get("input_lead", 0),
        },
        "latency": {
            "cpu": lat_cpu,
            "gpu": lat_gpu,
        },
        "ecg":  ecg_metrics,
        "pulm": pulm_metrics,
    }

    out_path = args.out or f"outputs/eval/{Path(args.checkpoint).stem}__{args.split}_metrics.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n=== {Path(args.checkpoint).name} ({args.split}, n={len(ds)}) ===")
    print(f"Backbone: {out['model']['backbone']}  params: {n_params:,}  ckpt: {ckpt_size_mb:.2f} MB")
    print(f"  CPU latency: {lat_cpu['mean_ms']:.2f} ± {lat_cpu['std_ms']:.2f} ms (p95={lat_cpu['p95_ms']:.2f})")
    if lat_gpu:
        print(f"  GPU latency: {lat_gpu['mean_ms']:.2f} ± {lat_gpu['std_ms']:.2f} ms (p95={lat_gpu['p95_ms']:.2f})")
    print(f"ECG  | macro AUROC={ecg_metrics['macro_auroc']:.4f}  AUPRC={ecg_metrics['macro_auprc']:.4f}  "
          f"F1={ecg_metrics['macro_f1']:.4f}  bal_acc={ecg_metrics['macro_bal_acc']:.4f}")
    print(f"Pulm | macro AUROC={pulm_metrics['macro_auroc']:.4f}  AUPRC={pulm_metrics['macro_auprc']:.4f}  "
          f"F1={pulm_metrics['macro_f1']:.4f}  bal_acc={pulm_metrics['macro_bal_acc']:.4f}")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
