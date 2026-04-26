"""
scripts/generate_report_figures.py

Build the report figures from training logs + collected ablation results:
  figures/Fig4_Ablation_AUROC.png  - grouped horizontal bar chart of ECG/Pulm AUROC
  figures/Fig5_Training_Curves.png - val AUROC vs epoch overlay for all 7 runs
  figures/Fig6_PerClass_AUROC.png  - per-class AUROC heat strip for A0_baseline
  figures/Fig7_Latency_vs_AUROC.png - tradeoff plot (CPU latency vs AUROC)

Usage:
    cd /scratch/user/dshekar/more-clinical-distill
    python scripts/generate_report_figures.py
"""

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
except ImportError:
    plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3})

REPO   = Path(__file__).resolve().parent.parent
LOGDIR = REPO / "logs" / "ablations"
EVAL   = REPO / "outputs" / "eval"
OUT    = REPO / "figures"
OUT.mkdir(parents=True, exist_ok=True)

RUNS = [
    ("A0_baseline_strat",     "MobileNetV3-S - static, ECG-only, Lead I"),
    ("A0_efficientnet_strat", "EfficientNet-B0 - static, ECG-only, Lead I"),
    ("A1_uniform_strat",      "MobileNetV3-S - uniform 1/5, all mods, Lead I"),
    ("A2_lossgate_strat",     "MobileNetV3-S - learned gate, all mods, Lead I"),
    ("A3_ecgonly_strat",      "MobileNetV3-S - learned gate, ECG-only, Lead I"),
    ("B1_lead2_strat",        "MobileNetV3-S - learned gate, all mods, Lead II"),
    ("B2_v2_strat",           "MobileNetV3-S - learned gate, all mods, V2"),
]

EPOCH_RE = re.compile(
    r"Epoch (\d+)/\d+ \| train=([\d.]+)\s+ecg_auc=([\d.]+)\s+pulm_auc=([\d.]+)\s+H_gate=([\d.]+)"
)


def parse_log(run_name):
    pattern = LOGDIR / f"{run_name}_*.log"
    logs = sorted(pattern.parent.glob(pattern.name))
    if not logs:
        return None
    log = logs[-1]
    text = log.read_text(errors="ignore")
    text = text.replace("\r", "\n")
    rows = []
    for m in EPOCH_RE.finditer(text):
        ep, tr, eau, pau, hg = m.groups()
        rows.append((int(ep), float(tr), float(eau), float(pau), float(hg)))
    return rows


def load_eval_json(run_name):
    candidates = list(EVAL.glob("student_best_*__test_metrics.json"))
    for p in candidates:
        with open(p) as f:
            d = json.load(f)
        if f"/{run_name}/" in d.get("checkpoint", ""):
            return d
    return None


# ---------- Fig4: Ablation AUROC bar chart ----------
def fig_ablation_bar():
    val_ecg, val_pulm, test_ecg, test_pulm = [], [], [], []
    labels = []
    for name, _desc in RUNS:
        rows = parse_log(name)
        ev   = load_eval_json(name)
        if not rows or not ev:
            continue
        labels.append(name.replace("_", "\n"))
        val_ecg.append(max(r[2] for r in rows))
        val_pulm.append(max(r[3] for r in rows))
        test_ecg.append(ev["ecg"]["macro_auroc"])
        test_pulm.append(ev["pulm"]["macro_auroc"])

    n = len(labels)
    y = np.arange(n)
    h = 0.18

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(y - 1.5*h, val_ecg,   h, label="Val ECG AUROC",   color="#1f77b4")
    ax.barh(y - 0.5*h, test_ecg,  h, label="Test ECG AUROC",  color="#aec7e8")
    ax.barh(y + 0.5*h, val_pulm,  h, label="Val Pulm AUROC",  color="#d62728")
    ax.barh(y + 1.5*h, test_pulm, h, label="Test Pulm AUROC", color="#ff9896")

    for yi, vals in zip(y, zip(val_ecg, test_ecg, val_pulm, test_pulm)):
        for k, v in enumerate(vals):
            offset = (k - 1.5) * h
            ax.text(v + 0.005, yi + offset, f"{v:.3f}", va="center", fontsize=8)

    ax.axvline(0.7043, color="grey", linestyle="--", alpha=0.7,
               label="Prior baseline (28k recs): 0.7043")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0.40, 0.92)
    ax.set_xlabel("Macro AUROC")
    ax.set_title("Single-Lead ECG Student - Ablation Sweep\n"
                 "Val (n=379) and Test (n=274), best epoch via early stopping",
                 fontsize=11)
    ax.legend(loc="lower right", framealpha=0.95, fontsize=9)
    fig.tight_layout()
    out = OUT / "Fig4_Ablation_AUROC.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# ---------- Fig5: Training curves overlay ----------
def fig_training_curves():
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    palette = plt.get_cmap("tab10")

    for i, (name, _desc) in enumerate(RUNS):
        rows = parse_log(name)
        if not rows:
            continue
        ep   = [r[0] for r in rows]
        ecg  = [r[2] for r in rows]
        pulm = [r[3] for r in rows]
        c = palette(i)

        axes[0].plot(ep, ecg, "-o", ms=3.5, color=c, label=name)
        best = int(np.argmax(ecg))
        axes[0].plot(ep[best], ecg[best], marker="*", ms=14,
                     color=c, markeredgecolor="black", linewidth=0.5)

        axes[1].plot(ep, pulm, "-o", ms=3.5, color=c, label=name)
        bestp = int(np.argmax(pulm))
        axes[1].plot(ep[bestp], pulm[bestp], marker="*", ms=14,
                     color=c, markeredgecolor="black", linewidth=0.5)

    axes[0].set_ylabel("Val Macro AUROC - ECG (10 classes)")
    axes[0].set_title("Training Dynamics Across 7 Ablations - Val AUROC vs Epoch\n"
                      "Star = best-epoch checkpoint (early stopping at patience 10)",
                      fontsize=11)
    axes[0].legend(loc="lower right", ncol=2, fontsize=8)
    axes[0].axhline(0.7043, color="grey", linestyle="--", alpha=0.6)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Val Macro AUROC - Pulm (4 classes)")
    axes[1].legend(loc="lower right", ncol=2, fontsize=8)

    fig.tight_layout()
    out = OUT / "Fig5_Training_Curves.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# ---------- Fig6: per-class AUROC strip (A0_baseline) ----------
def fig_per_class():
    ev = load_eval_json("A0_baseline")
    if ev is None:
        print("skip Fig6: A0_baseline metrics JSON not found")
        return
    classes = [c["class"] for c in ev["ecg"]["per_class"]] + \
              [c["class"] for c in ev["pulm"]["per_class"]]
    aurocs  = [c["auroc"] for c in ev["ecg"]["per_class"]] + \
              [c["auroc"] for c in ev["pulm"]["per_class"]]
    auprcs  = [c["auprc"] for c in ev["ecg"]["per_class"]] + \
              [c["auprc"] for c in ev["pulm"]["per_class"]]
    npos    = [c["n_pos"] for c in ev["ecg"]["per_class"]] + \
              [c["n_pos"] for c in ev["pulm"]["per_class"]]

    n = len(classes)
    y = np.arange(n)
    fig, ax = plt.subplots(figsize=(10, 6))
    colours = ["#2ca02c" if v >= 0.80 else ("#ff7f0e" if v >= 0.70 else "#d62728")
               for v in aurocs]
    ax.barh(y, aurocs, color=colours, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.scatter(auprcs, y, color="#1f77b4", marker="D", zorder=3,
               label="AUPRC", s=44)
    for yi, (a, p, n_) in enumerate(zip(aurocs, auprcs, npos)):
        ax.text(a + 0.005, yi - 0.15, f"AUROC {a:.3f}", va="center", fontsize=8)
        ax.text(p + 0.005, yi + 0.20, f"AUPRC {p:.3f} (n+={n_})",
                va="center", fontsize=8, color="#1f77b4")

    ax.axhline(9.5, color="black", linewidth=0.8)
    ax.text(0.02, 4.5,  "ECG rhythm",      rotation=90, va="center", fontsize=10, color="grey")
    ax.text(0.02, 11.5, "Pulm (CXR)",      rotation=90, va="center", fontsize=10, color="grey")
    ax.set_yticks(y, classes)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.05)
    ax.set_xlabel("AUROC (bar) and AUPRC (diamond)")
    ax.set_title("Per-Class Performance - A0_baseline (MobileNetV3-Small) on Test (n=274)\n"
                 "Green AUROC>=0.80 | Orange 0.70-0.80 | Red <0.70",
                 fontsize=11)
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = OUT / "Fig6_PerClass_AUROC.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# ---------- Fig7: latency vs AUROC tradeoff ----------
def fig_latency_tradeoff():
    pts = []
    for name, _ in RUNS:
        ev = load_eval_json(name)
        if ev is None:
            continue
        pts.append({
            "name":   name,
            "params": ev["model"]["n_params"] / 1e6,
            "cpu":    ev["latency"]["cpu"]["mean_ms"],
            "auroc":  ev["ecg"]["macro_auroc"],
            "auprc":  ev["ecg"]["macro_auprc"],
        })
    if not pts:
        print("skip Fig7: no eval JSONs")
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    sizes = [p["params"] * 80 for p in pts]
    sc = ax.scatter([p["cpu"] for p in pts],
                    [p["auroc"] for p in pts],
                    s=sizes,
                    c=[p["auprc"] for p in pts],
                    cmap="viridis", edgecolor="black", alpha=0.85)
    cbar = plt.colorbar(sc, ax=ax, label="Test ECG AUPRC")

    for p in pts:
        ax.annotate(p["name"],
                    (p["cpu"], p["auroc"]),
                    xytext=(7, 7), textcoords="offset points", fontsize=9)

    ax.set_xlabel("CPU latency per inference, mean ms (Xeon, fp32, batch=1)")
    ax.set_ylabel("Test ECG Macro AUROC")
    ax.set_title("Compute vs Quality Tradeoff\n"
                 "Marker size = params (M). Watch budget: <100 ms",
                 fontsize=11)
    ax.axvline(100, color="red", linestyle=":", alpha=0.6, label="100 ms watch budget")
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = OUT / "Fig7_Latency_vs_AUROC.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    fig_ablation_bar()
    fig_training_curves()
    fig_per_class()
    fig_latency_tradeoff()
    print("All figures in", OUT)
