# more-clinical-distill

Distil a 280M-parameter multimodal teacher (**MoRE**: ECG + Chest X-Ray + Text) into a 1.82M-parameter single-lead **ECG student** that is small and fast enough to run on a smartwatch. Trained on MIMIC-IV-ECG + MIMIC-CXR-JPG (49,076 paired records) on the Texas A&M Grace cluster.

**Primary results — patient-stratified 80/10/10 split (val / test n = 4 908 each):**

| Backbone | Params | .pth size | CPU latency (1 ECG) | Test ECG AUROC | Test ECG AUPRC |
|---|---:|---:|---:|---:|---:|
| MobileNetV3-Small | 1.82 M | 7.07 MB | **4.44 ms** | 0.7623 | 0.2692 |
| **EfficientNet-B0** | 4.37 M | 16.97 MB | 10.01 ms | **0.7955** | **0.3457** |
| Prior baseline (28 k records, legacy split) | 1.82 M | 7.07 MB | n/a | 0.7043 (val) | n/a |

> **Headline:** A patient-stratified 80/10/10 split (zero subject overlap, per-class proportions within 0.1 pt across splits) gives an honest read on rare-class generalisation. EfficientNet-B0 wins decisively (test AUROC 0.7955, AUPRC 0.3457 → +28 % over MobileNet). A learned ResidualLossGate over five distillation losses **hurts by 10.7 pts** AUROC vs static weighting; post-analysis confirmed an inverted gradient direction — the gate was trained to avoid hard terms rather than focus on them. The fix (stop-gradient decoupling + KL reinstatement via a frozen LR probe) is queued as **A4_correctgate_strat** (see `docs/team_report.md` §6.3). Legacy CXR-derived split numbers are kept as a parallel study in Appendix D.

---

## Pipeline at a glance

![Model architecture](figures/Fig1_Model_Architecture.jpeg)

A frozen MoRE teacher emits 128-d embeddings for every record across three modalities (ECG, CXR, text). Those embeddings are pre-cached to `data/processed/teacher_triplet.npz`. The student is a MobileNetV3-Small (or EfficientNet-B0) that takes a single ECG lead, predicts ECG rhythm + pulmonary findings via two heads, and is supervised by BCE + cosine-alignment to the cached teacher embeddings.

A second one-time cache (`data/processed/ecg_signals_100hz.npy`, 2.36 GB) holds all 12-lead waveforms resampled to 100 Hz, eliminating per-batch WFDB I/O and giving a **~300× epoch speedup**.

![ResidualLossGate detail](figures/Fig2_Residual_LossGate.jpeg)

The proposed ResidualLossGate (left out of the deployment student because it underperforms uniform/static weighting) routes per-sample loss weight by reading the student's embedding and its residuals to all three teacher modalities.

![On-device deployment](figures/Fig3_ECG_Inference.jpeg)

The trained student fits comfortably in a smartwatch budget: 1.82 M params, ~7 MB on disk, < 5 ms on CPU. Suitable for continuous AFib screening, post-stroke surveillance, and AV-block alerting.

---

## Results

![Ablation AUROC](figures/Fig4_Ablation_AUROC.png)

![Training curves](figures/Fig5_Training_Curves.png)

![Per-class performance](figures/Fig6_PerClass_AUROC.png)

![Latency vs AUROC](figures/Fig7_Latency_vs_AUROC.png)

Full per-class metrics, threshold tuning, and clinical interpretation are in [`docs/team_report.md`](docs/team_report.md).

---

## Repo layout

```
more-clinical-distill/
├── Readme.md                   # this file
├── CLAUDE.md                   # detailed project context (read for engineering work)
├── plan.md                     # forward execution plan
├── configs/
│   ├── paths.yaml              # all data/output paths
│   ├── distill_config.yaml     # training hyperparameters
│   └── ablations/              # 7 ablation configs (A0–B2 + EfficientNet)
├── data/
│   ├── ecg_flat/               # 49,076 × {study_id}.{hea,dat}
│   ├── cxr_flat/               # 49,071 × {dicom_id}.jpg
│   ├── combined_reports.csv    # paired ECG + CXR text
│   └── processed/
│       ├── more_{train,val,test}.npy
│       ├── ecg_signals_100hz.npy        # (49076, 1000, 12) cache
│       ├── ecg_signal_ids.npy
│       └── teacher_triplet.npz          # cached teacher ECG/CXR/text embeddings
├── distill/
│   ├── student_model.py        # configurable MobileNetV3-Small / EfficientNet-B0
│   ├── distill_dataset.py      # mmap fast path + WFDB fallback
│   ├── distill_train.py        # training loop with gate + DistillationLoss
│   ├── loss_gate.py            # ResidualLossGate
│   ├── distill_loss.py         # 5-term per-sample loss
│   ├── evaluate_full_metrics.py# AUROC, AUPRC, F1, Se/Sp, latency
│   └── evaluate_student.py     # legacy AUROC-only benchmark
├── scripts/
│   ├── cache_ecg_signals.py    # build ECG signal cache (~3 min)
│   ├── submit_ablations.sh     # fan out all 7 SLURM jobs
│   ├── collect_ablation_results.py  # → outputs/ablation_results.csv
│   └── generate_report_figures.py   # builds Fig4–Fig7 from logs + eval JSONs
├── slurm/
│   ├── cache_signals.slurm
│   ├── run_ablation.slurm
│   ├── distill_train.slurm
│   └── …
├── tests/                      # 37 TDD tests for gate + loss
├── figures/                    # report figures (jpegs + matplotlib pngs)
├── outputs/
│   ├── ablations/<run>/        # per-run student_best_*.pth
│   ├── eval/                   # full-metric JSONs per checkpoint
│   └── ablation_results.csv
└── docs/
    └── team_report.md          # publishable report (start here for the science)
```

---

## Quickstart on Grace

### 1. Environment

Use `deepship_xares2` (preinstalled with `torch`, `timm`, `wfdb`, `peft`, `transformers`):

```bash
conda activate /scratch/user/dshekar/.conda/envs/deepship_xares2
```

### 2. Build caches (once)

```bash
sbatch slurm/cache_signals.slurm     # 3 min on the short partition, ~2.36 GB output
```

The teacher triplet cache (`teacher_triplet.npz`) is already present; rebuild via `distill/cache_teacher_embeddings.py` only if you re-run the teacher.

### 3. Run the full ablation sweep

```bash
bash scripts/submit_ablations.sh                  # submit all 7 jobs
# or one at a time
bash scripts/submit_ablations.sh A0_baseline
```

Each job runs ~3–10 min wall-clock on a single A100 with the cache enabled.

### 4. Aggregate and evaluate

```bash
python scripts/collect_ablation_results.py        # → outputs/ablation_results.csv
for c in A0_baseline A0_efficientnet A1_uniform A2_lossgate A3_ecgonly B1_lead2 B2_v2; do
    CK=$(ls outputs/ablations/$c/student_best_*.pth | head -1)
    python distill/evaluate_full_metrics.py \
        --config configs/ablations/${c}.yaml \
        --checkpoint "$CK" --split test
done
python scripts/generate_report_figures.py         # → figures/Fig4–Fig7.png
```

### 5. Tests

```bash
pytest tests/ -v                                  # 37 tests, ~10 s
```

---

## Data sources

| Source | Records | Notes |
|---|---|---|
| MIMIC-IV-ECG v1.0 | 49,076 12-lead WFDB | 500 Hz × 10 s, 12 leads |
| MIMIC-CXR-JPG v2.1.0 | 49,071 PA/AP JPEGs | 5 absent on PhysioNet |
| Combined reports CSV | 49,076 paired text rows | ECG + CXR reports |

Both datasets are gated on PhysioNet — set `PHYSIONET_USER` and `PHYSIONET_PASS` in a local `.env` (never committed).

---

## Cluster-specific notes

- **Cluster:** TAMU HPRC Grace, SLURM
- **GPU partitions:** `gpu`, `medium` (A100 40 GB)
- **CPU partition:** `short` (used by `cache_signals.slurm`)
- **Proxy required for `wget` from compute nodes:** `http://10.73.132.63:8080`
- **Inode quota:** 500 k (currently ~260 k used after the flatten migration)

---

## License & acknowledgements

- MoRE teacher weights from [`chenxshuo/MoRE`](https://github.com/chenxshuo/MoRE).
- MIMIC datasets via PhysioNet; downstream use must comply with the PhysioNet DUA.
- TAMU HPRC for compute.
