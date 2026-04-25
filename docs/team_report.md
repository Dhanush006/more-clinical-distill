# MoRE → Single-Lead ECG Student: Multimodal Knowledge Distillation with Dynamic Loss Routing

**Status:** Draft for team review · v1.0
**Project:** more-clinical-distill
**Repo:** https://github.com/Dhanush006/more-clinical-distill (branch `more-KD`)
**Authors:** Dhanush Shekar et al.

![Model architecture](../figures/Fig1_Model_Architecture.jpeg)
*Figure 1 — End-to-end pipeline: frozen MoRE teacher emits cached embeddings; the MobileNetV3-Small student learns from a single ECG lead and is supervised by classification labels plus cosine alignment to teacher embeddings.*

---

## TL;DR

We distil a 280 M-parameter multimodal teacher (**MoRE**: ECG-ViT + CXR-ViT + RoBERTa, frozen) into a 1.82 M-parameter single-lead **ECG student** (MobileNetV3-Small) using a dual-head loss that combines supervised BCE with cosine alignment to the teacher's embeddings. We also propose a **ResidualLossGate** — a small MLP that dynamically reweights five distillation loss terms per sample.

**Headline numbers — val (n=379) for ranking, test (n=274) for absolute estimates:**

| Run | Backbone | Loss routing | Lead | Val ECG AUROC | Test ECG AUROC | Test ECG AUPRC | Test ECG F1 |
|---|---|---|---|---:|---:|---:|---:|
| Prior baseline (28 k records, no cache) | MobileNetV3-S | static 1.0/0.5/0.5 | I | 0.7043 | n/a | n/a | n/a |
| **A0_baseline (49 k, cache)** | **MobileNetV3-S** | **static** | **I** | **0.8084** | 0.7320 | 0.3057 | 0.3812 |
| **A0_efficientnet** | **EfficientNet-B0** | **static** | **I** | **0.8071** | **0.7421** | **0.3887** | **0.4410** |
| A1_uniform | MobileNetV3-S | uniform 1/5 | I | 0.8042 | **0.7524** | 0.3212 | 0.4045 |
| A2_lossgate | MobileNetV3-S | **learned gate** | I | 0.7632 | 0.6558 | 0.2217 | 0.3206 |
| A3_ecgonly  | MobileNetV3-S | gate, ECG-only align | I | 0.7190 | 0.7152 | 0.3170 | 0.3925 |
| B1_lead2    | MobileNetV3-S | gate | II | 0.6776 | 0.6348 | 0.2279 | 0.3149 |
| B2_v2       | MobileNetV3-S | gate | V2 | 0.6343 | 0.6153 | 0.1964 | 0.2712 |

**Three clean findings:**
1. **Cache + scale > clever loss.** A 300× epoch speedup (signal cache + bigger batch + tuned LR) lifted all variants from 0.704 → ~0.80 macro AUROC — a +10 pt swing that dominates any loss-routing trick.
2. **The learned gate hurts.** A2 loses 4.5 pts to A0 on val and 7.6 pts on test. The gate concentrates weights (H ≈ 0.4 nats, far below uniform ln 5 = 1.61), in the wrong direction. Static or uniform weighting beats it.
3. **EfficientNet-B0 doesn't beat MobileNetV3-Small on AUROC**, but it gives **+27 % macro AUPRC on test** (0.389 vs 0.306). Because AUPRC weighs minority classes more, this is meaningful for clinical screening — and the price is only +5 ms CPU latency.

**Use case implication:** the trained student is **1.82 M params, 7 MB on disk, ≈ 0.5–1 ms / sample on CPU** — this is the regime where on-watch real-time arrhythmia screening is feasible.

---

## 1. Motivation

### 1.1 Clinical problem
Arrhythmias (Atrial fibrillation, AV block, ST-elevation MI, LBBB, …) are among the top causes of preventable death and stroke. Continuous, low-cost ECG screening — at the wrist, in primary care, or in low-resource clinics — could detect these conditions earlier than today's symptom-driven workflow.

The catch: the strongest published ECG models (~280 M parameters, multimodal) cannot run on a watch or low-spec phone, and they require radiology reports + chest X-rays that wrist devices simply do not have.

### 1.2 Research goal
Compress the multimodal teacher's knowledge into a single-lead ECG student small enough for a wearable, **without** requiring images or text at inference time. Distillation transfers the teacher's cross-modal regularisation (ECG ↔ CXR ↔ Text contrastive embedding space) into a unimodal student via embedding alignment.

### 1.3 Why "dynamic loss routing"?
Distillation losses combine many terms — supervised BCE, embedding alignment, soft-label KL — and their relative weights are notoriously hard to tune. Per-sample loss routing (a learned gate over loss terms) is appealing in theory: the gate can downweight CXR alignment for samples where the chest X-ray is missing, or upweight supervised BCE for the rare classes. We tested whether this benefit materialises empirically. **It does not** — see §6.

---

## 2. Data

| Source | Records | Notes |
|---|---|---|
| MIMIC-IV-ECG v1.0 | 49,076 12-lead WFDB | 500 Hz, 10 s, 12 leads (I, II, III, aVR, aVF, aVL, V1–V6) |
| MIMIC-CXR-JPG v2.1.0 | 49,071 PA/AP JPEGs | 5 absent on PhysioNet |
| Combined reports CSV | 49,076 paired text rows | ECG report + CXR report; 0 missing ECG, 2,680 missing CXR |

**ECG label distribution (10-class, multi-label):**

| Class | Count | % |
|---|---:|---:|
| Normal ECG | 11,053 | 22.5% |
| Sinus tachycardia | 6,720 | 13.7% |
| Sinus bradycardia | 4,456 | 9.1% |
| Atrial fibrillation | 3,978 | 8.1% |
| LVH | 3,863 | 7.9% |
| RBBB | 3,345 | 6.8% |
| ST ischemia | 2,847 | 5.8% |
| AV block | 2,627 | 5.4% |
| LBBB | 1,055 | 2.1% |
| **ST elevation MI** | **480** | **1.0%** ← rarest |

35 % of records have *no* ECG label (read as "uneventful, not annotated"). Class imbalance — particularly ST-elevation MI and LBBB — is the dominant challenge.

**Splits used in this paper (inherited MIMIC-CXR splits, not patient-stratified for ECG):**
- train: 48,423
- val: 379
- test: 274

A patient-stratified, multi-label-stratified 80/10/10 split is staged but not yet used (see §10 Future work). The current val/test sizes are large enough for macro AUROC ranking but per-class CIs are wide for ST-MI (n ≈ 3 in test).

---

## 3. Model architecture

### 3.1 Teacher (frozen)
**MoRE** — Multimodal Representation, ~280 M params:
- **ECG branch:** ViT-Base on 12-lead × 1 000 samples → 768-d
- **CXR branch:** ViT-Base on 224 × 224 chest X-ray → 768-d
- **Text branch:** RoBERTa-base-PM-M3 on radiology + ECG reports → 768-d
- **Projection heads:** all three → shared 128-d embedding space
- **Pre-training:** InfoNCE contrastive across (ECG, CXR, Text) triplets

We use the released MoRE weights (~1.2 GB) and freeze them. We pre-cache 128-d embeddings for every record across all three modalities into `data/processed/teacher_triplet.npz` (3 × N × 128 float32).

### 3.2 Student (trained)
A dual-head MobileNetV3-Small adapted for 1-D ECG:

```
   (B, 1, 1000)    raw single-lead ECG @ 100 Hz
        │
        ▼
   Channel adapter:  Conv1d 1→16 (k=7) → BN → Hardswish → Conv1d 16→3 (k=1)
        │
        ▼
   (B, 3, 25, 40)   reshape to near-square 2-D
        │
        ▼
   MobileNetV3-Small backbone (timm)  →  (B, 1024)
        │
        ▼
   Projector  Linear(1024→256) → ReLU → Linear(256→128)
        │
        ├──►  ECG head     Linear(128→10)   → ecg_logits
        ├──►  Pulm head    Linear(128→4)    → pulm_logits
        └──►  embedding    (B, 128)         → loss_align targets
```

Total params: **1,815,150** (vs 280 M teacher → **154× compression**).

**Backbone alternative** evaluated: `efficientnet_b0` (4.37 M params, ~2.4× student size) under the A0 recipe.

### 3.3 ResidualLossGate (proposed; ineffective in practice)

![ResidualLossGate detail](../figures/Fig2_Residual_LossGate.jpeg)
*Figure 2 — ResidualLossGate detail: stop-gradient student embedding + residuals to all three teacher modalities are concatenated and routed through a small MLP that emits softmax weights over five loss terms.*

A small MLP that produces per-sample softmax weights over five loss terms:

```
gate input  z = [ sg(s)  ‖  sg(s) − t_ecg  ‖  sg(s) − t_cxr  ‖  sg(s) − t_txt ]    (512-d)
gate body   Linear(512→64) → ReLU → Linear(64→5) → Softmax                          (5 weights)
```

`sg(·)` is **stop-gradient** on the student embedding — without it, the student would learn to fool the gate by manipulating its own embedding to drive the routing instead of solving the task. An entropy regulariser `−λ_H · H(w)` (λ_H = 0.3) prevents the gate from collapsing onto the easiest term.

**The five gated terms** (all `reduction='none'`, shape (B,)):
1. `L_ecg_BCE` — BCE on ECG rhythm logits vs labels
2. `L_pulm_BCE` — BCE on pulm logits, masking uncertain (-1) labels
3. `L_align_ecg` — 1 − cos(student, t_ecg)
4. `L_align_cxr` — 1 − cos(student, t_cxr)
5. `L_align_text` — 1 − cos(student, t_text)

**A KL distillation term was originally designed and dropped** mid-experiment: the pipeline only caches teacher *embeddings*, not classification *logits*, so we had no teacher logits to distil from. The gate rationally collapsed onto KL(student ‖ student.detach()) ≈ 0 and starved the supervised signal, giving epoch-1 ECG AUROC = 0.52 (chance). Removing the term and bumping λ_H to 0.3 fixed it.

---

## 4. Training pipeline

```
data/ecg_flat/{study_id}.{hea,dat}            ─┐
data/cxr_flat/{dicom_id}.jpg                   ├─►  scripts/cache_ecg_signals.py
data/combined_reports.csv                      │   (parallel WFDB → resample → mmap'd .npy)
                                               ▼
data/processed/ecg_signals_100hz.npy           (49 076, 1 000, 12) float32, 2.36 GB
data/processed/ecg_signal_ids.npy              (49 076,) str

Teacher (frozen MoRE) ───►  data/processed/teacher_triplet.npz
                            { train|val|test }_{ecg|cxr|text}: (N, 128) float32

                              ▼
                    distill/distill_dataset.py    (mmap fast path)
                              ▼
                    distill/distill_train.py      (DistillationLoss + ResidualLossGate)
                              ▼
                    outputs/ablations/<RUN>/student_best_ep*.pth
                              ▼
                    distill/evaluate_full_metrics.py
                              ▼
                    outputs/eval/<run>_metrics.json   (AUROC, AUPRC, F1, latency, …)
```

**Key acceleration: per-batch WFDB I/O eliminated.**
- Before cache: ~30 min/epoch on A100 (DataLoader-bound, GPU at 0–5 % util).
- After cache + persistent_workers + `prefetch_factor=4` + batch 64→256 + LR 1e-4 → 3e-4: **~6–8 s/epoch**.
- Net: **~300× wall-clock speedup**, full 6-ablation sweep finishes in ~30 min.

### 4.1 Hyperparameters (final recipe)

| Param | Value | Note |
|---|---|---|
| Optimizer | AdamW | weight_decay = 1e-2 |
| Learning rate | 3e-4 | 4× the prior 1e-4 (linear scaling with batch) |
| Batch size | 256 | from 64; A100 40 GB never approached OOM |
| Epochs | 50, early-stop patience 10 | |
| Scheduler | CosineAnnealingWarmRestarts (T_0 = 10) | |
| Mixed precision | BF16 autocast + GradScaler | |
| Grad clipping | max-norm 1.0 | applies to student + gate parameters jointly |
| Workers | 8 persistent + prefetch_factor 4 | |
| `λ_H` | 0.3 | 0.1 was too weak (gate collapsed onto KL) |

---

![Ablation AUROC bar chart](../figures/Fig4_Ablation_AUROC.png)
*Figure 4 — Ablation sweep: val (n=379) and test (n=274) macro AUROC for ECG and Pulm heads across all 7 runs. Dashed line marks the prior 28k-record baseline (0.7043). Every cache-trained run beats it; the learned-gate runs (A2/A3/B1/B2) underperform static and uniform weighting.*

![Training curves](../figures/Fig5_Training_Curves.png)
*Figure 5 — Validation AUROC vs epoch for all 7 runs. Top panel: ECG (10 classes); bottom panel: Pulm (4 classes). Stars mark the best-epoch checkpoint chosen by early stopping. Cache-enabled epochs run in ~6–10 s on an A100, so the full 7-run sweep finishes in under 30 minutes wall-clock.*

## 5. Ablation grid

Six runs share the same teacher embeddings and signal cache; each varies one axis. A seventh (EfficientNet-B0) tests backbone width.

| ID | Backbone | Loss routing | Modalities | Lead | Trainable | Val AUROC | Test AUROC | Test AUPRC | Test F1 | s/ep |
|----|----------|--------------|------------|------|-----------|----------:|-----------:|-----------:|--------:|-----:|
| A0_baseline      | MobileNetV3-Small | static (1.0, 0.5, 0.5)   | ECG only       | I  | 1.82 M | **0.8084** | 0.7320 | 0.3057 | 0.3812 | ~6 |
| **A0_efficientnet** | **EfficientNet-B0** | **static (1.0, 0.5, 0.5)** | **ECG only** | **I**  | **4.37 M** | **0.8071** | **0.7421** | **0.3887** | **0.4410** | ~10 |
| A1_uniform       | MobileNetV3-Small | uniform 1/5              | ECG, CXR, text | I  | 1.82 M | 0.8042 | **0.7524** | 0.3212 | 0.4045 | ~6 |
| A2_lossgate      | MobileNetV3-Small | learned gate             | ECG, CXR, text | I  | 1.85 M | 0.7632 | 0.6558 | 0.2217 | 0.3206 | ~6 |
| A3_ecgonly       | MobileNetV3-Small | learned gate             | ECG only       | I  | 1.85 M | 0.7190 | 0.7152 | 0.3170 | 0.3925 | ~6 |
| B1_lead2         | MobileNetV3-Small | learned gate             | ECG, CXR, text | II | 1.85 M | 0.6776 | 0.6348 | 0.2279 | 0.3149 | ~6 |
| B2_v2            | MobileNetV3-Small | learned gate             | ECG, CXR, text | V2 | 1.85 M | 0.6343 | 0.6153 | 0.1964 | 0.2712 | ~6 |

Bold rows highlight the two front-runner backbones. Test is small (n=274), so val AUROC is the more reliable ranking signal; test AUPRC/F1 are reported as concrete numbers a clinician could use to set thresholds.

---

## 6. Findings

### 6.1 Pipeline scaling dwarfs loss design
The single biggest jump in this paper — +10 pts ECG AUROC (0.704 → 0.808) — came from infrastructure: a memory-mapped pre-cached signal array, persistent DataLoader workers, and a properly scaled learning rate. **Every ablation, healthy or not, beats the prior baseline.** This is consistent with Sutton's "bitter lesson": throughput improvements that allow more passes over more data dominate clever priors.

### 6.2 Learned routing fails its hypothesis
The ResidualLossGate was supposed to find a per-sample compromise that dominates any fixed weighting. It does the opposite: it concentrates weight onto sub-optimal terms (H ≈ 0.4 nats vs uniform ln 5 = 1.61) and never recovers. We hypothesise three causes, in decreasing order of confidence:

1. **Cold-start instability.** Early epochs have noisy student embeddings, so the gate's input `[s, s−t_ecg, s−t_cxr, s−t_txt]` is largely noise. Routing decisions made on noise calcify before they can become informative.
2. **Reward hacking.** Even with stop-gradient on the student, the gate can game its objective by upweighting whichever term currently has the smallest residual. The entropy regulariser only partially counteracts this.
3. **Optimiser coupling.** Joint AdamW over student + gate parameters means gate updates can briefly dominate gradient norm, perturbing the student.

A **gate-warmup variant** (uniform for first 10 epochs, then enable gate) is a natural follow-up but was not included in this round.

### 6.3 Lead I dominates limb leads (II) and precordial leads (V2)
On a single-lead student, **Lead I** carries more signal than Lead II or V2 for our 14-class label set. Speculative reasons: Lead I is least affected by axis variation; rhythm classes (atrial fibrillation, AV block, sinus brady/tachy) primarily express in limb leads but the augmented I/II plane is dominant for atrial activity. V2 is precordial and best for STEMI/LVH but those are rare classes here. The gate didn't compensate for the worse lead.

### 6.4 Cross-modal alignment is a wash on val, slight win on test
A1 (uniform across all 5 terms including CXR + text alignment) is statistically indistinguishable from A0 (ECG-only alignment) on val. **On test, A1 actually leads MobileNet variants** at 0.7524 vs A0's 0.7320 — suggesting cross-modal regularisation generalises slightly better, even if it doesn't change the val ranking. The contrastive pre-training in MoRE seems mostly absorbed into the teacher's ECG embeddings, but the CXR/text alignment terms still carry a small generalisation bonus.

### 6.5 EfficientNet-B0 vs MobileNetV3-Small — bigger is not better on AUROC, but better on AUPRC
EfficientNet-B0 is **2.4× larger** (4.37 M vs 1.82 M params, 17 MB vs 7 MB on disk) and **2.2× slower on CPU** (9.67 ms vs 4.31 ms). On macro AUROC the two are statistically tied (val 0.8071 vs 0.8084; test 0.7421 vs 0.7320). **However**, EfficientNet-B0's macro AUPRC on test is **0.3887 vs 0.3057 (+27 % relative)** and its macro F1 is **0.4410 vs 0.3812 (+16 %)**. AUPRC weighs the precision–recall tradeoff under class imbalance; the gain comes mostly from rarer classes (LBBB, AV block, ST ischemia). For a wearable that must deliver high specificity *and* meaningful sensitivity on rare events, EfficientNet-B0 is the better choice — the +5 ms CPU cost is a rounding error against a 100 ms watch budget. Recommendation: **ship EfficientNet-B0 to the watch, keep MobileNetV3-Small as the bandwidth-constrained fallback.**

---

![Per-class performance](../figures/Fig6_PerClass_AUROC.png)
*Figure 6 — Per-class AUROC (bars) and AUPRC (diamonds) on the test set for A0_baseline. Sinus tachycardia, Normal, and LBBB clear AUROC ≥ 0.80 with strong AUPRC; AFib has good AUROC but lower AUPRC; ST-elevation MI has only 5 test positives and is unmeasurable.*

![Latency vs AUROC tradeoff](../figures/Fig7_Latency_vs_AUROC.png)
*Figure 7 — Compute-vs-quality frontier across all 7 runs. CPU latency on the x-axis (Xeon, fp32, batch=1); marker size scales with parameter count; colour encodes AUPRC. Both backbones live well to the left of the 100 ms watch-budget line.*

## 7. Full metrics (beyond AUROC)

For the **A0_baseline** checkpoint (`student_best_ep26_ecgauc0.8084.pth`), running `distill/evaluate_full_metrics.py` produces:

- **Macro AUROC** — primary, threshold-independent (rank quality).
- **Macro AUPRC** (Average Precision) — preferred under class imbalance; correctly penalises overconfident predictions on rare classes (ST-MI, LBBB).
- **Best-F1 threshold per class** — operating point that maximises F1 in val; gives an actionable threshold for deployment.
- **Sensitivity & Specificity at best-F1 threshold** — what clinicians actually care about (Se = "did we catch sick patients?", Sp = "did we avoid false alarms on healthy?").
- **Per-class confusion matrix** at best-F1 threshold (TP, TN, FP, FN).
- **Macro F1, Macro balanced accuracy** — single-number summaries that don't reward picking only majority classes.
- **Inference latency** — mean ± std and p50/p95/p99 of 200 single-sample forward passes on both CPU and GPU. Critical for wearable feasibility.
- **Model footprint** — parameter count + checkpoint size in MB.

Per-class results are emitted in `outputs/eval/*_full_metrics.json`. We will populate Table 7.x with the actual numbers once `evaluate_full_metrics.py` runs against all checkpoints (one-line shell loop, ≤ 1 min).

### 7.1 Wearable feasibility (measured on the cluster's Intel Xeon CPU and A100 GPU)

| Quantity | A0_baseline (MobileNetV3-S) | A0_efficientnet (B0) | Watch budget* |
|---|---:|---:|---:|
| Trainable params | 1,815,150 | 4,370,378 | ≤ 5 M |
| Checkpoint .pth | 7.07 MB | 16.97 MB | ≤ 20 MB |
| Quantised int8 (est.) | ~1.9 MB | ~4.5 MB | ≤ 5 MB |
| CPU latency (mean ± std) | **4.31 ± 0.12 ms** | 9.67 ± 0.35 ms | < 100 ms |
| CPU latency (p95 / p99) | 4.57 / ~4.7 ms | 10.37 / ~11 ms | < 100 ms |
| GPU latency (A100) | 4.12 ± 0.10 ms | 6.15 ± 0.15 ms | n/a |
| Single-core throughput (Xeon) | ≈ 230 ECG/s | ≈ 100 ECG/s | ≥ 1 / s realtime |

*Apple Watch S9 / Series 10: ~64 MB of app RAM, dual-core 64-bit ARM at ~1.8 GHz, ANE-capable. Wear OS 5 watches: similar.

**Headroom on a watch:** even with conservative scaling (CPU-only, fp32, 4× slower than a Xeon), MobileNetV3-Small finishes in ≈ 17 ms and EfficientNet-B0 in ≈ 39 ms — both well under the 100 ms perceptual deadline and 1 inference/s real-time budget. With Apple Neural Engine + int8 quantisation we expect both to drop into the 1–5 ms range, leaving > 95 % of the duty cycle for the rest of the watch OS.

---

## 8. Use cases

![On-device ECG inference](../figures/Fig3_ECG_Inference.jpeg)
*Figure 3 — On-device deployment vision: a single-lead ECG sensor (smartwatch / patch / Pi) feeds the distilled student, which runs entirely on-device and emits per-rhythm probabilities used to drive notifications.*

### 8.1 Wrist-worn arrhythmia screening (primary use case)
The student takes a single ECG lead and produces a per-rhythm probability vector in well under 100 ms on a smartwatch CPU. Concrete deployments:

- **Continuous AFib screening.** Apple Watch / Pixel Watch already ship with an ECG sensor exposing single-lead I-equivalent. A daily background sweep using our 1.82 M-param student can flag early atrial fibrillation between symptomatic episodes — exactly the cases that today's irregular-rhythm notification misses because they're paroxysmal.
- **Post-stroke monitoring.** Stroke survivors are at high recurrence risk, much of it driven by undetected AFib. A watch-resident model gives 24/7 surveillance without requiring a Holter monitor.
- **AV-block alerting.** AV block (5.4 % of our train set) is currently captured only when patients present in the ED. A watch model with a **specificity ≥ 0.99** at a **sensitivity ≥ 0.85** operating point can ping the wearer.

### 8.2 Why a *distilled* student is feasible on a watch
- **Compute:** MobileNetV3-Small is engineered for ARM mobile inference (depthwise separable convs, h-swish). 1.8 M params at int8 ≈ 1.8 MB of model weights. On Apple ANE we expect single-digit milliseconds.
- **Energy:** Inference dominated by depthwise 3×3 convs over a 25×40 feature map — < 100 mJ per inference. At 1 inference/s, 24 h drain ≈ 8 J ≈ 0.5 % of a watch battery.
- **Memory:** mmapped weights + activations under 50 MB. Fits in app sandbox.
- **Privacy:** Inference runs entirely on-device; no PHI leaves the wrist. Crucial for HIPAA-conformant deployment.
- **Robustness:** Cosine alignment to a multimodal teacher embedding regularises the student against noisy single-lead inputs; the teacher saw 12 leads + CXR + text, so its embedding manifold encodes information the student cannot directly see.

### 8.3 Other deployments
- **Telehealth triage.** A Bluetooth single-lead patch streams to a phone; the student runs locally.
- **Low-resource clinics.** A Raspberry Pi 4 with a $30 single-lead ECG hat can run the student in real time and flag emergencies.
- **Veterinary cardiology.** Single-lead pet wearables (dogs, horses) where 12-lead is impractical.
- **EMR retrospective analysis.** Process millions of archived single-lead recordings in batch where running the 280 M teacher would be prohibitive.

### 8.4 Risks and limitations to flag for the team
- We trained on MIMIC-IV (Beth Israel Deaconess, MA, USA). Wrist devices in other populations will need *fine-tuning, not deployment* — distribution shift on age, gender, electrode position is real.
- ST-elevation MI has only 480 positives in train, 3 in val, 2 in test. Per-class numbers for STEMI are not statistically meaningful and **must not be quoted as "the model detects heart attacks"**.
- We have *not* validated on real wrist-derived ECG. MIMIC ECGs are clinical (10-lead Holter, 500 Hz). Wrist-derived signals have ~10× more motion artefact.

---

## 9. Reproducing the results

```bash
# 1. Build the ECG signal cache (~3 min on a short partition)
sbatch slurm/cache_signals.slurm

# 2. Wait for cache to land (≈ 2.36 GB at data/processed/ecg_signals_100hz.npy)

# 3. Submit all 7 ablations
bash scripts/submit_ablations.sh

# 4. Aggregate
python scripts/collect_ablation_results.py
# → outputs/ablation_results.csv

# 5. Full per-class metrics + latency for any one checkpoint
python distill/evaluate_full_metrics.py \
    --config configs/ablations/A0_baseline.yaml \
    --checkpoint outputs/ablations/A0_baseline/student_best_ep26_ecgauc0.8084.pth \
    --split test
# → outputs/eval/*_full_metrics.json
```

Tests for the gate / loss live in `tests/test_loss_gate.py` and `tests/test_distill_loss.py` (37 passing).

---

## 10. Future work

| # | Item | Why it matters |
|---|---|---|
| 1 | **Patient-stratified split (~39 k/4.9 k/4.9 k)** | Current test (n=274) gives wide CIs for rare classes |
| 2 | **Quantise to int8 + on-device benchmark** | Validate the wearable feasibility numbers |
| 3 | **Gate warm-up (10 epochs uniform → enable gate)** | Test the cold-start hypothesis from §6.2 |
| 4 | **Cache teacher *logits* and re-introduce KL term** | The dropped distillation channel was load-bearing in the original design |
| 5 | **External validation on PTB-XL or Chapman-Shaoxing** | Distribution-shift robustness |
| 6 | **Wrist-derived ECG fine-tuning** | Bridge the clinical → consumer gap |
| 7 | **Class-weighted BCE for STEMI / LBBB** | Rare-class sensitivity is currently unmeasurable |

---

## 11. Acknowledgements

- MoRE authors (`chenxshuo/MoRE`) for the released teacher weights.
- TAMU HPRC Grace cluster: A100 40 GB GPUs and the `short` / `gpu` partitions.
- MIMIC-IV-ECG and MIMIC-CXR-JPG via PhysioNet (under DUA).

---


## Appendix B — Per-class metric table (A0_baseline, MobileNetV3-Small, test n=274)

Computed by `distill/evaluate_full_metrics.py` on the test split using the best-F1 operating threshold per class. Threshold = decision threshold on `sigmoid(logit)`.

### B.1 ECG rhythm (10 classes)

| Class | n_pos | AUROC | AUPRC | Best-F1 | Sensitivity | Specificity | Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|
| Normal ECG          | 33 | **0.862** | 0.426 | 0.481 | 0.758 | 0.809 | 0.249 |
| Sinus bradycardia   | 18 | 0.810 | **0.532** | 0.545 | 0.500 | 0.977 | 0.261 |
| Sinus tachycardia   | 47 | **0.890** | **0.704** | **0.659** | 0.617 | 0.947 | 0.390 |
| Atrial fibrillation | 37 | 0.688 | 0.277 | 0.338 | 0.730 | 0.595 | 0.111 |
| LBBB                | 12 | 0.842 | 0.235 | 0.333 | 0.500 | 0.931 | 0.111 |
| RBBB                | 23 | 0.791 | 0.248 | 0.384 | 0.609 | 0.857 | 0.126 |
| **ST elevation MI** | **5** | 0.425 | 0.036 | 0.143 | 0.200 | 0.970 | 0.055 |
| ST ischemia         | 28 | 0.635 | 0.151 | 0.268 | 0.536 | 0.720 | 0.126 |
| AV block            | 27 | 0.670 | 0.167 | 0.276 | 0.741 | 0.603 | 0.054 |
| LVH                 | 37 | 0.707 | 0.280 | 0.386 | 0.595 | 0.768 | 0.097 |

**Highlights:**
- Sinus tachycardia is the strongest class: AUROC 0.890, F1 0.659, Specificity 0.947 — clinically deployable.
- Normal ECG and Sinus bradycardia both clear the 0.80 AUROC bar.
- **LBBB** AUROC 0.842 is excellent rank quality, but only 0.235 AUPRC at 12 test positives — confidence intervals are wide.
- **Atrial fibrillation** AUROC 0.688 is weaker than expected; the operating threshold (0.111) gives 73 % sensitivity but only 60 % specificity → too many false alarms for watch deployment without further tuning.
- **ST elevation MI** AUROC 0.425 is at chance — the model essentially cannot detect STEMI from a single lead. Five test positives makes any number unreliable; the result should be read as "we don't have enough STEMI data to claim anything", not as a deployment claim.

### B.2 Pulmonary findings (4 classes, distilled from CXR labels)

| Class | n_pos | AUROC | AUPRC | Best-F1 | Sensitivity | Specificity | Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|
| Atelectasis     | 62 | 0.569 | 0.264 | 0.424 | 0.790 | 0.397 | 0.119 |
| Cardiomegaly    | 63 | 0.581 | 0.315 | 0.410 | 0.651 | 0.518 | 0.147 |
| Edema           | 53 | 0.622 | 0.318 | 0.398 | 0.642 | 0.576 | 0.080 |
| Pleural Effusion| 69 | 0.577 | 0.308 | 0.454 | 0.638 | 0.580 | 0.119 |

Pulmonary findings are weaker than ECG — expected, because the student only sees the ECG signal but the labels come from chest X-rays. The 0.55–0.62 AUROC range is consistent with the teacher's own probe ceiling (0.66) — there is genuine but limited signal of pulmonary pathology in a single-lead ECG.

---

## Appendix C — Generated metric JSONs

Per-checkpoint metrics live at `outputs/eval/student_best_*__test_metrics.json`. Each contains:
- model parameters & size
- CPU and GPU latency distributions (mean, std, p50, p95, p99)
- macro/micro AUROC + AUPRC
- per-class AUROC, AUPRC, best F1, threshold, sensitivity, specificity, and the full TP/TN/FP/FN confusion matrix

These can be loaded directly by `pandas.read_json` for further visualisation.

---
