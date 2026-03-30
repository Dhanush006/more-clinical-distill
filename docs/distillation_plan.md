# Distillation Plan

**Last updated:** 2026-03-30

**Goal:** Compress MoRE's multimodal teacher (ECG + CXR + Text, ~280M params)
into a lightweight single-lead ECG student (MobileNetV3-Small, ~2.54M params)
that retains strong CVD classification performance from a single lead.

---

## Architecture

### Teacher — MoRE MultiModal (frozen)

| Component | Details |
|---|---|
| ECG encoder | ViT-Base + custom PatchEmbed (Conv1D 12→256→768, 196 patches) |
| CXR encoder | ViT-Base + DropKey attention |
| Text encoder | RoBERTa-base-PM-M3 with LoRA (r=16, 0.6% params trainable) |
| Projectors   | xray→128, ecg→128, text→128 |
| Pre-training | InfoNCE contrastive loss (ECG↔CXR↔Text) |
| Input ECG    | 12-lead, (12, 1000) at 100 Hz |
| Total params | ~280M |

Teacher weights: pre-trained MoRE weights downloaded from Google Drive (see original MoRE README).
Teacher is fully frozen during student training.

### Student — MobileNetV3-Small (implemented)

| Component | Details |
|---|---|
| Input         | Single-lead ECG (Lead I), shape (1, 1000) at 100 Hz |
| Channel adapter | Conv1d 1→16, BN, ReLU, Conv1d 16→3, BN, ReLU |
| Backbone      | MobileNetV3-Small (timm `mobilenetv3_small_100`, 2D conv treated as temporal) |
| Projector     | Linear(576→128) — matches teacher ECG projector output dim |
| ECG head      | Linear(128→10) — 10 ECG rhythm classes (primary) |
| Pulm head     | Linear(128→4) — 4 CheXpert pulmonary classes (secondary, via distillation) |
| Total params  | ~2.54M |

---

## ECG Rhythm Classification Labels (Primary Head)

10-class multi-label binary vector parsed from `machine_measurements.csv` free-text reports:

| Index | Class | Records | % |
|---|---|---|---|
| 0 | Normal | 11,053 | 22.5% |
| 1 | Sinus bradycardia | 4,456 | 9.1% |
| 2 | Sinus tachycardia | 6,720 | 13.7% |
| 3 | Atrial fibrillation | 3,978 | 8.1% |
| 4 | LBBB | 1,055 | 2.1% |
| 5 | RBBB | 3,345 | 6.8% |
| 6 | ST elevation MI | 480 | 1.0% |
| 7 | ST ischemia | 2,847 | 5.8% |
| 8 | AV block | 2,627 | 5.4% |
| 9 | LVH | 3,863 | 7.9% |

35% of records have no ECG rhythm label assigned (these contribute only via pulm head and alignment loss).

## Pulmonary Classification Labels (Secondary Head)

4-class CheXpert labels joined from `mimic-cxr-2.0.0-chexpert.csv`:
`Atelectasis, Cardiomegaly, Edema, Pleural Effusion`

---

## Distillation Losses (Implemented)

```
L = alpha_ecg × L_ecg_BCE  +  alpha_pulm × L_pulm_BCE  +  gamma × L_align
```

| Term | Formula | Weight | Purpose |
|---|---|---|---|
| `L_ecg_BCE` | BCE(ecg_logits, ecg_labels) | 1.0 | Supervised ECG rhythm classification |
| `L_pulm_BCE` | BCE(pulm_logits, pulm_labels) | 0.5 | Supervised pulmonary distillation |
| `L_align` | 1 − CosineSim(student_emb, teacher_ecg_emb) | 0.5 | Embedding space alignment |

**Note on original plan:** The proposal included `L_kl` (soft logit KL divergence) and `L_xmodal` (cross-modal similarity matching). These were simplified in implementation:
- `L_kl` replaced by hard-label BCE on teacher-derived soft labels (more stable with frozen teacher)
- `L_xmodal` deferred — requires CXR embeddings which are unavailable due to missing CXR files

**Training stability fixes:**
- NaN guard: skip batch if loss is non-finite
- Gradient clipping: `clip_grad_norm_(max_norm=1.0)` before optimizer step
- Teacher embedding sanitization: `np.nan_to_num(raw, nan=0.0)` at load time (502 NaN rows in cache)

---

## Training Setup

- Optimizer: AdamW (lr=1e-4, weight_decay=1e-4)
- LR schedule: CosineAnnealingWarmRestarts (T_0=10)
- AMP: bfloat16 mixed precision via `torch.cuda.amp.GradScaler`
- Early stopping: on ECG macro AUC, patience=15 epochs
- Batch size: 128
- Max epochs: 100
- Hardware: 1× A100 40GB (Grace cluster)

---

## Evaluation (Benchmark Design)

`distill/evaluate_student.py` computes:
1. **Student AUC-ROC** per class + macro for both heads on val and test sets
2. **Teacher linear probe** — `MultiOutputClassifier(LogisticRegression)` fit on teacher train embeddings, evaluated on test/val. Represents the ceiling achievable from teacher ECG embeddings alone.
3. **Embedding cosine similarity** — mean cosine sim between student and teacher embeddings on test set

---

## Planned Ablations

### Lead Ablation
Train three student variants to compare single-lead performance:

| Variant | Lead index | Notes |
|---|---|---|
| Lead I  | 0 | Standard limb lead (current default) |
| Lead II | 1 | Often used in rhythm strips |
| V2      | 7 | Precordial, good for ventricular signals |

### Loss Weight Ablation
Vary `alpha_pulm` (0.0, 0.25, 0.5, 1.0) and `gamma` (0.0, 0.25, 0.5, 1.0):
- `alpha_pulm=0` tests ECG-only training (no pulm distillation)
- `gamma=0` tests task-only training (no embedding alignment)

### Class Imbalance
Test class-weighted BCE for ECG head (weights inversely proportional to class frequency) to improve performance on ST elevation MI (480 records) and LBBB (1,055 records).

---

## File Map

| File | Purpose | Status |
|---|---|---|
| `distill/student_model.py` | MobileNetV3-Small dual-head student | ✅ Done |
| `distill/distill_dataset.py` | Dataset: (single-lead ECG, teacher_emb, ecg_labels, pulm_labels) | ✅ Done |
| `distill/distill_train.py` | Training loop with dual-head loss | ✅ Done |
| `distill/evaluate_student.py` | Benchmark: student vs teacher linear probe | ✅ Done |
| `distill/cache_teacher_embeddings.py` | Pre-compute teacher ECG embeddings | ✅ Done |
| `configs/distill_config.yaml` | All hyperparameters | ✅ Done |
| `slurm/distill_train.slurm` | SLURM job (GPU, 8h) | ✅ Done |
| `slurm/evaluate_student.slurm` | SLURM evaluation job | ✅ Done |

---

## Execution Order

```
1. Download teacher weights (pre-trained MoRE from Google Drive)
   → outputs/teacher/best_multimodel.pth

2. Cache teacher ECG embeddings
   sbatch slurm/cache_teacher_embeddings.slurm
   → data/processed/teacher_ecg_embeddings.npy  (28,745 × 128)

3. Run distillation training
   sbatch slurm/distill_train.slurm
   → outputs/distill/student_best_ep{N}_ecgauc{X}.pth

4. Run evaluation benchmark
   sbatch slurm/evaluate_student.slurm
   → logs/eval_{jobid}.log (per-class AUC comparison table)

5. Lead ablation (pending)
   Modify distill_config.yaml input_leads, resubmit

6. Loss weight ablation (pending)
   Modify loss_weights in config, resubmit
```
