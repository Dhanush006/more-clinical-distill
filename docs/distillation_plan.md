# Distillation Plan

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

### Student — MobileNetV3-Small

| Component | Details |
|---|---|
| Input         | Single-lead ECG (Lead I by default), shape (1, 1000) |
| Channel adapter | Conv1D: 1→16→3 (adapts single lead to 3-channel for backbone) |
| Backbone      | MobileNetV3-Small (timm `mobilenetv3_small_100`) |
| Projector     | Linear(576→256→128) — matches teacher ECG projector output dim |
| Classifier    | Linear(128→4) — 4 CheXpert labels |
| Total params  | ~2.54M |

---

## Distillation Losses

```
L = α·L_task + β·L_kl + γ·L_align + δ·L_xmodal
```

| Term | Formula | Purpose |
|---|---|---|
| `L_task` | BCE(student_logits, labels) | Supervised CVD classification |
| `L_kl` | KL(σ(teacher_logits/T) ‖ σ(student_logits/T)) | Soft label transfer (T=4) |
| `L_align` | 1 − CosineSim(student_emb, teacher_ecg_emb) | Embedding space alignment |
| `L_xmodal` | MSE(student similarity matrix, teacher ECG-CXR sim matrix) | Preserve cross-modal structure |

Default weights: α=1.0, β=0.5, γ=0.5, δ=0.3 (tune via ablation).

---

## Training Data

- Same 49,076 matched subjects from `matched_patients_v1.csv`
- Student uses **ECG only** — no CXR or text at inference time
- Teacher embeddings are **pre-computed and cached** before distillation training
  (avoids re-running teacher forward pass every step)

### Lead Ablation Study

Train three separate students to compare lead quality:

| Variant | Lead index | Name |
|---|---|---|
| Lead I  | 0 | Standard limb lead |
| Lead II | 1 | Often used in rhythm strips |
| V2      | 7 | Precordial, good for LV |

---

## Evaluation

- **Primary metric:** AUC-ROC on test set for each of the 4 CheXpert labels
- **Baseline comparison:**
  - Teacher (MoRE ECG encoder only, 12-lead)
  - GLoRIA (CXR-text baseline from MoRE paper)
  - MedKLIP
- **Lead ablation:** compare Lead I vs II vs V2 student performance

---

## File Map

| File | Purpose |
|---|---|
| `distill/student_model.py` | MobileNetV3-Small 1-D ECG student |
| `distill/distill_dataset.py` | Dataset: (single-lead ECG, teacher_emb, labels) |
| `distill/distill_train.py` | Training loop with 4-term loss |
| `configs/distill_config.yaml` | All hyperparameters |
| `slurm/distill_train.slurm` | SLURM job (24h, 1×A100) |

---

## Execution Order

```
1. Run Phase 4 smoke test → confirm teacher loads correctly
   sbatch slurm/smoke_test.slurm

2. Pre-train teacher (or load pre-trained weights from Readme.md link)
   sbatch slurm/pretrain.slurm   # (future Phase 4 full run)

3. Cache teacher ECG embeddings
   python distill/cache_teacher_embeddings.py   # (to be implemented)

4. Run distillation
   sbatch slurm/distill_train.slurm

5. Evaluate & lead ablation
   python distill/evaluate.py --lead 0  # Lead I
   python distill/evaluate.py --lead 1  # Lead II
   python distill/evaluate.py --lead 7  # V2
```

---

## Open Items

- [ ] `distill/cache_teacher_embeddings.py` — batch inference script for teacher ECG embeddings
- [ ] `distill/evaluate.py` — AUC-ROC evaluation + lead ablation
- [ ] Implement `L_xmodal` (cross-modal similarity matching) — requires CXR embeddings alongside ECG
- [ ] Tune loss weights α, β, γ, δ via small grid search
- [ ] Decide on KL temperature T (default 4.0 per Hinton et al.)
