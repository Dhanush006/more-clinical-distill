# ResidualLossGate — Design Specification

**Project:** more-clinical-distill
**Author:** Dhanush Shekar
**Date:** 2026-04-24
**Status:** Approved

---

## 1. Motivation

The current distillation loss uses static scalar weights (`α_ecg=1.0`, `α_pulm=0.5`, `γ=0.5`). These weights are the same for every sample regardless of how hard each sample is or how well each teacher modality supervises it. A sample with a noisy CXR report should lean on ECG alignment; a Normal ECG with rich text findings should lean on text alignment. Static weights waste signal.

The ResidualLossGate learns, per sample, how to allocate learning effort across 6 loss terms using the geometric distance between the student and each teacher modality as a signal.

---

## 2. Architecture: ResidualLossGate

### 2.1 Gate Input

The gate receives a 512-dimensional vector constructed from:

```
z = [sg(s) ‖ sg(s) − t_ecg ‖ sg(s) − t_cxr ‖ sg(s) − t_txt]
```

- `s`: student embedding, shape `(B, 128)` — **stop-gradients applied** (`s.detach()`)
- `t_ecg`, `t_cxr`, `t_txt`: teacher embeddings, shape `(B, 128)` — frozen, already detached
- Each residual `sg(s) − t_*` encodes geometric distance per modality
- Stop-gradient on `s` is critical: the gate reads student state for routing, but must not receive gradient feedback through the gate input, which would create a pathological loop where the student learns to fool the gate rather than learn the task

### 2.2 Gate Architecture

```
Linear(512 → 64) → ReLU → Linear(64 → 6) → Softmax
```

- ~33k parameters — negligible relative to student (~2.54M)
- Softmax output: `w ∈ R^6`, `Σw_i = 1`, all weights positive
- Trained jointly with the student; gradients flow back from the 6 weighted losses

### 2.3 Output: 6 Loss Weights

| Index | Term | Type |
|---|---|---|
| 0 | L_ecg_BCE | Supervised ECG rhythm classification |
| 1 | L_pulm_BCE | Supervised pulmonary classification (via distillation) |
| 2 | L_align_ecg | Cosine alignment to teacher ECG embedding |
| 3 | L_align_cxr | Cosine alignment to teacher CXR embedding |
| 4 | L_align_text | Cosine alignment to teacher text embedding |
| 5 | L_kl | KL divergence soft-label distillation |

### 2.4 Entropy Regularization

To prevent the gate from collapsing to a single loss term (degenerate routing):

```
L_entropy_reg = −λ_H × H(w)    where H(w) = −Σ w_i log(w_i)
```

This is subtracted from the total loss (maximizing entropy = encouraging diverse routing).
Default: `λ_H = 0.1`

---

## 3. Total Loss

### Per-Sample Losses (reduction='none')

```
L_ecg_BCE[i]   = BCE(student_ecg_logits[i], ecg_labels[i])
L_pulm_BCE[i]  = BCE(student_pulm_logits[i], pulm_labels[i])
L_align_ecg[i] = 1 − cosine_sim(s[i], t_ecg[i])
L_align_cxr[i] = 1 − cosine_sim(s[i], t_cxr[i])
L_align_text[i]= 1 − cosine_sim(s[i], t_txt[i])
L_kl[i]        = KL(softmax(student_ecg_logits[i]/T) ‖ softmax(teacher_logits[i]/T))
```

### Gated Batch Loss

```
loss_vec[i] = [L0, L1, L2, L3, L4, L5][i]          # shape (B, 6)
w[i]        = gate(z[i])                              # shape (B, 6)

L_total = mean_over_batch( sum_k( w[i,k] × loss_vec[i,k] ) ) − λ_H × H(w)
```

---

## 4. Data: teacher_triplet.npz

All 3 teacher modality embeddings are pre-computed and available in the sister repo. No new caching infrastructure needed.

**Source:** `/scratch/user/dshekar/more-clinical-diagoniser/data/processed/teacher_triplet.npz`
**Integration:** Symlinked to `data/processed/teacher_triplet.npz` in this repo

**Keys and shapes:**

| Key | Shape | Description |
|---|---|---|
| `train_ecg` | (48423, 128) | Teacher ECG embeddings, train split |
| `train_cxr` | (48423, 128) | Teacher CXR embeddings, train split |
| `train_text` | (48423, 128) | Teacher text embeddings, train split |
| `val_ecg` | (379, 128) | Teacher ECG embeddings, val split |
| `val_cxr` | (379, 128) | Teacher CXR embeddings, val split |
| `val_text` | (379, 128) | Teacher text embeddings, val split |
| `test_ecg` | (274, 128) | Teacher ECG embeddings, test split |
| `test_cxr` | (274, 128) | Teacher CXR embeddings, test split |
| `test_text` | (274, 128) | Teacher text embeddings, test split |
| `train_ecg_labels` | (48423, 10) | ECG rhythm multi-labels |
| `train_pulm_labels` | (48423, 4) | Pulmonary multi-labels |
| `train_ids` | (48423,) | Study IDs for alignment |
| (val/test variants) | — | Same structure |

---

## 5. Dataset Integration

`distill/distill_dataset.py` will be updated to:
- Load `teacher_triplet.npz` on init, split by `split` arg
- Return `(ecg_signal, t_ecg, t_cxr, t_text, ecg_labels, pulm_labels)` per item
- Use `study_id` from `.npy` to index into npz arrays via an ID→index map built at init

ECG signal loading (`wfdb.rdrecord`) remains unchanged.

---

## 6. Ablation Grid

6 runs covering gate utility, modality ablation, and lead selection:

| Run | Name | Gate | Modalities | Lead |
|---|---|---|---|---|
| A0 | baseline | None (static: 1.0/0.5/0.5) | ECG only | I (0) |
| A1 | uniform | None (1/6 per term) | All 3 modalities | I (0) |
| A2 | lossgate-full | ResidualLossGate | All 3 modalities | I (0) |
| A3 | lossgate-ecgonly | ResidualLossGate (ECG residual only) | ECG only | I (0) |
| B1 | lead-ii | ResidualLossGate | All 3 modalities | II (1) |
| B2 | lead-v2 | ResidualLossGate | All 3 modalities | V2 (7) |

All runs use the same hyperparameters except those varied. Each run gets its own config in `configs/ablations/`.

**Primary metric:** ECG macro AUC on test set
**Secondary metrics:** Pulm macro AUC, embedding cosine similarity, gate entropy over training

---

## 7. File Map

### New Files

| File | Purpose |
|---|---|
| `distill/loss_gate.py` | `ResidualLossGate` module |
| `distill/distill_loss.py` | `DistillationLoss` with 6 per-sample terms |
| `tests/test_loss_gate.py` | Unit tests for gate shapes, stop-grad, entropy |
| `tests/test_distill_loss.py` | Unit tests for all 6 loss terms, gated total |
| `configs/ablations/A0_baseline.yaml` | Ablation config |
| `configs/ablations/A1_uniform.yaml` | Ablation config |
| `configs/ablations/A2_lossgate.yaml` | Ablation config |
| `configs/ablations/A3_ecgonly.yaml` | Ablation config |
| `configs/ablations/B1_lead2.yaml` | Ablation config |
| `configs/ablations/B2_v2.yaml` | Ablation config |
| `slurm/run_ablation.slurm` | Parametric SLURM job (takes config as env var) |
| `scripts/submit_ablations.sh` | Fan-out: submits all 6 ablation jobs |
| `scripts/collect_ablation_results.py` | Aggregates results into ablation_results.csv |

### Modified Files

| File | Change |
|---|---|
| `distill/distill_dataset.py` | Load all 3 teacher embeddings from teacher_triplet.npz |
| `distill/distill_train.py` | Integrate ResidualLossGate; use DistillationLoss |
| `configs/distill_config.yaml` | Add `loss_gate:` section (lambda_H, embedding_dim) |

---

## 8. Config Extension

```yaml
# configs/distill_config.yaml additions
loss_gate:
  enabled: true
  lambda_H: 0.1          # entropy regularization coefficient
  hidden_dim: 64          # MLP hidden layer size
  input_dim: 512          # 4 × embedding_dim (128)

teacher:
  triplet_cache: data/processed/teacher_triplet.npz  # all 3 modalities

training:
  kl_temperature: 4.0     # T for soft-label KL loss
```

---

## 9. Testing Strategy (TDD)

Tests are written before implementation. Each test defines the expected contract.

**`tests/test_loss_gate.py`:**
- Output shape is `(B, 6)` for any batch size
- All weights positive and sum to 1 per sample
- Stop-gradient: `gate(z.detach())` — verify `s.grad` is None after backward through gate
- Entropy regularization term is negative (encourages high entropy)
- Gate can be disabled (returns uniform `1/6` weights) for ablations A0, A1

**`tests/test_distill_loss.py`:**
- Each of 6 loss terms is finite and non-negative for valid inputs
- Per-sample reduction returns shape `(B,)` for each term
- Gated total loss is a scalar
- Pulm labels with `-1` (uncertain) are masked out of BCE
- KL loss uses temperature correctly

---

## 10. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Gate collapse (all weight on 1 term) | Entropy regularization (λ_H=0.1) |
| Pathological feedback through gate input | Stop-gradient on student embedding before gate |
| Uncertain pulm labels (−1) inflating loss | Mask uncertain labels before BCE |
| teacher_triplet.npz study IDs don't align with .npy | Build explicit ID→index map at dataset init; assert no misses |
| A0 baseline uses only 3 terms (current code) | A0 config explicitly sets `loss_gate.enabled=false` with original 3-term loss |
