# Project Plan: more-clinical-distill
# Full Pipeline — Dataset → Distillation → Evaluation

**Created:** 2026-03-31
**Branch:** more-KD
**Goal:** Distill the MoRE multimodal teacher (~280M params) into a lightweight ECG-only student (MobileNetV3-Small, ~2.54M params) that achieves strong cardiovascular diagnostic performance from a single ECG lead.

---

## Current Status at Plan Creation

| Component | Status |
|---|---|
| Teacher weights | ✅ Downloaded |
| ECG signal files | ⚠️ 28,745 / 49,076 — download running (job 18223148) |
| CXR image files | ⚠️ 2,064 / 49,076 — download running (job 18223149) |
| Teacher embedding cache | ✅ 28,745 × 128-dim cached |
| Baseline student training | ✅ Complete (ECG AUC 0.704, Pulm AUC 0.474) |
| Ablation studies | ❌ Not started |
| Clinical text integration | ❌ Not started |
| Splits redesign | ❌ Not started |

---

## Phase 1 — Dataset Validation and Cleaning

**Depends on:** ECG + CXR downloads completing (jobs 18223148, 18223149)

### Objective
Verify that all downloaded files are intact and consistent with the manifest. Identify and log missing or corrupt samples so downstream phases use only reliable data.

### Tasks

1. **Count downloaded files vs manifest**
   - Count `.dat` files under `data/mimic-iv-ecg/`
   - Count `.jpg` files under `data/mimic-cxr-jpg/`
   - Compare against manifest row count (49,076)
   - Log which `ecg_study_id` / `cxr_study_id` are missing

2. **Validate ECG signal integrity**
   - For each downloaded `.dat`, attempt `wfdb.rdsamp(stem)`
   - Verify output shape is `(T, 12)` with T ≥ 5000
   - Flag and log any unreadable or malformed records

3. **Validate CXR image integrity**
   - For each downloaded `.jpg`, open with PIL and verify file is a valid image and size > 0
   - Flag and log corrupt images

4. **Cross-check manifest consistency**
   - Ensure every manifest row with a downloaded ECG file also has metadata in `machine_measurements.csv`
   - Ensure CheXpert label join succeeds for all CXR-available rows

5. **Generate clean manifest**
   - Produce `manifests/clean_manifest.csv` — rows where both ECG and CXR are available, all metadata present, no corrupt files
   - Produce `manifests/ecg_only_manifest.csv` — rows where ECG is available (no CXR requirement), for student training

### Expected Outputs
- `logs/phase1_validation.txt` — per-file check results
- `manifests/clean_manifest.csv` — fully paired ECG+CXR rows
- `manifests/ecg_only_manifest.csv` — ECG-available rows for student
- `logs/phase1_summary.txt` — total counts, corrupt counts, missing counts

### Checkpoint — Pass Criteria Before Phase 2
- [ ] Total available ECG records counted and confirmed ≥ 40,000 (expected ~49,076 after full download)
- [ ] Zero silent corruptions — all flagged files are logged
- [ ] `clean_manifest.csv` and `ecg_only_manifest.csv` exist and row counts match logs

---

## Phase 2 — Dataset Restructuring and Split Redesign

**Depends on:** Phase 1 complete

### Objective
Replace the current split (which uses the official MIMIC-CXR split file and yields 48,423 train / 379 val / 274 test) with a patient-stratified, class-balanced split that supports reliable generalization evaluation.

### Problem with current split
- Val set: only 379 records → unstable AUC estimates
- Test set: only 274 records → insufficient for per-class evaluation on rare classes (ST elevation MI: 480 total)
- Official MIMIC-CXR split was designed for CXR tasks, not ECG rhythm classification
- No explicit guarantee of patient-level separation (a patient could appear in train and val)

### Tasks

1. **Audit current split for leakage**
   - Check that no `subject_id` appears in more than one split
   - If leakage found, document extent

2. **Design new patient-stratified split**
   - Split at patient level (`subject_id`), not record level
   - Ratios: 80% train / 10% val / 10% test
   - Use stratified sampling on the multi-label ECG vector to preserve class proportions in each split
   - Use `iterstrat` (multi-label stratification) or custom iterative approach

3. **Verify class representation**
   - Compute per-class counts in train/val/test
   - Ensure rare classes (ST elevation MI, LBBB) have ≥ 20 samples in val and test each

4. **Address class imbalance — ECG**
   - Compute class frequencies from training split
   - Compute inverse-frequency weights per class for weighted BCE
   - Save weights to `configs/ecg_class_weights.json`

5. **Address class imbalance — augmentation strategy (design only)**
   - Identify which ECG classes are critically underrepresented (< 5% of dataset)
   - Document augmentation candidates: time-shift, amplitude scaling, Gaussian noise, lead dropout
   - Implementation deferred to Phase 5

6. **Regenerate `.npy` splits**
   - Re-run `scripts/convert_manifest_to_more_format.py` with new split assignments
   - Overwrite `data/processed/*.npy` with new train/val/test arrays

### Expected Outputs
- `manifests/split_v2.csv` — new patient-level splits with class-stratification
- `configs/ecg_class_weights.json` — per-class BCE weights
- `data/processed/train_v2.npy`, `val_v2.npy`, `test_v2.npy`
- `logs/phase2_split_stats.txt` — per-class distribution in each split

### Checkpoint — Pass Criteria Before Phase 3
- [ ] No `subject_id` appears in more than one split
- [ ] All 10 ECG classes present in val and test with ≥ 20 samples each
- [ ] Class distribution in train mirrors overall distribution (within 5% per class)
- [ ] New `.npy` files generated and readable

---

## Phase 3 — Clinical Text Integration

**Depends on:** Phase 1 complete (CXR files downloaded; text files available via manifest)

### Objective
Download and integrate real radiology reports (CXR) and ECG machine reports into the dataset items, replacing the synthetic CheXpert-label-derived xray notes currently used. This enriches the teacher's text modality signal.

### Background
- **ECG reports**: Already available via `machine_measurements.csv` (`report_0`–`report_6` columns). Currently used, but only for label parsing. Full free text can be used as ECG note.
- **CXR radiology reports**: `.txt` files under `data/mimic-cxr-jpg/files/**/s*.txt`. Previously not downloaded due to inode quota. With 500K inode limit now in effect, these are downloadable.

### Tasks

1. **Download CXR radiology report `.txt` files**
   - Build a manifest of `.txt` file URLs for all CXR studies in `clean_manifest.csv`
   - Submit SLURM job to download via `wget -nc` (authenticated)
   - Estimated: ~49,076 `.txt` files

2. **Validate text file availability**
   - Cross-check: for each CXR study, does the `.txt` file exist?
   - Log missing studies (some may genuinely lack reports)

3. **Parse and clean CXR radiology reports**
   - Extract the `FINDINGS` and `IMPRESSION` sections from each `.txt`
   - Fallback: full text if sections not found; label-derived note if no `.txt`
   - Strip boilerplate / PHI-scrubbing artifacts

4. **Rebuild ECG note text**
   - Replace current `"The report from ECG is: {report_0}"` with concatenated `report_0`–`report_6` (deduplicated)
   - Truncate to 512 tokens (RoBERTa limit)

5. **Update `.npy` items with real text**
   - Re-run preprocessing to rebuild items with real CXR radiology notes and richer ECG notes
   - Overwrite or create `data/processed/*_with_text.npy` variants

6. **Validate text-updated dataset**
   - Spot-check 20 random items for correct note content
   - Confirm no empty notes remain (fallback to label-derived if needed)

### Expected Outputs
- `data/mimic-cxr-jpg/files/**/s*.txt` — radiology report text files
- `manifests/cxr_reports_manifest.txt` — download list for text files
- `data/processed/train_v2_text.npy`, `val_v2_text.npy`, `test_v2_text.npy`
- `logs/phase3_text_stats.txt` — counts of real vs fallback notes

### Checkpoint — Pass Criteria Before Phase 4
- [ ] ≥ 90% of CXR studies have real radiology report text (not synthetic fallback)
- [ ] All `.npy` items have non-empty xray_note and ecg_note fields
- [ ] Text lengths within tokenizer limits (≤ 512 tokens after truncation)

---

## Phase 4 — Teacher-Side Preparation

**Depends on:** Phase 2 complete (new splits), Phase 3 complete (text updated)

### Objective
Recompute teacher embeddings on the full dataset (all ~49,076 records after complete download) using the new splits, and validate modality alignment quality.

### Tasks

1. **Rerun teacher ECG embedding cache on full dataset**
   - Re-run `distill/cache_teacher_embeddings.py` on all available ECG records (expected ~49,076 after full download)
   - Output: `data/processed/teacher_ecg_embeddings_v2.npy` shape `(N, 128)`
   - Track NaN rate; apply `nan_to_num` sanitization as before

2. **Compute teacher CXR embeddings (if CXR files available)**
   - Run teacher's CXR encoder on downloaded `.jpg` files
   - Output: `data/processed/teacher_cxr_embeddings_v2.npy` shape `(N, 128)`
   - Required for cross-modal loss (`L_xmodal`) in Phase 6

3. **Validate embedding quality**
   - Compute pairwise cosine similarity between ECG and CXR embeddings for same-patient pairs
   - Expected: paired embeddings should be more similar than random pairs (InfoNCE pretraining signal)
   - Log mean paired vs random cosine similarity

4. **Teacher linear probe re-baseline**
   - Refit `LogisticRegression` probe on new train split teacher embeddings
   - Evaluate on new val/test splits
   - This sets the updated performance ceiling for ablation comparisons

### Expected Outputs
- `data/processed/teacher_ecg_embeddings_v2.npy`
- `data/processed/teacher_cxr_embeddings_v2.npy` (if CXR available)
- `logs/phase4_teacher_baseline.txt` — updated teacher probe AUC per class

### Checkpoint — Pass Criteria Before Phase 5
- [ ] Full ECG embedding cache complete (NaN rate < 5%)
- [ ] Teacher probe ECG macro AUC within 2% of previous baseline (0.933) on new splits
- [ ] If CXR available: paired cosine similarity > random cosine similarity

---

## Phase 5 — Student Model Pipeline

**Depends on:** Phase 2 complete (splits), Phase 4 complete (embeddings)

### Objective
Build a robust, configurable student training pipeline supporting ECG signal augmentation, class-weighted loss, and multiple input lead configurations.

### Tasks

1. **ECG signal augmentation module**
   - Implement `distill/ecg_augmentations.py` with:
     - Time shift (random roll ±50 samples)
     - Amplitude scaling (random scale 0.8–1.2×)
     - Gaussian noise (σ = 0.01–0.05)
     - Lead dropout (zero out 1–2 leads randomly, training only)
   - Apply augmentations only during training (not val/eval)

2. **Configurable lead selection**
   - Update `distill/distill_dataset.py` to support `input_leads` config parameter
   - Support: single lead index (e.g., Lead I = 0), list of leads, or `"all"` (12-lead)
   - Current default: Lead I (index 0)

3. **Class-weighted BCE**
   - Load `configs/ecg_class_weights.json` in training loop
   - Pass weights to `torch.nn.BCEWithLogitsLoss(pos_weight=...)`
   - Make configurable via `distill_config.yaml`

4. **Update `distill_config.yaml`**
   - Add: `augmentation: {enabled, time_shift, amplitude_scale, noise_sigma, lead_dropout}`
   - Add: `input_leads: [0]` (default Lead I)
   - Add: `use_class_weights: true`
   - Add: `data_version: v2` (points to new `.npy` files)

5. **Validate updated pipeline end-to-end**
   - Run 2-epoch smoke test with augmentation + class weights on new splits
   - Confirm no NaN loss, correct batch shapes, AUC computed correctly

### Expected Outputs
- `distill/ecg_augmentations.py`
- Updated `distill/distill_dataset.py`
- Updated `distill_config.yaml`
- `logs/phase5_smoketest.log` — 2-epoch run confirming pipeline integrity

### Checkpoint — Pass Criteria Before Phase 6
- [ ] Smoke test runs 2 epochs without NaN or shape errors
- [ ] Augmentations verified visually (plot 3 augmented vs original samples)
- [ ] Class-weighted loss confirmed lower on rare classes vs unweighted baseline
- [ ] Lead selection config switches correctly between single/multi lead inputs

---

## Phase 6 — Distillation Losses

**Depends on:** Phase 5 complete

### Objective
Implement the full modular loss suite. Current baseline uses only cosine alignment + BCE. This phase adds KL divergence soft-label distillation and (if CXR embeddings available) cross-modal similarity matching.

### Loss Formulation

```
L = α_ecg × L_ecg_BCE
  + α_pulm × L_pulm_BCE
  + γ_align × L_align        (cosine embedding alignment — existing)
  + γ_kl × L_kl              (KL divergence on teacher soft logits — new)
  + γ_xmodal × L_xmodal      (cross-modal similarity matching — new, CXR-gated)
```

### Tasks

1. **Implement `L_kl` — KL divergence soft-label distillation**
   - Teacher linear probe (fit in Phase 4) generates soft probability vectors for each ECG training record
   - Cache these soft labels to `data/processed/teacher_ecg_soft_labels.npy`
   - In training loop: `L_kl = KLDivLoss(student_log_softmax, teacher_softmax)`
   - Temperature parameter T (default T=4) to soften distributions

2. **Implement `L_xmodal` — cross-modal alignment (CXR-gated)**
   - Only active when `teacher_cxr_embeddings_v2.npy` is available
   - `L_xmodal = 1 − cosine_sim(student_ecg_emb, teacher_cxr_emb)` for same-patient pairs
   - Encourages student ECG embeddings to align with teacher's CXR space
   - Gate: skip this term if CXR embedding not cached for a given record

3. **Refactor loss into `distill/distill_loss.py`**
   - `DistillationLoss` class with configurable weights: `alpha_ecg, alpha_pulm, gamma_align, gamma_kl, gamma_xmodal`
   - All weights loaded from `distill_config.yaml`
   - Modular: each term computable independently, logged separately to wandb/tensorboard

4. **Validate loss components**
   - Run 1 epoch with each loss term in isolation to confirm correct gradient flow
   - Confirm `L_kl` decreases as student logit distributions match teacher

### Expected Outputs
- `distill/distill_loss.py` — modular `DistillationLoss` class
- `data/processed/teacher_ecg_soft_labels.npy`
- Updated `distill_config.yaml` with full loss weight config
- `logs/phase6_loss_validation.txt` — per-term loss values at epoch 1

### Checkpoint — Pass Criteria Before Phase 7
- [ ] All loss terms computed without NaN on a full training batch
- [ ] Each term individually reduces over 1 epoch of training
- [ ] `distill_loss.py` unit test passes: correct output shapes and gradient flow
- [ ] `L_xmodal` gracefully skips when CXR embeddings unavailable

---

## Phase 7 — Ablation Studies

**Depends on:** Phase 6 complete (full loss suite)

### Objective
Systematically evaluate the effect of key design choices. All ablations use the same seed and new v2 splits for comparability.

### Ablation Grid

#### A — Loss Weight Ablations

| Run | α_ecg | α_pulm | γ_align | γ_kl | γ_xmodal | Description |
|---|---|---|---|---|---|---|
| A0 (baseline) | 1.0 | 0.5 | 0.5 | 0.0 | 0.0 | Current baseline |
| A1 | 1.0 | 0.0 | 0.5 | 0.0 | 0.0 | ECG + alignment only |
| A2 | 1.0 | 0.5 | 0.0 | 0.0 | 0.0 | No alignment |
| A3 | 1.0 | 0.5 | 0.5 | 0.5 | 0.0 | Add KL |
| A4 | 1.0 | 0.5 | 0.5 | 1.0 | 0.0 | Higher KL weight |
| A5 | 1.0 | 0.5 | 0.5 | 0.5 | 0.5 | Full loss (if CXR available) |

#### B — Lead Selection Ablations

| Run | Lead | Index | Notes |
|---|---|---|---|
| B0 (baseline) | Lead I | 0 | Standard limb lead |
| B1 | Lead II | 1 | Common rhythm strip lead |
| B2 | Lead V2 | 7 | Precordial, ventricular signal |
| B3 | 3-lead | [0,1,7] | I + II + V2 |
| B4 | 12-lead | all | Upper bound |

#### C — Class Balancing Ablations

| Run | Strategy | Description |
|---|---|---|
| C0 (baseline) | None | Unweighted BCE |
| C1 | Class-weighted BCE | Inverse frequency weights |
| C2 | Oversampling | Oversample rare classes (ST MI, LBBB) to 2× frequency |
| C3 | Augmentation | Add ECG augmentation for rare-class records only |

### Tasks

1. **Create ablation config system**
   - `configs/ablations/` directory with one `.yaml` per ablation run
   - Each overrides only the changed fields from base `distill_config.yaml`

2. **Create ablation SLURM sweep script**
   - `slurm/run_ablation.slurm` — parameterized job accepting `--config` argument
   - `scripts/submit_ablations.sh` — submits all ablation runs as a job array

3. **Run ablation grid**
   - Submit all A and B runs as job array (10 jobs)
   - Submit C runs after A/B complete (3 jobs)
   - Each job: full training to early stopping, then evaluation

4. **Collect and tabulate results**
   - `scripts/collect_ablation_results.py` — parses all eval logs, produces summary CSV
   - Output: `results/ablation_summary.csv`

### Expected Outputs
- `configs/ablations/*.yaml` — one per ablation run
- `slurm/run_ablation.slurm`
- `outputs/distill/ablation_*/` — checkpoint per run
- `results/ablation_summary.csv` — ECG macro AUC, per-class AUC, cosine sim, for all runs

### Checkpoint — Pass Criteria Before Phase 8
- [ ] All 13 ablation runs complete without crash
- [ ] `ablation_summary.csv` populated with AUC for every run
- [ ] Best configuration identified (highest ECG macro AUC on val set)

---

## Phase 8 — Evaluation

**Depends on:** Phase 7 complete

### Objective
Rigorous final evaluation of the best student configuration against the teacher baseline, with per-class analysis, calibration, and generalization checks.

### Metrics

| Metric | Description |
|---|---|
| AUC-ROC per class | Primary metric, all 10 ECG classes |
| Macro AUC | Unweighted mean across classes |
| Weighted AUC | Weighted by class frequency |
| Calibration (ECE) | Expected calibration error for confidence reliability |
| Embedding cosine sim | Mean cosine similarity to teacher on test set |
| Teacher probe AUC | Performance ceiling from teacher ECG embeddings |
| Parameter count | Model size (student vs teacher) |
| Inference time | Forward pass time (ms) per sample on GPU and CPU |

### Tasks

1. **Evaluate best ablation checkpoint on held-out test set**
   - Use test split from `test_v2.npy` (never seen during training or val-based early stopping)
   - Compute all metrics above

2. **Per-class deep-dive**
   - Plot ROC curves for all 10 ECG classes
   - Plot calibration curves (reliability diagrams)
   - Identify worst-performing classes and relate to class frequency

3. **Student vs teacher comparison table**
   - Per-class AUC: student vs teacher probe vs teacher upper bound
   - Embedding cosine similarity breakdown by class

4. **Inference efficiency benchmark**
   - Measure forward pass time for batch size 1 and 32 on:
     - A100 GPU (Grace cluster)
     - CPU-only
   - Measure peak GPU memory usage

5. **Error analysis**
   - Sample 20 high-confidence wrong predictions per poorly-performing class
   - Inspect ECG signals for potential labeling noise or signal quality issues

### Expected Outputs
- `results/final_evaluation.json` — all metrics for best model
- `results/figures/roc_curves.png` — per-class ROC
- `results/figures/calibration.png`
- `results/comparison_table.csv` — student vs teacher
- `logs/phase8_inference_benchmark.txt`

### Checkpoint — Pass Criteria Before Phase 9
- [ ] Test set AUC computed on held-out split (not val)
- [ ] Per-class ROC curves generated for all 10 classes
- [ ] Inference time measured and documented
- [ ] Student ECG macro AUC ≥ 0.75 (target), or best ablation documented with gap analysis

---

## Phase 9 — Final Validation and Reproducibility

**Depends on:** Phase 8 complete

### Objective
Ensure the full pipeline is clean, reproducible, and documented. Prepare for submission / sharing.

### Tasks

1. **Seed and reproducibility audit**
   - Set `torch.manual_seed`, `numpy.random.seed`, `random.seed` consistently across all scripts
   - Re-run best ablation from scratch with fixed seed, confirm AUC within ±0.005 of original

2. **Clean up outputs and checkpoints**
   - Keep only: best checkpoint per ablation, final eval results, summary CSVs
   - Remove intermediate/debug checkpoints

3. **Update all documentation**
   - `docs/implementation_plan.md` — mark all phases complete, add final results
   - `docs/distillation_plan.md` — update ablation results table
   - `docs/assumptions.md` — update A11 (CXR now available) and A12 (NaN rate on full data)
   - `README.md` — update results table and add final best model summary

4. **Final commit and tag**
   - Commit all result files, updated docs, and clean configs
   - Tag: `v1.0-ablations-complete`

### Expected Outputs
- All docs updated with final results
- `results/` directory with complete evaluation artifacts
- Git tag `v1.0-ablations-complete`

### Checkpoint — Done When
- [ ] Reproducibility re-run within ±0.005 AUC
- [ ] All docs reflect final state
- [ ] Repo is clean and pushable

---

## Dependency Graph

```
Downloads (jobs 18223148, 18223149)
        │
        ▼
Phase 1: Validation & Cleaning
        │
        ├──────────────────────┐
        ▼                      ▼
Phase 2: Split Redesign    Phase 3: Clinical Text
        │                      │
        └──────────┬───────────┘
                   ▼
        Phase 4: Teacher Prep (full embeddings)
                   │
                   ▼
        Phase 5: Student Pipeline (augmentation, leads)
                   │
                   ▼
        Phase 6: Distillation Losses (KL, xmodal)
                   │
                   ▼
        Phase 7: Ablation Studies
                   │
                   ▼
        Phase 8: Evaluation
                   │
                   ▼
        Phase 9: Final Validation
```

---

## Key Configuration Files

| File | Purpose |
|---|---|
| `configs/distill_config.yaml` | Main training hyperparameters |
| `configs/paths.yaml` | Data root paths |
| `configs/ecg_class_weights.json` | Per-class BCE weights (Phase 2 output) |
| `configs/ablations/*.yaml` | Per-ablation config overrides |
| `.env` | PhysioNet credentials (gitignored) |

## Key Scripts and Modules

| File | Purpose | Status |
|---|---|---|
| `distill/student_model.py` | MobileNetV3-Small dual-head student | ✅ Done |
| `distill/distill_dataset.py` | Dataset loader | ✅ Done — update in Phase 5 |
| `distill/distill_train.py` | Training loop | ✅ Done — update in Phase 5/6 |
| `distill/evaluate_student.py` | Benchmark evaluation | ✅ Done |
| `distill/cache_teacher_embeddings.py` | Teacher embedding cache | ✅ Done — rerun in Phase 4 |
| `distill/ecg_augmentations.py` | ECG augmentations | ❌ Phase 5 |
| `distill/distill_loss.py` | Modular loss module | ❌ Phase 6 |
| `scripts/convert_manifest_to_more_format.py` | Preprocessing / split generation | ✅ Done — update in Phase 2 |
| `scripts/collect_ablation_results.py` | Ablation result aggregation | ❌ Phase 7 |
