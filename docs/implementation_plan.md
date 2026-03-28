# Implementation Plan: more-clinical-distill

**Date:** 2026-03-28
**Branch:** more-KD
**Goal:** Manifest-driven multimodal pipeline → MoRE pretraining → Knowledge distillation scaffold

---

## Overview

This plan covers 5 phases. Each phase builds on the previous. The immediate deliverable is a working, smoke-testable MoRE pipeline fed by a selective subset download from PhysioNet, guided by `matched_patients_v1.csv`.

---

## Phase 0 — Inspect and Plan ✅ COMPLETE

**Deliverables:**
- [x] `docs/repo_audit.md`
- [x] `docs/manifest_audit.md`
- [x] `docs/implementation_plan.md`

**Key findings:**
- MoRE expects a `.npy` file of `[xray_path, ecg_stem, xray_note, ecg_note, labels, split]` tuples
- ECG files are WFDB format (`.hea` + `.dat`), loaded with `wfdb.rdsamp(stem)`
- CXR images are JPEG, loaded with PIL
- Text model is RoBERTa-base-PM-M3, loaded from `./data/RoBERTa-base-PM-M3/`
- Multiple hardcoded paths and missing imports in original scripts must be fixed
- Manifest has 49,076 matched pairs, 0 nulls, all unique subjects

---

## Phase 1 — Environment and Reproducibility

### Folder Structure to Create
```
more-clinical-distill/
├── data/
│   ├── mimic-iv-ecg/          # Downloaded ECG .hea/.dat files + metadata CSVs
│   ├── mimic-cxr-jpg/         # Downloaded CXR .jpg files + metadata CSVs
│   └── processed/             # MoRE-format .npy output files
├── manifests/                 # Derived file lists for downloads
├── scripts/                   # Python and shell scripts
├── slurm/                     # Slurm job scripts
├── logs/                      # Run logs
├── configs/                   # YAML config files
├── docs/                      # Documentation (this folder)
├── outputs/                   # Model checkpoints and results
└── distill/                   # Placeholder for distillation phase
```

### Files to Create
1. `.gitignore` — exclude `data/`, `outputs/`, `logs/`, `*.npy`, `*.pth`, secrets
2. `configs/paths.yaml` — all configurable paths in one place
3. `.env.example` — credential template (never commit `.env`)
4. `README.md` update — setup flow for HPC

---

## Phase 2 — Manifest-Driven Download Pipeline

### Scripts to Create

#### `scripts/inspect_manifest.py`
- Load `matched_patients_v1.csv` safely
- Validate columns: `subject_id`, `ecg_path`, `cxr_path`, `ecg_date`, `cxr_date`, `day_diff`
- Report: row count, null counts, path format, duplicate subjects
- Output summary to stdout and `logs/manifest_inspection.txt`

#### `scripts/build_download_lists.py`
- Input: `matched_patients_v1.csv` + `configs/paths.yaml`
- For each ECG row: emit `{ecg_path}.hea` and `{ecg_path}.dat`
- For each CXR row: emit `{cxr_path}`
- Also emit paths for required metadata files:
  - `mimic-cxr-2.0.0-metadata.csv`
  - `mimic-cxr-2.0.0-split.csv`
  - `mimic-cxr-2.0.0-chexpert.csv`
  - `machine_measurements.csv`
  - `record_list.csv`
  - Radiology report `.txt` files (by study_id from manifest)
- Output: `manifests/ecg_files.txt`, `manifests/cxr_files.txt`
- Deduplication: ensure no repeated download paths
- Logging: print total file counts

#### `scripts/download_subset.sh`
- Read credentials from environment (never hardcoded): `$PHYSIONET_USER`, `$PHYSIONET_PASS`
- Loop over `manifests/ecg_files.txt` and `manifests/cxr_files.txt`
- Use `wget` with:
  - `--user` / `--password` from env
  - `--no-clobber` (skip if file exists)
  - `--directory-prefix` to set local root
  - `--tries=3` (retry on failure)
- Log failed downloads to `logs/download_failures.txt`
- Resume-safe by design

#### `slurm/download_subset.slurm`
- Partition, account, walltime as configurable placeholders
- Sets `PHYSIONET_USER` and `PHYSIONET_PASS` from a sourced `.env` file
- Calls `scripts/download_subset.sh`
- Redirects output to `logs/slurm_download_%j.out`

---

## Phase 3 — MoRE Compatibility Layer

### Scripts to Create

#### `scripts/verify_local_dataset_layout.py`
- Input: `matched_patients_v1.csv` + `configs/paths.yaml`
- Check for each row:
  - ECG: `{ecg_root}/{ecg_path}.hea` exists
  - ECG: `{ecg_root}/{ecg_path}.dat` exists
  - CXR: `{cxr_root}/{cxr_path}` exists
- Report: counts of missing files, list of missing paths
- Output: `logs/dataset_layout_check.txt`
- Exit code 1 if any missing files

#### `scripts/convert_manifest_to_more_format.py`
- Input:
  - `matched_patients_v1.csv`
  - `data/mimic-cxr-jpg/mimic-cxr-2.0.0-chexpert.csv`
  - `data/mimic-cxr-jpg/mimic-cxr-2.0.0-split.csv`
  - `data/mimic-cxr-jpg/mimic-cxr-2.0.0-metadata.csv`
  - `data/mimic-iv-ecg/machine_measurements.csv`
  - Radiology report `.txt` files
- Operations:
  1. Join CheXpert labels via `subject_id` + `study_id`
  2. Join official split via `dicom_id`
  3. Build xray note from radiology report text (FINDINGS + IMPRESSION)
  4. Build ecg note from `machine_measurements.csv` `merged_report`
  5. Construct absolute file paths
  6. Produce item per row: `[xray_path, ecg_stem, xray_note, ecg_note, labels_array, split]`
  7. Save to `data/processed/xray_ecg_notes_labels.npy`
- Logging: report counts per split, missing labels, missing notes

#### `scripts/run_preprocessing_pipeline.sh`
- Wrapper that runs:
  1. `python scripts/verify_local_dataset_layout.py` — abort if fails
  2. `python scripts/convert_manifest_to_more_format.py`
- Logs each stage to `logs/preprocessing_{timestamp}.txt`
- Exits immediately on error (set -e)

### Required Minimal Refactors to MoRE
These are the minimum changes needed to run MoRE on our data without rewriting it:

1. **`utils/build_model.py`:** Make RoBERTa path configurable (read from `configs/paths.yaml`)
2. **`pretrain_multimodel.py`:** Add missing imports (`torch`, `tqdm`, `autocast`, `GradScaler`, `nn`), add `--epochs` argument
3. **`preprocessing/preprocess_notes.py`:** Fix early return bug in `get_clinical_xray()`

---

## Phase 4 — Smoke Test

### Smoke Test Strategy
- Use 50-100 samples from the manifest (first N rows, preferably same-day pairs)
- Verify:
  1. ECG loads correctly (wfdb.rdsamp returns valid shape)
  2. CXR loads and preprocesses to (3, 224, 224) tensor
  3. Notes tokenize to (512,) token IDs
  4. One forward pass through MultiModal model succeeds
  5. InfoNCE loss is computed without NaN

### Files to Create

#### `configs/smoke_test.yaml`
```yaml
data_path: data/processed/smoke_test.npy
batch_size: 4
num_workers: 0
epochs: 1
learning_rate: 3e-5
n_samples: 50  # rows from manifest to include
```

#### `scripts/build_smoke_test_data.py`
- Takes first N rows from manifest
- Runs through `convert_manifest_to_more_format.py` logic on just those rows
- Outputs `data/processed/smoke_test.npy`

#### `slurm/smoke_test.slurm`
- Short walltime (30 min), 1 GPU
- Runs smoke test forward pass check
- Logs to `logs/slurm_smoke_{j}.out`

#### `docs/smoke_test.md`
- Step-by-step instructions to run smoke test from scratch
- Expected outputs and what success looks like

---

## Phase 5 — Distillation-Ready Scaffolding

### Purpose
Lay the groundwork for the next development phase without implementing the full distillation loop.

### Files to Create

#### `docs/distillation_plan.md`
- Teacher: MoRE MultiModal (frozen, pre-trained)
- Student: MobileNetV3-Small adapted for 1-lead ECG input
- Loss functions:
  - Task loss: CrossEntropy on CVD labels
  - Soft logits: KL divergence (teacher logits vs student logits)
  - Embedding alignment: Cosine similarity (teacher ECG embedding vs student embedding)
  - Cross-modal similarity: Preserve teacher ECG-CXR similarity matrix
- Training data: same manifest subset, ECG only for student
- Evaluation: Lead ablation study (Lead I, II, V2)

#### `distill/` directory
```
distill/
├── __init__.py
├── student_model.py       # Placeholder: MobileNetV3-1D adapter
├── distill_dataset.py     # Dataset that returns (ecg, teacher_embedding, labels)
├── distill_train.py       # Main distillation training loop (scaffold)
└── configs/
    └── distill_config.yaml  # Teacher ckpt path, student config, loss weights
```

#### `configs/distill_config.yaml`
```yaml
teacher:
  checkpoint: outputs/teacher/best_multimodel.pth
  embedding_cache: outputs/teacher/ecg_embeddings_cache.npy

student:
  architecture: mobilenetv3_small
  input_leads: [1]   # lead index for single-lead mode
  num_classes: 4     # Atelectasis, Cardiomegaly, Edema, Pleural Effusion

loss_weights:
  alpha: 1.0    # task loss
  beta: 0.5     # soft logits (KL)
  gamma: 0.5    # embedding alignment (cosine)
  delta: 0.3    # cross-modal similarity matching
```

---

## Critical Path

```
Phase 0 (done)
    │
    ▼
Phase 1: Setup dirs + configs
    │
    ▼
Phase 2: Build download lists → submit Slurm download job
    │             (runs in background, ~hours)
    ▼
Phase 3: Once download complete → verify layout → convert to .npy
    │
    ▼
Phase 4: Smoke test → confirm forward pass
    │
    ▼
Phase 5: Distillation scaffold (can begin in parallel with Phase 4)
```

---

## Open Assumptions

See `docs/assumptions.md` for assumptions that may need validation:

1. **ECG sampling rate:** Assumed 500 Hz (5000 samples per 10s recording → downsample by 5 → 1000 samples at 100 Hz). Verify with `wfdb.rdheader()` on a downloaded record.
2. **No CheXpert labels in manifest:** We assume labels can be joined via `subject_id` + `cxr_study_id`. If a patient's study has no CheXpert label row, we use zeros (no finding).
3. **One CXR per subject:** The manifest has `rn=1` for all rows (first-ranked match). We use this single CXR per patient.
4. **Radiology report availability:** Not all CXR studies have associated `.txt` report files. Where missing, we use the label-derived synthetic note (same logic as MoRE's `generate_xray_note()`).
5. **ECG text report:** Constructed from `machine_measurements.csv` columns `report_0`–`report_6` joined by space. May be empty for some records.
6. **Split assignment:** We follow the official MIMIC-CXR split (`mimic-cxr-2.0.0-split.csv`) keyed on `dicom_id`. If a dicom_id is missing from the split file, the sample is assigned to 'train'.
7. **Day_diff direction:** The manifest stores absolute day difference. For same-day pairs (day_diff=0), the ECG and CXR are treated as fully aligned.
