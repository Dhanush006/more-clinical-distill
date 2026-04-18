# more-clinical-distill — CLAUDE.md

This file documents the full project for context initialization. Read this before working on any task in this repo.

---

## What This Project Does

Distills a large pretrained multimodal teacher model (MoRE, ~280M params) into a lightweight ECG-only student model (MobileNetV3-Small, ~2.54M params) for cardiovascular disease classification. The student learns from a single ECG lead (Lead I by default) using knowledge distillation.

**Key result so far:** Student ECG macro AUC = 0.704 vs teacher probe ceiling = 0.933.

---

## Cluster Environment

- **Cluster:** TAMU HPRC Grace (SLURM)
- **Scratch:** `/scratch/user/dshekar/more-clinical-distill/`
- **Conda env:** `deepship_xares2` at `/scratch/user/dshekar/.conda/envs/deepship_xares2/`
- **GPU:** A100 40GB (partition: `gpu`, `medium`)
- **Proxy (required for wget):** `http://10.73.132.63:8080`
- **Inode quota:** 500k (currently ~260k used). Run `showquota` to check.

---

## Repository Layout

```
more-clinical-distill/
├── configs/
│   ├── paths.yaml              # All absolute data/output paths — edit this, not code
│   └── distill_config.yaml     # Student training hyperparameters
├── data/
│   ├── ecg_flat/               # 49,076 × {study_id}.hea + {study_id}.dat  (FLAT, post-migration)
│   ├── cxr_flat/               # 49,071 × {dicom_id}.jpg                   (FLAT, post-migration)
│   ├── mimic-iv-ecg/           # Metadata CSVs only (machine_measurements.csv, record_list.csv)
│   ├── mimic-cxr-jpg/          # Metadata CSVs only (chexpert, split, metadata)
│   ├── combined_reports.csv    # 49,076 rows: subject_id, ecg_study_id, cxr_study_id, ecg_reports, cxr_reports
│   └── processed/
│       ├── more_train.npy      # MoRE-format items, train split (48,423 rows)
│       ├── more_val.npy        # val split (379 rows)
│       ├── more_test.npy       # test split (274 rows)
│       ├── teacher_ecg_embeddings.npy   # (28,745, 128) float32 — teacher ECG projector output
│       └── teacher_ecg_study_ids.npy    # (28,745,) str — study IDs for embedding alignment
├── distill/
│   ├── student_model.py        # MobileNetV3-Small dual-head student
│   ├── distill_dataset.py      # Dataset loader — reads .npy + ecg_flat/ + embeddings
│   ├── distill_train.py        # Training loop
│   ├── evaluate_student.py     # Benchmark: student AUC vs teacher linear probe
│   └── cache_teacher_embeddings.py  # Pre-computes teacher ECG embeddings from ecg_flat/
├── scripts/
│   ├── convert_manifest_to_more_format.py  # Builds .npy splits from manifest + reports
│   ├── flatten_data.py         # One-time migration: deep wget tree → flat dirs (DONE)
│   ├── build_download_lists.py # Generates wget URL lists
│   └── download_metadata_csvs.sh
├── slurm/
│   ├── distill_train.slurm     # GPU training job (8h)
│   ├── evaluate_student.slurm  # Evaluation job
│   ├── cache_embeddings.slurm  # Teacher embedding cache job
│   ├── download_ecg.slurm      # ECG download (DONE)
│   └── download_cxr.slurm      # CXR download (DONE)
├── manifests/
│   ├── ecg_download_list.txt   # 98,152 wget URLs for ECG
│   └── cxr_download_list.txt   # 49,076 wget URLs for CXR
├── outputs/
│   ├── more_pretrained.pth     # Frozen MoRE teacher weights
│   └── distill/
│       └── student_best_ep34_ecgauc0.7043.pth  # Best baseline checkpoint
├── logs/                       # SLURM .log / .err files
├── docs/
│   ├── assumptions.md          # All validated assumptions (A1–A12)
│   ├── distillation_plan.md    # Architecture + loss design
│   ├── implementation_plan.md  # Phase-by-phase execution log
│   └── progress_report.tex     # Course progress report (ECEN 766)
└── plan.md                     # Full forward-looking execution plan (Phases 1–9)
```

---

## Data: What Was Downloaded and How

### ECG Data (MIMIC-IV-ECG v1.0)

- **Source:** PhysioNet, authenticated wget (`PHYSIONET_USER` / `PHYSIONET_PASS` in `.env`)
- **Format:** WFDB — each study has two files:
  - `{study_id}.hea` — plain-text header: sampling rate, lead names, signal format, units
  - `{study_id}.dat` — binary 16-bit ADC values for all 12 leads, interleaved sample-by-sample
- **Contents:** 12 standard leads: I, II, III, aVR, aVF, aVL, V1–V6 (in that order)
- **Duration:** 10 seconds at 500 Hz = **5,000 samples per lead**
- **Units:** millivolts (mV), after applying gain/offset from `.hea`
- **Loading:** `wfdb.rdrecord(stem)` returns `.p_signal` shape `(5000, 12)` in mV
- **Resampling:** Pipeline resamples 500 Hz → 100 Hz using `scipy.signal.resample` → shape `(1000, 12)`
- **Student uses:** Lead I only (index 0, column 0 of the signal matrix) — shape `(1, 1000)`
- **Location after migration:** `data/ecg_flat/{study_id}.hea` and `data/ecg_flat/{study_id}.dat`
- **Count:** 49,076 studies downloaded

### CXR Data (MIMIC-CXR-JPG v2.1.0)

- **Format:** JPEG, one image per study (PA or AP view, ~1–3 MB each)
- **Location after migration:** `data/cxr_flat/{dicom_id}.jpg`
- **Count:** 49,071 / 49,076 downloaded (5 missing — absent on PhysioNet)

### Report Text (combined_reports.csv)

- **Location:** `data/combined_reports.csv`
- **Columns:** `subject_id, ecg_study_id, cxr_study_id, ecg_reports, cxr_reports`
- **Coverage:** 49,076 rows (full manifest); 0 missing ECG reports, 2,680 missing CXR reports
- **Missing CXR fallback:** label-derived text from CheXpert labels (`label_fallback()` in convert script)
- **How used:** `convert_manifest_to_more_format.py` reads this CSV to populate `ecg_note` and `xray_note` fields in `.npy` items

---

## Data Format: .npy Item Structure

Each `.npy` file (train/val/test) is a numpy object array where each row is a 7-element list:

```python
item = [
    xray_path,    # str: absolute path to .jpg (data/cxr_flat/{dicom_id}.jpg)
    ecg_stem,     # str: absolute stem for wfdb (data/ecg_flat/{study_id}) — no extension
    xray_note,    # str: "The report from Xray is: {cxr_reports text}"
    ecg_note,     # str: "The report from ECG is: {ecg_reports text}"
    ecg_labels,   # np.ndarray (10,) float32 — ECG rhythm multi-label vector
    pulm_labels,  # np.ndarray (4,)  float32 — CheXpert pulmonary labels
    split,        # str: 'train', 'validate', or 'test'
]
```

### ECG Rhythm Labels (10 classes, index order)

| Index | Class |
|---|---|
| 0 | Normal ECG |
| 1 | Sinus bradycardia |
| 2 | Sinus tachycardia |
| 3 | Atrial fibrillation |
| 4 | LBBB (Left bundle branch block) |
| 5 | RBBB (Right bundle branch block) |
| 6 | ST elevation MI |
| 7 | ST ischemia |
| 8 | AV block |
| 9 | LVH (Left ventricular hypertrophy) |

### CheXpert Pulmonary Labels (4 classes)

`[Atelectasis, Cardiomegaly, Edema, Pleural Effusion]` — values: 1.0 present, 0.0 absent, -1.0 uncertain

---

## Teacher Model (MoRE)

- **Architecture:** Multimodal (ECG ViT-Base + CXR ViT-Base + RoBERTa text), ~280M params
- **Weights:** `outputs/more_pretrained.pth` (downloaded from Google Drive, see original MoRE repo)
- **Used for:** Pre-computing 128-dim ECG embeddings cached in `data/processed/teacher_ecg_embeddings.npy`
- **Frozen:** always — never trained, only used for embedding extraction
- **ECG input shape:** `(B, 12, 1000)` at 100 Hz
- **ECG output:** 128-dim projected embedding via `projector_ecg_text`

---

## Student Model

- **Architecture:** MobileNetV3-Small with channel adapter + dual classification heads
- **Input:** `(B, 1, 1000)` — single lead ECG (Lead I by default) at 100 Hz
- **Embedding:** 128-dim (matches teacher projector output dimension)
- **ECG head:** Linear(128 → 10) — primary, ECG rhythm classification
- **Pulm head:** Linear(128 → 4) — secondary, pulmonary condition classification
- **Params:** ~2.54M
- **File:** `distill/student_model.py`

---

## Training Loss

```
L = 1.0 × L_ecg_BCE  +  0.5 × L_pulm_BCE  +  0.5 × L_align
```

- `L_ecg_BCE`: BCE between student ECG logits and ECG rhythm labels
- `L_pulm_BCE`: BCE between student pulm logits and CheXpert labels
- `L_align`: `1 − cosine_similarity(student_emb, teacher_ecg_emb)` — embedding alignment

Config: `configs/distill_config.yaml`

---

## Baseline Results (Epoch 34 checkpoint)

| Metric | Value |
|---|---|
| Student ECG macro AUC | 0.7043 |
| Teacher linear probe AUC | 0.9332 |
| Student Pulm macro AUC | 0.4735 |
| Teacher Pulm probe AUC | 0.6596 |
| Embedding cosine similarity | 0.5821 |

Best checkpoint: `outputs/distill/student_best_ep34_ecgauc0.7043.pth`

---

## Pipeline Execution Order

```
1. [DONE] Download ECG + CXR data
   → data/ecg_flat/, data/cxr_flat/

2. [DONE] Flatten directory tree (saves ~200k inodes)
   python scripts/flatten_data.py

3. [DONE] Cache teacher ECG embeddings
   sbatch slurm/cache_embeddings.slurm
   → data/processed/teacher_ecg_embeddings.npy

4. [DONE] Convert manifest → .npy splits
   python scripts/convert_manifest_to_more_format.py
   → data/processed/more_{train,val,test}.npy

5. [DONE] Train baseline student
   sbatch slurm/distill_train.slurm

6. [DONE] Evaluate baseline
   sbatch slurm/evaluate_student.slurm

7. [NEXT] Redesign splits (Phase 2 of plan.md)
   → better val/test size, patient-stratified, class-balanced

8. [NEXT] Re-run convert_manifest + cache_embeddings with new splits

9. [NEXT] Ablation studies (plan.md Phase 7)
```

---

## Key Files to Understand First

If starting fresh on a task:

1. **`plan.md`** — full 9-phase forward plan with checkpoints
2. **`configs/paths.yaml`** — all configurable paths (change here, not in code)
3. **`configs/distill_config.yaml`** — all training hyperparameters
4. **`docs/assumptions.md`** — known issues and validated assumptions (A1–A12)
5. **`docs/distillation_plan.md`** — architecture and loss design rationale

---

## Known Issues / Watch Out For

- **Inode quota:** 500k limit on `/scratch/user/dshekar`. Run `showquota` before creating many files.
- **NaN embeddings:** 502/28,745 teacher ECG embeddings contain NaN — sanitized with `np.nan_to_num` in `distill_dataset.py`. Will need to recheck after full re-cache.
- **Split imbalance:** Current official MIMIC-CXR split gives only 379 val / 274 test rows — too small for reliable evaluation. Phase 2 of plan.md redesigns this.
- **Flat paths in .npy:** The existing `more_{train,val,test}.npy` were generated with deep wget paths. After `flatten_data.py` ran, you must re-run `convert_manifest_to_more_format.py` to regenerate them with flat paths before any training.
- **Proxy required:** All `wget` commands need `http_proxy=http://10.73.132.63:8080` on Grace.
- **No Co-Authored-By:** Never add `Co-Authored-By: Claude` to git commits in this repo.
