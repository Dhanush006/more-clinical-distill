# more-clinical-distill — CLAUDE.md

This file documents the full project for context initialization. Read this before working on any task in this repo.

---

## What This Project Does

Distills a large pretrained multimodal teacher model (MoRE, ~280M params) into a lightweight ECG-only student model (MobileNetV3-Small, ~2.54M params) for cardiovascular disease classification. The student learns from a single ECG lead (Lead I by default) using knowledge distillation.

**Key result so far (baseline):** Student ECG macro AUC = 0.704 vs teacher probe ceiling = 0.933.

---

## Current Pipeline State (as of 2026-04-17)

| Step | Status | Notes |
|---|---|---|
| ECG download (49,076 studies) | ✅ DONE | `data/ecg_flat/` |
| CXR download (49,071 images) | ✅ DONE | `data/cxr_flat/` |
| Directory flattening | ✅ DONE | Freed ~200k inodes; was 451k, now ~260k/500k |
| Combined reports CSV integrated | ✅ DONE | Real ECG + CXR clinical text in `.npy` |
| `.npy` splits regenerated | ✅ DONE | Flat paths + real text; 48,423 / 379 / 274 |
| Teacher ECG embedding re-cache | 🔄 RUNNING | Job 18378665 (49,076 records, ~75 min) |
| Split redesign (patient-stratified) | ⏳ NEXT | After re-cache; Phase 2 of plan.md |
| Class imbalance handling | ⏳ NEXT | With split redesign |
| Ablation studies | ⏳ PENDING | Phase 7 of plan.md |

---

## Cluster Environment

- **Cluster:** TAMU HPRC Grace (SLURM)
- **Scratch:** `/scratch/user/dshekar/more-clinical-distill/`
- **Conda env:** `deepship_xares2` at `/scratch/user/dshekar/.conda/envs/deepship_xares2/`
- **GPU:** A100 40GB (partition: `gpu`, `medium`)
- **Proxy (required for wget):** `http://10.73.132.63:8080`
- **Inode quota:** 500k (currently ~260k used). Run `showquota` to check.
- **No Co-Authored-By:** Never add `Co-Authored-By: Claude` to git commits in this repo.

---

## Repository Layout

```
more-clinical-distill/
├── CLAUDE.md                   # This file — read first
├── plan.md                     # Full 9-phase forward execution plan with checkpoints
├── configs/
│   ├── paths.yaml              # All absolute data/output paths — edit here, not in code
│   └── distill_config.yaml     # Student training hyperparameters
├── data/
│   ├── ecg_flat/               # 49,076 × {study_id}.hea + {study_id}.dat (FLAT)
│   ├── cxr_flat/               # 49,071 × {dicom_id}.jpg (FLAT)
│   ├── mimic-iv-ecg/           # Metadata CSVs only (machine_measurements.csv, record_list.csv)
│   ├── mimic-cxr-jpg/          # Metadata CSVs only (chexpert, split, metadata)
│   ├── combined_reports.csv    # 49,076 rows: subject_id, ecg_study_id, cxr_study_id,
│   │                           #   ecg_reports, cxr_reports (0 missing ECG, 2,680 missing CXR)
│   └── processed/
│       ├── more_train.npy      # 48,423 rows — MoRE-format items, train split
│       ├── more_val.npy        # 379 rows (⚠ too small — redesign pending)
│       ├── more_test.npy       # 274 rows (⚠ too small — redesign pending)
│       ├── teacher_ecg_embeddings.npy   # (N, 128) float32 — re-cache running (job 18378665)
│       └── teacher_ecg_study_ids.npy    # (N,) str — study IDs for embedding alignment
├── distill/
│   ├── student_model.py        # MobileNetV3-Small dual-head student
│   ├── distill_dataset.py      # Dataset loader — reads .npy + ecg_flat/ + embeddings
│   ├── distill_train.py        # Training loop with NaN guard + gradient clipping
│   ├── evaluate_student.py     # Benchmark: student AUC vs teacher linear probe
│   └── cache_teacher_embeddings.py  # Pre-computes teacher ECG embeddings from ecg_flat/
├── scripts/
│   ├── convert_manifest_to_more_format.py  # Builds .npy splits — uses combined_reports.csv
│   │                                        # Auto-detects flat vs deep dirs from paths.yaml
│   ├── flatten_data.py         # One-time migration: deep wget tree → flat dirs (DONE)
│   ├── build_download_lists.py # Generates wget URL lists from manifest
│   └── download_metadata_csvs.sh
├── slurm/
│   ├── distill_train.slurm     # GPU training job (8h, A100)
│   ├── evaluate_student.slurm  # Evaluation job
│   ├── cache_embeddings.slurm  # Teacher embedding cache job (~75 min, GPU)
│   ├── download_ecg.slurm      # ECG download (DONE)
│   └── download_cxr.slurm      # CXR download (DONE)
├── manifests/
│   ├── ecg_download_list.txt   # 98,152 wget URLs for ECG
│   └── cxr_download_list.txt   # 49,076 wget URLs for CXR
├── outputs/
│   ├── more_pretrained.pth     # Frozen MoRE teacher weights (~1.2GB)
│   └── distill/
│       └── student_best_ep34_ecgauc0.7043.pth  # Best baseline checkpoint
├── logs/                       # SLURM .log / .err files
└── docs/
    ├── assumptions.md          # All validated assumptions (A1–A12)
    ├── distillation_plan.md    # Architecture + loss design detail
    ├── implementation_plan.md  # Phase-by-phase execution history
    └── progress_report.tex     # Course progress report (ECEN 766)
```

---

## Data: What Was Downloaded and How

### ECG Data (MIMIC-IV-ECG v1.0)

- **Source:** PhysioNet, authenticated wget (`PHYSIONET_USER` / `PHYSIONET_PASS` in `.env`)
- **Format:** WFDB — each study has two files:
  - `{study_id}.hea` — plain-text header (sampling rate, lead names, signal format, units)
  - `{study_id}.dat` — binary 16-bit ADC values for all 12 leads, interleaved sample-by-sample
- **12 leads (in order):** I, II, III, aVR, aVF, aVL, V1, V2, V3, V4, V5, V6
- **Duration:** 10 seconds at 500 Hz = **5,000 samples per lead**
- **Units:** millivolts (mV) after gain/offset from `.hea`
- **Loading:** `wfdb.rdrecord(stem)` returns `.p_signal` shape `(5000, 12)` in mV
- **Resampling:** 500 Hz → 100 Hz via `scipy.signal.resample` → shape `(1000, 12)`
- **Student input:** Lead I only (index 0) → shape `(1, 1000)` — configurable via `input_lead` in config
- **Location:** `data/ecg_flat/{study_id}.hea` and `data/ecg_flat/{study_id}.dat`
- **Count:** 49,076 studies

### CXR Data (MIMIC-CXR-JPG v2.1.0)

- **Format:** JPEG, one image per study (PA or AP view, ~1–3 MB each)
- **Location:** `data/cxr_flat/{dicom_id}.jpg`
- **Count:** 49,071 / 49,076 (5 absent on PhysioNet)

### Report Text (combined_reports.csv)

- **Location:** `data/combined_reports.csv`
- **Columns:** `subject_id, ecg_study_id, cxr_study_id, ecg_reports, cxr_reports`
- **Coverage:** 49,076 rows; 0 missing ECG, 2,680 missing CXR (fallback: CheXpert label-derived text)
- **Reusable:** This CSV + the flat dirs make all three modalities available for any future model (VLM, etc.)

---

## Data Format: .npy Item Structure

Each `.npy` file is a numpy object array; each row is a 7-element Python list:

```python
item = [
    xray_path,    # str: absolute path → data/cxr_flat/{dicom_id}.jpg
    ecg_stem,     # str: absolute stem  → data/ecg_flat/{study_id}  (no extension, for wfdb)
    xray_note,    # str: "The report from Xray is: {cxr_reports text}"
    ecg_note,     # str: "The report from ECG is: {ecg_reports text}"
    ecg_labels,   # np.ndarray (10,) float32 — ECG rhythm multi-label (0/1 per class)
    pulm_labels,  # np.ndarray (4,)  float32 — CheXpert pulmonary (1=present, 0=absent, -1=uncertain)
    split,        # str: 'train', 'validate', or 'test'
]
```

### ECG Rhythm Labels (10 classes)

| Index | Class | Count | % |
|---|---|---|---|
| 0 | Normal ECG | 11,053 | 22.5% |
| 1 | Sinus bradycardia | 4,456 | 9.1% |
| 2 | Sinus tachycardia | 6,720 | 13.7% |
| 3 | Atrial fibrillation | 3,978 | 8.1% |
| 4 | LBBB | 1,055 | 2.1% |
| 5 | RBBB | 3,345 | 6.8% |
| 6 | ST elevation MI | 480 | 1.0% ← rare |
| 7 | ST ischemia | 2,847 | 5.8% |
| 8 | AV block | 2,627 | 5.4% |
| 9 | LVH | 3,863 | 7.9% |

35% of records have no ECG label. ST elevation MI (480) and LBBB (1,055) are the rarest classes.

### CheXpert Pulmonary Labels (4 classes)
`[Atelectasis, Cardiomegaly, Edema, Pleural Effusion]`

---

## Teacher Model (MoRE)

- **Architecture:** Multimodal — ECG ViT-Base + CXR ViT-Base + RoBERTa-base-PM-M3 text, ~280M params
- **Pretrained with:** InfoNCE contrastive loss across ECG ↔ CXR ↔ Text modalities
- **Weights:** `outputs/more_pretrained.pth` (Google Drive, see original MoRE repo README)
- **Frozen always** — only used for embedding extraction, never trained
- **ECG input:** `(B, 12, 1000)` at 100 Hz → **128-dim** projected embedding via `projector_ecg_text`
- **Embeddings cached at:** `data/processed/teacher_ecg_embeddings.npy`

---

## Student Model

- **Architecture:** MobileNetV3-Small + channel adapter (1→16→3 Conv1d) + dual heads
- **Input:** `(B, 1, 1000)` — single ECG lead at 100 Hz
- **Embedding:** 128-dim (matches teacher projector dim)
- **ECG head:** Linear(128 → 10) — primary, ECG rhythm classification
- **Pulm head:** Linear(128 → 4) — secondary, pulmonary via distillation
- **Params:** ~2.54M (vs ~280M teacher)
- **File:** `distill/student_model.py`

---

## Training Loss

```
L = 1.0 × L_ecg_BCE  +  0.5 × L_pulm_BCE  +  0.5 × L_align
```

- `L_ecg_BCE`: BCE(student ECG logits, ECG rhythm labels) — supervised
- `L_pulm_BCE`: BCE(student pulm logits, CheXpert labels) — supervised via distillation
- `L_align`: `1 − cosine_sim(student_emb, teacher_ecg_emb)` — embedding alignment

Planned additions (Phase 6): KL divergence soft-label loss, cross-modal alignment loss.
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

Note: baseline used only 28,745/49,076 ECG records (old partial download). Re-training after full re-cache expected to improve.

---

## Pipeline Execution Order

```
1. [DONE] Download ECG + CXR data
   → data/ecg_flat/ (49,076), data/cxr_flat/ (49,071)

2. [DONE] Flatten directory tree
   python scripts/flatten_data.py
   → Freed ~200k inodes (451k → ~260k used)

3. [DONE] Integrate combined_reports.csv
   → data/combined_reports.csv (real clinical ECG + CXR text)

4. [DONE] Regenerate .npy splits with flat paths + real text
   python scripts/convert_manifest_to_more_format.py
   → data/processed/more_{train,val,test}.npy

5. [RUNNING] Re-cache teacher ECG embeddings on full dataset
   Job 18378665 — sbatch slurm/cache_embeddings.slurm
   → data/processed/teacher_ecg_embeddings.npy (49,076 × 128)

6. [NEXT] Redesign train/val/test splits
   → Patient-stratified, class-balanced (see plan.md Phase 2)
   → Fix: val=379 and test=274 are too small for reliable eval
   → Re-run convert_manifest after split redesign

7. [NEXT] Re-run training with full data + proper splits
   sbatch slurm/distill_train.slurm

8. [PENDING] Ablation studies (plan.md Phase 7)
   → Loss weights, lead selection, class balancing

9. [PENDING] Final evaluation + reproducibility (plan.md Phase 8–9)
```

---

## Split Redesign — What Needs to Happen (Phase 2)

Current problem:
- Val: only **379 rows** — AUC estimates are noisy
- Test: only **274 rows** — ST elevation MI has only 2–3 test samples
- Split was inherited from MIMIC-CXR official split (designed for CXR, not ECG rhythms)
- No explicit patient-level separation check

Plan:
- Split at `subject_id` level (patient-level, no leakage)
- 80% train / 10% val / 10% test (~39k / 4.9k / 4.9k)
- Stratified on multi-label ECG vector (preserve class proportions)
- Compute inverse-frequency class weights → `configs/ecg_class_weights.json`

---

## Key Files to Understand First

1. **`CLAUDE.md`** (this file) — project overview
2. **`plan.md`** — 9-phase forward plan with checkpoints
3. **`configs/paths.yaml`** — all configurable paths
4. **`configs/distill_config.yaml`** — all training hyperparameters
5. **`docs/assumptions.md`** — known issues A1–A12
6. **`docs/distillation_plan.md`** — architecture and loss design rationale

---

## Known Issues / Watch Out For

- **Inode quota:** 500k limit. Run `showquota` before creating many files. Currently ~260k used.
- **NaN embeddings:** 502/28,745 teacher ECG embeddings had NaN in the old cache. Will recheck after job 18378665 completes on full 49,076 records. Sanitized via `np.nan_to_num` in `distill_dataset.py`.
- **Split sizes:** Current val (379) and test (274) are too small for reliable per-class AUC. Redesign pending after re-cache completes.
- **Class imbalance:** ST elevation MI (480 records, 1%) and LBBB (1,055, 2.1%) are severely underrepresented. Class-weighted BCE planned for Phase 5.
- **Proxy required:** All `wget` from Grace needs `http_proxy=http://10.73.132.63:8080`.
- **tmux for long jobs:** Login-node downloads/scripts should run in tmux (`tmux new-session -d -s <name>`).
