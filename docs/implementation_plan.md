# Implementation Plan: more-clinical-distill

**Date created:** 2026-03-28
**Last updated:** 2026-03-30
**Branch:** more-KD
**Goal:** Manifest-driven multimodal pipeline → MoRE teacher ECG embedding cache → Knowledge distillation into single-lead ECG student

---

## Status Summary

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 0 — Inspect and Plan | ✅ COMPLETE | Repo audit, manifest audit, assumptions documented |
| Phase 1 — Environment Setup | ✅ COMPLETE | Conda env, configs, folder structure |
| Phase 2 — Download Pipeline | ✅ COMPLETE (partial data) | Hit HPRC inode quota; 28,745/49,076 ECG files obtained |
| Phase 3 — MoRE Compatibility | ✅ COMPLETE | .npy splits generated; ECG rhythm labels added |
| Phase 4 — Teacher Embedding Cache | ✅ COMPLETE | 28,745 × 128-dim float32 embeddings cached |
| Phase 5 — Distillation | 🔄 IN PROGRESS | Student model implemented; training submitted |

---

## Phase 0 — Inspect and Plan ✅ COMPLETE

**Deliverables:**
- [x] `docs/repo_audit.md` — full audit of MoRE repo structure, hardcoding issues, bugs
- [x] `docs/manifest_audit.md` — 49,076 matched pairs, column analysis, path formats
- [x] `docs/assumptions.md` — explicit assumptions with validation status
- [x] `docs/implementation_plan.md` — this file

**Key findings:**
- MoRE expects `.npy` items of the form `[xray_path, ecg_stem, xray_note, ecg_note, labels, split]`
- ECG files: WFDB format (`.hea` + `.dat`), loaded with `wfdb.rdsamp(stem)`
- CXR images: JPEG, loaded with PIL; X-ray notes: from radiology `.txt` files
- Text model: RoBERTa-base-PM-M3, hardcoded path requiring refactor
- Multiple hardcoded paths and missing imports in original scripts
- Manifest: 49,076 matched pairs, 0 nulls, all unique subjects, 87.2% same-day pairs

---

## Phase 1 — Environment and Reproducibility ✅ COMPLETE

**Deliverables:**
- [x] Folder structure: `data/`, `manifests/`, `scripts/`, `slurm/`, `configs/`, `logs/`, `outputs/`, `distill/`
- [x] `configs/paths.yaml` — all configurable data paths
- [x] `.env.example` — credential template (PHYSIONET_USER / PHYSIONET_PASS)
- [x] `.gitignore` — excludes `data/`, `outputs/`, `logs/`, `*.npy`, `*.pth`, `.env`
- [x] Conda env: `deepship_xares2` on Grace HPC (A100 40GB)

---

## Phase 2 — Manifest-Driven Download Pipeline ✅ COMPLETE (partial data)

**Scripts created:**
- `scripts/build_download_lists.py` — generates `manifests/ecg_files.txt` (98,152 paths) and `manifests/cxr_files.txt` (49,076 paths)
- `slurm/download_subset.slurm` — authenticated wget download job using `.env` credentials

**Outcome:**
- ECG metadata CSVs downloaded: `machine_measurements.csv`, `record_list.csv`
- CXR metadata CSVs downloaded: `mimic-cxr-2.0.0-chexpert.csv`, `mimic-cxr-2.0.0-split.csv`, `mimic-cxr-2.0.0-metadata.csv`
- **ECG signal files**: 28,745 / 49,076 studies downloaded (20,331 missing)
- **CXR image files**: 0 / 49,076 downloaded — inode quota exhausted
- **Radiology report .txt files**: not downloaded — inode quota exhausted

**HPRC Inode Quota Issue:**
The Grace cluster enforces a per-user inode limit (~1M inodes). The MIMIC-CXR-JPG directory tree alone spans ~200K+ files/directories. The ECG + CXR + metadata files together exceed quota. Download was terminated after ECG signal files were partially obtained. CXR JPEG files could not be stored.

**Mitigation:**
- For the distillation pipeline, the student uses ECG only — CXR files are not needed at inference time
- Teacher embedding caching (Phase 4) runs teacher's ECG encoder only — no CXR files needed
- The 28,745 available ECG records are sufficient for distillation training
- Preprocessing uses label-derived xray notes (CheXpert-label-to-text fallback built into MoRE) for all records

---

## Phase 3 — MoRE Compatibility Layer ✅ COMPLETE

**Scripts created:**
- `scripts/verify_local_dataset_layout.py` — checks file existence for all manifest rows; outputs `logs/layout_verification.txt`
- `scripts/convert_manifest_to_more_format.py` — full preprocessing pipeline

**Key changes to item format:**
Original MoRE 6-element item:
```python
[xray_path, ecg_stem, xray_note, ecg_note, pulm_labels(4,), split]
```

Our extended 7-element item:
```python
[xray_path, ecg_stem, xray_note, ecg_note, ecg_labels(10,), pulm_labels(4,), split]
```

**ECG Rhythm Label Parsing:**
Added regex-based parsing of `machine_measurements.csv` free-text report fields into 10-class multi-label binary vectors. Classes:
`Normal, Sinus bradycardia, Sinus tachycardia, Atrial fibrillation, LBBB, RBBB, ST elevation MI, ST ischemia, AV block, LVH`

**MoRE bug fixes applied:**
- `preprocess_notes.py`: Fixed early `return` inside loop (should be `break`) in `get_clinical_xray()`
- `pretrain_multimodel.py`: Added missing imports (`torch`, `tqdm`, `autocast`, `GradScaler`, `nn`), added `--epochs` argparse argument
- `utils/build_model.py`: Made RoBERTa model path configurable via `configs/paths.yaml`

**Output:**

| Split | Records |
|-------|---------|
| Train | 48,423 |
| Val | 379 |
| Test | 274 |
| **Total** | **49,076** |

Note: All records are in the `.npy` files (paths stored), but only 28,745 ECG records have downloadable signal files. Records with missing ECG files are filtered out during distillation dataset loading.

---

## Phase 4 — Teacher ECG Embedding Cache ✅ COMPLETE

**Script:** `distill/cache_teacher_embeddings.py`
**SLURM job:** `slurm/cache_teacher_embeddings.slurm`

**Process:**
- Loads frozen MoRE teacher weights (downloaded from Google Drive link in original MoRE README)
- Runs teacher's ECG encoder on all available ECG records
- Caches 128-dim projected embeddings to `data/processed/teacher_ecg_embeddings.npy`
- Also saves study ID index to `data/processed/teacher_ecg_study_ids.npy` for lookup

**Output:**
- `data/processed/teacher_ecg_embeddings.npy`: shape `(28,745, 128)`, float32
- `data/processed/teacher_ecg_study_ids.npy`: shape `(28,745,)`

**Issue discovered:**
502 out of 28,745 embeddings contained NaN values — caused by ViT encoder instability on some ECG records (likely signal quality issues). These are sanitized with `np.nan_to_num(..., nan=0.0)` at dataset load time.

---

## Phase 5 — Distillation 🔄 IN PROGRESS

**Architecture:** MobileNetV3-Small adapted for 1-D single-lead ECG input

```
Input: (B, 1, 1000)  — Lead I at 100 Hz, 10s
  └─ Conv1D channel adapter: 1→16→3
       └─ MobileNetV3-Small backbone (timm)
            └─ 128-dim embedding
                 ├─ ecg_classifier  → (B, 10)  primary head
                 └─ pulm_classifier → (B, 4)   secondary head
```

Total student parameters: ~2.54M (vs ~280M teacher)

**Files implemented:**
- `distill/student_model.py` — dual-head MobileNetV3-Small
- `distill/distill_dataset.py` — returns `(ecg_tensor, teacher_emb, ecg_labels, pulm_labels)`
- `distill/distill_train.py` — training loop with dual-head loss + NaN guard + gradient clipping
- `distill/evaluate_student.py` — student vs teacher linear probe benchmark
- `distill/cache_teacher_embeddings.py` — Phase 4 caching script
- `configs/distill_config.yaml` — all hyperparameters
- `slurm/distill_train.slurm` — SLURM job (GPU, 8h)
- `slurm/evaluate_student.slurm` — SLURM evaluation job

**Training loss:**
```
L = 1.0 × L_ecg_BCE  +  0.5 × L_pulm_BCE  +  0.5 × L_align
```
where `L_align = 1 − mean_cosine_similarity(student_emb, teacher_emb)`

**Early stopping:** on ECG head macro AUC (patience=15 epochs)

**Pending ablations:**
- [ ] Lead ablation: Lead I vs Lead II vs Lead V2
- [ ] Loss weight ablation: vary `alpha_pulm` and `gamma`
- [ ] Class-weighted BCE for rare ECG classes (ST elevation MI: 480 records, LBBB: 1,055)
- [ ] EfficientNetV2 backbone comparison

---

## Critical Path

```
Phase 0 (done)
    │
    ▼
Phase 1: Setup dirs + configs (done)
    │
    ▼
Phase 2: Download pipeline → partial ECG data (done, inode limited)
    │
    ▼
Phase 3: Verify layout → convert to .npy with ECG + pulm labels (done)
    │
    ▼
Phase 4: Cache teacher ECG embeddings (done, 28,745 × 128)
    │
    ▼
Phase 5: Student training → evaluation → ablations (in progress)
```
