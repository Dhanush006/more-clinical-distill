# Assumptions and Open Questions

**Date:** 2026-03-28
**Updated by:** Phase 0 inspection

This file records explicit assumptions made during implementation where the ground truth was not directly verifiable at the time. Each assumption should be validated before use in training.

---

## A1 — ECG Sampling Rate

**Assumption:** All MIMIC-IV-ECG records in our manifest are sampled at 500 Hz, yielding 5000 samples for a 10-second recording.

**Why it matters:** MoRE applies `resample_poly(x, up=1, down=5)` which assumes 500 Hz input to produce 1000-sample output at 100 Hz. If a record is at a different rate, the shape will be wrong.

**How to validate:** Run `wfdb.rdheader(stem)['fs']` on a sample of downloaded records. Expected: `fs == 500`.

**Status:** Unverified — validate after first batch download.

---

## A2 — CheXpert Labels Joinable via subject_id + cxr_study_id

**Assumption:** Every CXR in our manifest has a corresponding row in `mimic-cxr-2.0.0-chexpert.csv`, joinable on `subject_id` and `study_id` (= `cxr_study_id` in our manifest).

**Why it matters:** MoRE uses CheXpert labels (14 pathology labels) as the supervised signal for fine-tuning and as item metadata in the `.npy` file.

**Fallback:** If a study is missing from the chexpert file, use a zero-vector (= "No Finding") and log a warning.

**Status:** Unverified — validate during `convert_manifest_to_more_format.py`.

---

## A3 — Official MIMIC-CXR Split Covers Our Manifest

**Assumption:** The `mimic-cxr-2.0.0-split.csv` file contains entries for all `dicom_id` values in our manifest CXR paths.

**Why it matters:** MoRE filters `train_data = [item for item in data_new if item[5] == 'train']`. Samples with no split assignment would be silently dropped.

**Fallback:** Assign to 'train' if `dicom_id` not found in split file. Log count.

**Status:** Unverified — validate during conversion step.

---

## A4 — Radiology Report Text Availability

**Assumption:** Most CXR studies in our manifest have associated `.txt` radiology reports under the MIMIC-CXR `files/` tree. The `preprocess_notes.py` `get_clinical_xray()` function extracts FINDINGS + IMPRESSION sections.

**Fallback (already in MoRE code):** If no report text, fall back to synthetic note constructed from CheXpert labels via `generate_xray_note()`.

**Coverage estimate:** MIMIC-CXR has ~95% report coverage. A small fraction may be empty.

**Status:** Accept MoRE fallback logic. No change needed unless coverage is unexpectedly low.

---

## A5 — ECG Machine Report Availability

**Assumption:** `machine_measurements.csv` has rows for all `ecg_study_id` values in our manifest, and `report_0`–`report_6` are populated (at least partially).

**Fallback:** If ECG report is empty after join, use `"ECG note not available."` (same as MoRE default in `process_item()`).

**Status:** Unverified — validate during conversion step.

---

## A6 — One JPG per CXR Study (PA or AP view)

**Assumption:** Each row in our manifest corresponds to exactly one JPEG file (PA or AP frontal view). The `cxr_path` field already encodes the specific dicom_id, so there is no ambiguity.

**Why it matters:** MIMIC-CXR studies can have multiple views (PA, lateral). MoRE filters to PA/AP in `preprocess_data.py`. Our manifest has pre-selected a specific file per study.

**Status:** Confirmed from manifest inspection — `cxr_path` includes the full dicom_id filename.

---

## A7 — ECG WFDB Records Have 12 Leads

**Assumption:** All MIMIC-IV-ECG records are 12-lead ECGs. The dataset documentation states this, and MoRE hardcodes 12-lead processing throughout.

**Why it matters:** `ViTModelEcg` and `PatchEmbed` assume `(12, 1000)` input shape.

**Status:** Confirmed by MIMIC-IV-ECG documentation. Low risk.

---

## A8 — Local Data Root Structure

**Assumption:** Downloaded files will be stored under:
- ECG: `data/mimic-iv-ecg/files/...`
- CXR: `data/mimic-cxr-jpg/files/...`

And the absolute paths stored in the `.npy` file will use these roots.

**Action required:** `configs/paths.yaml` must set `ecg_root` and `cxr_root` to absolute paths on this HPC system. The `.npy` file must use these absolute paths.

**Status:** Will be handled by `configs/paths.yaml` in Phase 1.

---

## A9 — preprocess_notes.py Bug

**Known bug:** `preprocess_notes.py` line 75 references `clinical_xray` (output of `get_clinical_xray()`) outside the function, but `get_clinical_xray()` has an early `return` inside the for loop (line 49), meaning only the first `.txt` file is ever processed. This is almost certainly a bug — the `return` should be `break` or the return should be outside the loop.

**Action:** Fix in Phase 3 minimal refactor. Do not use the original `preprocess_notes.py` for text extraction — re-implement this step in `scripts/convert_manifest_to_more_format.py` with the correct logic.

**Status:** Known defect — will be fixed.

---

## A10 — pretrain_multimodel.py Missing Imports

**Known issue:** `pretrain_multimodel.py` is missing the following imports at the top of the file:
- `import torch`
- `from torch.cuda.amp import autocast, GradScaler`
- `import torch.nn as nn`
- `from tqdm import tqdm`
- `args.epochs` is referenced but `--epochs` is not added to argparse

**Action:** Fix in Phase 3 minimal refactor before smoke test.

**Status:** Known defect — will be fixed.
