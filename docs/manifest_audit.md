# Manifest Audit: matched_patients_v1.csv

**Date:** 2026-03-28
**Source file:** `/scratch/user/dshekar/matched_patients_v1.csv`
**Audited by:** Phase 0 inspection

---

## 1. Basic Statistics

| Metric | Value |
|--------|-------|
| Total rows | 49,076 |
| Unique subjects | 49,076 (1 row per subject) |
| Columns | 9 |
| Null / empty values | **None** (0 across all columns) |

---

## 2. Column Inventory

| Column | Type | Example Value | Description |
|--------|------|---------------|-------------|
| `subject_id` | int | `19128402` | MIMIC patient subject ID |
| `ecg_study_id` | int | `43658161` | MIMIC-IV-ECG study ID |
| `cxr_study_id` | int | `53334082` | MIMIC-CXR study ID |
| `ecg_path` | str | `files/p1912/p19128402/s43658161/43658161` | Relative WFDB stem path for ECG (no extension) |
| `cxr_path` | str | `files/p19/p19128402/s53334082/1633ff06-....jpg` | Relative path to CXR JPG image |
| `ecg_date` | str (date) | `2146-06-30` | Date of ECG recording |
| `cxr_date` | str (date) | `2146-06-30` | Date of CXR study |
| `day_diff` | int | `0` | Absolute days between ECG and CXR |
| `rn` | int | `1` | Row number (always 1 — first match per subject) |

---

## 3. ECG Path Analysis

**Format:** `files/p{4-digit-prefix}/{full_patient_id}/{study_dir}/{study_id_stem}`

```
files/p1912/p19128402/s43658161/43658161
  └─ 4-char prefix: p1912
  └─ full patient: p19128402
  └─ study dir: s43658161
  └─ stem: 43658161 (WFDB stem, no .hea/.dat extension)
```

**Download base URL:**
`https://physionet.org/files/mimic-iv-ecg/1.0/`

**Full WFDB path for download:**
For each row, need to download **two** files:
- `{ecg_path}.hea` → header file
- `{ecg_path}.dat` → signal data file

**Download URL pattern:**
```
https://physionet.org/files/mimic-iv-ecg/1.0/files/p{4digit}/p{subject}/s{study}/{study}.hea
https://physionet.org/files/mimic-iv-ecg/1.0/files/p{4digit}/p{subject}/s{study}/{study}.dat
```

**Number of unique ECG prefix dirs:** ~700+ (spreads across many `p{XXXX}` dirs)
**Total ECG files to download:** 49,076 × 2 = 98,152 files

---

## 4. CXR Path Analysis

**Format:** `files/p{2-digit-prefix}/{full_patient_id}/{study_dir}/{dicom_id}.jpg`

```
files/p19/p19128402/s53334082/1633ff06-c85a649a-dbc4ef77-1dfa76c8-5c2bd05f.jpg
  └─ 2-char prefix: p19
  └─ full patient: p19128402
  └─ study dir: s53334082
  └─ filename: {dicom_id}.jpg
```

**Download base URL:**
`https://physionet.org/files/mimic-cxr-jpg/2.0.0/`

**Full download URL pattern:**
```
https://physionet.org/files/mimic-cxr-jpg/2.0.0/files/p{2digit}/p{subject}/s{study}/{dicom_id}.jpg
```

**CXR prefix directories:** 10 unique (`p10` through `p19`)
**Total CXR images to download:** 49,076 files

---

## 5. Additional Files Needed from PhysioNet

Beyond the raw signal files, the MoRE preprocessing pipeline also needs:

### From MIMIC-CXR-JPG (mimic-cxr-jpg/2.0.0):
| File | Purpose |
|------|---------|
| `mimic-cxr-2.0.0-metadata.csv` | Image metadata (ViewPosition, StudyDate, dicom_id) |
| `mimic-cxr-2.0.0-split.csv` | Official train/validate/test splits |
| `mimic-cxr-2.0.0-chexpert.csv` | CheXpert labels (14 pathology labels) |
| Radiology report `.txt` files under `files/` | Xray notes for text encoder |

### From MIMIC-IV-ECG (mimic-iv-ecg/1.0):
| File | Purpose |
|------|---------|
| `machine_measurements.csv` | ECG machine-generated reports (report_0..6) |
| `record_list.csv` | Maps study_id → file path + ECG datetime |

---

## 6. day_diff Distribution

| day_diff | Count | % of total |
|----------|-------|-----------|
| 0 (same day) | 42,780 | 87.2% |
| 1 | 3,103 | 6.3% |
| 2 | 649 | 1.3% |
| 3 | 377 | 0.8% |
| 4 | 231 | 0.5% |
| 5-7 | 419 | 0.9% |
| 8-60 | 1,517 | 3.1% |

87% of pairs are same-day, which is excellent for multimodal alignment quality.

---

## 7. Path Transformation Required

The manifest paths are **relative** to their respective PhysioNet dataset roots.

### ECG: Manifest path → local storage path
```
Manifest: files/p1912/p19128402/s43658161/43658161
Local:    data/mimic-iv-ecg/files/p1912/p19128402/s43658161/43658161.hea
          data/mimic-iv-ecg/files/p1912/p19128402/s43658161/43658161.dat
```

### CXR: Manifest path → local storage path
```
Manifest: files/p19/p19128402/s53334082/1633ff06-....jpg
Local:    data/mimic-cxr-jpg/files/p19/p19128402/s53334082/1633ff06-....jpg
```

### MoRE compatibility: local path → MoRE expected path
MoRE's preprocessing script hardcodes the CXR root at `./mimic-cxr-jpg-2.0.0.physionet.org/`.
Options:
1. Create symlink: `./mimic-cxr-jpg-2.0.0.physionet.org` → `data/mimic-cxr-jpg/`
2. Refactor preprocessing script to accept a config-driven base path (preferred)

---

## 8. MoRE `.npy` Format Compatibility

The final preprocessed file (`xray_ecg_notes_labels_combined_60days.npy`) is a list of items:
```python
item = [
    xray_absolute_path,      # str: full path to .jpg
    ecg_wfdb_stem,           # str: full path WITHOUT extension (for wfdb.rdsamp)
    xray_note,               # str: "The report from Xray is: ..."
    ecg_note,                # str: "The report from ECG is: ..."
    labels_array,            # np.ndarray shape (13,): CheXpert labels
    split,                   # str: 'train', 'validate', or 'test'
]
```

**Our manifest provides:** subject_id, ecg/cxr paths, dates, day_diff
**Our manifest lacks:** CheXpert labels, radiology/ECG text reports, split assignment

These must be **joined** from MIMIC metadata files during preprocessing.

---

## 9. Validation Requirements

Before training, the following must hold for every manifest row:
1. `{data_root}/mimic-iv-ecg/{ecg_path}.hea` exists
2. `{data_root}/mimic-iv-ecg/{ecg_path}.dat` exists
3. `{data_root}/mimic-cxr-jpg/{cxr_path}` exists
4. ECG wfdb record is readable by `wfdb.rdsamp()`
5. ECG signal shape is `(T, 12)` with T ≥ 5000 (≥10s at 500Hz)
6. CXR image is a valid JPEG, size > 0

---

## 10. Summary of Conversion Tasks

| Task | Source | Output |
|------|--------|--------|
| Build ECG download list | manifest `ecg_path` | `manifests/ecg_files.txt` (`.hea` + `.dat` per line) |
| Build CXR download list | manifest `cxr_path` | `manifests/cxr_files.txt` |
| Download ECG metadata | PhysioNet MIMIC-IV-ECG | `data/mimic-iv-ecg/machine_measurements.csv`, `record_list.csv` |
| Download CXR metadata | PhysioNet MIMIC-CXR-JPG | `data/mimic-cxr-jpg/mimic-cxr-2.0.0-*.csv` |
| Download radiology reports | PhysioNet MIMIC-CXR | `data/mimic-cxr-jpg/files/**/s*.txt` |
| Join labels + notes | manifest + MIMIC metadata | intermediate merged CSV |
| Create `.npy` | merged CSV + text notes | `data/processed/xray_ecg_notes_labels.npy` |
| Verify local layout | downloaded files | pass/fail report |
