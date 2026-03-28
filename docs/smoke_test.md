# Smoke Test Guide

Validates the full MoRE pipeline end-to-end on 50 subjects before committing
to a full training run.

---

## Prerequisites

| Requirement | How to satisfy |
|---|---|
| `more_kd` conda env | `bash scripts/setup_env.sh` |
| 50+ subjects downloaded | `python scripts/build_download_lists.py --subset 500` then `sbatch slurm/download_subset.slurm` |
| MIMIC metadata CSVs | Download separately (see below) |
| RoBERTa-base-PM-M3 | Download from HuggingFace (see below) |

### Download MIMIC metadata CSVs

These are small files (~MB each), downloadable separately from the bulk ECG/CXR data:

```bash
source .env   # load PHYSIONET_USER / PHYSIONET_PASS

# CXR metadata
wget --user=$PHYSIONET_USER --password=$PHYSIONET_PASS -P data/mimic-cxr-jpg \
  https://physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-split.csv \
  https://physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-chexpert.csv \
  https://physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv

# ECG metadata
wget --user=$PHYSIONET_USER --password=$PHYSIONET_PASS -P data/mimic-iv-ecg \
  https://physionet.org/files/mimic-iv-ecg/1.0/machine_measurements.csv \
  https://physionet.org/files/mimic-iv-ecg/1.0/record_list.csv
```

### Download RoBERTa-base-PM-M3

```bash
mkdir -p data/RoBERTa-base-PM-M3
cd data/RoBERTa-base-PM-M3
# Download from HuggingFace Hub
python -c "
from transformers import AutoTokenizer, AutoModel
AutoTokenizer.from_pretrained('allenai/biomed_roberta_base').save_pretrained('RoBERTa-base-PM-M3-hf')
AutoModel.from_pretrained('allenai/biomed_roberta_base').save_pretrained('RoBERTa-base-PM-M3-hf')
"
# Or download the exact MoRE model from Google Drive (see Readme.md weights link)
# and unpack as data/RoBERTa-base-PM-M3/RoBERTa-base-PM-M3-hf/
```

---

## Running the Smoke Test

### Option A — SLURM (recommended)

```bash
sbatch slurm/smoke_test.slurm
# Monitor:
tail -f logs/smoke_<JOB_ID>.log
```

### Option B — Interactive (for debugging)

```bash
conda activate more_kd
export MORE_ROBERTA_PATH="data/RoBERTa-base-PM-M3/RoBERTa-base-PM-M3-hf"

# Build subset
python scripts/build_smoke_test_data.py --n 50

# Run forward pass checks
python scripts/run_smoke_test.py
```

---

## What the Smoke Test Checks

| # | Check | Expected output |
|---|---|---|
| 1 | `.npy` loads correctly | 50 items, each with 6 elements |
| 2 | ECG: `wfdb.rdsamp` → resample → tensor | shape `(12, 1000)`, no NaN |
| 3 | CXR: PIL load → CLAHE → tensor | shape `(3, 224, 224)` |
| 4 | Notes: tokenize via RoBERTa | shape `(1, 512)` |
| 5 | `MultiModal` forward pass | no exception, no shape mismatch |
| 6 | InfoNCE loss is finite | `loss < inf`, not NaN |

---

## Expected Success Output

```
================================================================
MoRE Smoke Test
================================================================
  [PASS] Load smoke .npy
         50 items loaded
  [PASS] Items have 6 elements
  [PASS] ECG: wfdb load + resample → (12,1000)
         shape=(12, 1000), dtype=torch.float32
  [PASS] CXR: PIL load + CLAHE → (3,224,224)
         shape=(3, 224, 224)
  [PASS] Notes: tokenize to (512,)
  [PASS] Forward pass + InfoNCE loss is finite
         device=cuda
         loss=2.7731
================================================================
ALL CHECKS PASSED — MoRE pipeline is smoke-test ready.
================================================================
```

---

## Common Failures

| Symptom | Likely cause | Fix |
|---|---|---|
| `FileNotFoundError` on `.hea` | ECG not downloaded | Re-run download with that subject in the list |
| `FileNotFoundError` on `.jpg` | CXR not downloaded | Re-run download |
| `ModuleNotFoundError: wfdb` | Wrong conda env | `conda activate more_kd` |
| `OSError: RoBERTa not found` | Model path wrong | Set `MORE_ROBERTA_PATH` env var |
| `NaN loss` | All-zero labels + no text | Check metadata CSV join; verify chexpert.csv present |
| CUDA OOM | Batch too large | Reduce `batch_size` in `configs/smoke_test.yaml` |
