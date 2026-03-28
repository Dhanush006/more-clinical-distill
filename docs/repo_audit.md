# MoRE Repository Audit

**Date:** 2026-03-28
**Branch:** more-KD
**Source:** https://github.com/svthapa/MoRE
**Audited by:** Phase 0 inspection

---

## 1. Top-Level File Structure

```
more-clinical-distill/
├── pretrain_multimodel.py        # Main pre-training entry point
├── multimodal_infer.py           # Fine-tuning / inference entry point
├── xray_retrieval.py             # X-ray retrieval script
├── zero_shot_xray_more.py        # Zero-shot X-ray classification
├── zero_shot_ecg_more.py         # Zero-shot ECG classification
├── requirements.txt              # Python dependencies
├── Readme.md                     # Project README
├── diagramMultimodal_final.png   # Architecture diagram
├── tnse_plot_example.ipynb       # t-SNE visualization notebook
├── xray_ecg_retrieval.ipynb      # Retrieval example notebook
├── preprocessing/
│   ├── preprocess_data.py        # Dataset merging / CSV preprocessing
│   └── preprocess_notes.py       # Text notes preprocessing + .npy file creation
├── utils/
│   ├── build_model.py            # All model architecture definitions
│   ├── create_dataset.py         # PyTorch Dataset and Dataloader classes
│   ├── ecg_augmentations.py      # ECG data augmentation utilities
│   ├── get_xray_mean_std.py      # X-ray normalization stats helper
│   ├── metrics.py                # Training metrics (EarlyStopping, etc.)
│   └── scheduler.py              # LR scheduler utilities
└── Transformer-Explainability/   # Submodule: LRP-based ViT explainability
    └── ...
```

No existing `data/`, `scripts/`, `slurm/`, `configs/`, `logs/`, or `outputs/` directories — these must be created.

---

## 2. Entry Points

### Pre-training
```bash
python pretrain_multimodel.py --data_path ../data/path_to_data.npy [--batch_size N ...]
```
- Loads a `.npy` file via `np.load(filepath, allow_pickle=True)`
- Feeds it to `MultiModalData` dataset class
- Trains `MultiModal` model with InfoNCE contrastive loss

### Fine-tuning / Inference
```bash
python multimodal_infer.py
```
- Uses `MultiModalHead` model (frozen pretrained encoders + classification head)
- Paths must be manually edited in the script

### Zero-shot
```bash
python zero_shot_xray_more.py
python zero_shot_ecg_more.py
```

---

## 3. Data Pipeline (Preprocessing Flow)

```
Raw MIMIC sources
       │
       ▼
preprocessing/preprocess_data.py
  - Reads mimic-cxr-jpg-2.0.0.physionet.org/mimic-cxr-2.0.0-metadata.csv
  - Reads mimic-cxr-jpg-2.0.0.physionet.org/mimic-cxr-2.0.0-split.csv
  - Reads mimic-cxr-jpg-2.0.0.physionet.org/mimic-cxr-2.0.0-chexpert.csv
  - Reads physionet.org/files/mimic-iv-ecg/1.0/machine_measurements.csv
  - Reads physionet.org/files/mimic-iv-ecg/1.0/record_list.csv
  - Merges by subject_id, filters to ≤60 day window
  - Outputs: data/cxr_ecg_merged_labels_60days.csv
       │
       ▼
preprocessing/preprocess_notes.py
  - Reads data/cxr_ecg_merged_labels_60days.csv
  - Reads data/notes_xray_path.json (maps CXR path -> radiology report text)
  - Builds xray note + ecg note for each sample
  - Outputs: data/xray_ecg_notes_labels_combined_60days.npy
       │
       ▼
pretrain_multimodel.py
  - Loads data/xray_ecg_notes_labels_combined_60days.npy
  - Each item: [xray_path, ecg_path_stem, xray_note, ecg_note, labels_array, split]
```

---

## 4. Expected Folder Layout (Hardcoded in MoRE)

MoRE preprocessing scripts hardcode the following relative paths (relative to repo root):

| Path | Purpose |
|------|---------|
| `./mimic-cxr-jpg-2.0.0.physionet.org/mimic-cxr-2.0.0-metadata.csv` | CXR metadata |
| `./mimic-cxr-jpg-2.0.0.physionet.org/mimic-cxr-2.0.0-split.csv` | Train/val/test splits |
| `./mimic-cxr-jpg-2.0.0.physionet.org/mimic-cxr-2.0.0-chexpert.csv` | CheXpert labels |
| `./physionet.org/files/mimic-iv-ecg/1.0/machine_measurements.csv` | ECG machine reports |
| `./physionet.org/files/mimic-iv-ecg/1.0/record_list.csv` | ECG file listing |
| `../data/cxr_paths.txt` | List of local CXR file paths (one per line) |
| `../data/notes_xray_path.json` | CXR path → radiology note mapping |
| `./data/cxr_ecg_merged_labels_60days.csv` | Intermediate merged dataset |
| `./data/xray_ecg_notes_labels_combined_60days.npy` | Final preprocessed dataset (numpy) |
| `./data/RoBERTa-base-PM-M3/RoBERTa-base-PM-M3-hf/` | ClinicalBERT model weights |

**Action required:** All paths need to be made configurable. Minimal refactor: read from a config YAML.

---

## 5. Dataset Classes (utils/create_dataset.py)

| Class | Input | Description |
|-------|-------|-------------|
| `MultiModalData` | list of `[xray_path, ecg_stem, xray_note, ecg_note]` | Main pretraining dataset |
| `MultiModalDataset` | list of `[xray_path, ecg_stem, label, ...]` | Used for supervised evaluation |
| `MultiModalLMDBDataset` | LMDB file path | Alternative fast-load format |
| `H5Dataset` | HDF5 file path | Another alternative fast-load format |
| `EcgNotesDataset` | list of `[ecg_stem, note]` | ECG+text only |
| `XrayDataset` | list of `[xray_path, label]` | X-ray only |

The primary class for pretraining is **`MultiModalData`**, which expects items of the form:
```python
item = [xray_absolute_path, ecg_wfdb_stem, xray_note_text, ecg_note_text]
```

---

## 6. Model Architecture (utils/build_model.py)

| Class | Description |
|-------|-------------|
| `ViTModelEcg` | ViT-Base with custom 1D conv patch embedding for 12-lead ECG (12×1000 → 196 patches of dim 768) |
| `ViTModelXray` | ViT-Base with standard 2D patch embedding for 224×224 grayscale X-ray |
| `PatchEmbed` | Custom Conv1D patcher: 12-ch ECG → 196×768 patch tokens |
| `ProjectionHead` | 2-layer MLP projection: 768 → 512 or 128 |
| `MultiModal` | Full teacher model: ViTEcg + ViTXray + RoBERTa-base-PM-M3 with LoRA |
| `MultiModalHead` | Frozen encoders + classification head for fine-tuning |

**ECG input shape:** `(batch, 12, 1000)` — 12 leads, 1000 timepoints at 100 Hz
**X-ray input shape:** `(batch, 3, 224, 224)` — grayscale repeated 3× for ViT

---

## 7. ECG Preprocessing Pipeline

1. Load via `wfdb.rdsamp(stem_path)` → `(T, 12)` array (original 500 Hz)
2. Remove NaN → set to 0
3. `resample_poly(x, up=1, down=5)` → downsample to 100 Hz
4. Transpose: `(T, 12)` → `(12, T)`
5. Baseline wander removal (median filter)
6. Per-lead min-max normalization to `[-1, 1]`
7. Convert to `torch.FloatTensor`, shape `(12, 1000)`

---

## 8. X-ray Preprocessing Pipeline

1. Load via `PIL.Image.open()`, convert to grayscale `'L'`
2. Resize to `224×224`
3. CLAHE equalization (`exposure.equalize_adapthist`)
4. Scale to uint8, convert back to PIL
5. Apply transforms (augmentation or val), normalize with `mean=[0.499], std=[0.293]`
6. Repeat grayscale channel 3× → `(3, 224, 224)`

---

## 9. Text Model

- **Model:** `RoBERTa-base-PM-M3` (PubMed + MIMIC-III fine-tuned RoBERTa)
- **Source:** Must be downloaded to `./data/RoBERTa-base-PM-M3/RoBERTa-base-PM-M3-hf/`
- **Fine-tuning:** LoRA (r=16, alpha=8, dropout=0.1) — only 0.6% of parameters trained
- **Tokenizer:** `RobertaTokenizerFast` with max_length=512
- **Input format:** Combined xray_note + ecg_note, prefixed:
  - `"The report from Xray is: {xray_note}"`
  - `"The report from ECG is: {ecg_note}"`

---

## 10. Known Hardcoding Issues

1. **`preprocessing/preprocess_data.py`:** All data source paths are hardcoded relative strings
2. **`preprocessing/preprocess_notes.py`:** Reads from `./data/cxr_ecg_merged_labels_60days.csv` without arguments; also has a bug — `clinical_xray` is referenced before assignment (line 75) because `get_clinical_xray()` returns early
3. **`utils/build_model.py`:** Text model path `./data/RoBERTa-base-PM-M3/RoBERTa-base-PM-M3-hf` is hardcoded in `MultiModal.__init__`
4. **`pretrain_multimodel.py`:** Missing imports (`torch`, `tqdm`, `autocast`, `GradScaler`, `nn`) at the top of the file — the script as shipped has incomplete imports
5. **`pretrain_multimodel.py`:** References `args.epochs` but the argparse does not define `--epochs`

---

## 11. Pretrained Model Weights

Google Drive link: `https://drive.google.com/file/d/1BB9dT6iYihqJarD5qX0bdnfYhiwhBgmH/`
Must be downloaded manually; no automated download script exists in the repo.

---

## 12. Key Dependencies

| Package | Version | Notes |
|---------|---------|-------|
| torch | 2.0.0+cu118 | CUDA 11.8 |
| timm | 0.9.12 | ViT architectures |
| transformers | 4.35.0.dev0 | HuggingFace |
| peft | 0.8.2 | LoRA |
| wfdb | 3.2.0 | WFDB ECG format reading |
| info_nce_pytorch | 0.1.4 | InfoNCE loss |
| lmdb | 1.4.1 | Fast data store |
| h5py | 3.8.0 | HDF5 data store |
| scipy, numpy, PIL, skimage | — | Signal and image processing |

---

## 13. Summary of Required Actions

| Priority | Action |
|----------|--------|
| High | Create `data/`, `scripts/`, `slurm/`, `configs/`, `logs/`, `outputs/`, `manifests/` dirs |
| High | Build manifest-driven selective downloader (replaces full-dataset download assumption) |
| High | Refactor hardcoded paths into config file |
| High | Fix `preprocess_notes.py` bug (early return in `get_clinical_xray`) |
| High | Fix `pretrain_multimodel.py` missing imports and missing `--epochs` arg |
| Medium | Download RoBERTa-base-PM-M3 model weights |
| Medium | Create MoRE-compatible `.npy` file from our manifest subset |
| Medium | Verify ECG `.hea`+`.dat` pairs exist for all manifest entries |
| Low | Download pretrained MoRE teacher checkpoint from Google Drive |
