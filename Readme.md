# more-clinical-distill

**Branch:** `more-KD`
**Goal:** Multimodal clinical pipeline for heart disease classification via knowledge distillation — multimodal MoRE teacher (ECG + CXR + Text) → lightweight single-lead ECG student (MobileNetV3-Small).
**Data:** MIMIC-IV-ECG + MIMIC-CXR-JPG, 49,076 matched patient records.
**HPC:** Texas A&M Grace cluster (A100 40GB), SLURM job scheduler.

---

## HPC Setup Flow

### 1. Clone and checkout branch
```bash
cd /scratch/user/dshekar
git clone <repo_url> more-clinical-distill
cd more-clinical-distill
git checkout more-KD
```

### 2. Set up credentials (never committed)
```bash
cp .env.example .env
# Edit .env and fill in PHYSIONET_USER and PHYSIONET_PASS
```

### 3. Conda environment
Use existing env `deepship_xares2`, or create a fresh one if dependency conflicts arise:
```bash
# Check compatibility first:
conda run -n deepship_xares2 python -c "import torch, timm, wfdb, peft, transformers"

# If needed, create a new env:
conda create -n more_kd python=3.10
conda activate more_kd
pip install -r requirements.txt
```

### 4. Review and edit configs/paths.yaml
All data paths are defined in `configs/paths.yaml`. Edit to match your local layout before running any scripts.

### 5. Data download (Phase 2)
```bash
# Build download lists from manifest
python scripts/build_download_lists.py

# Submit SLURM download job (reads .env for credentials)
sbatch slurm/download_subset.slurm
```

### 6. Preprocess to MoRE format (Phase 3)
```bash
# Verify downloaded layout
python scripts/verify_local_dataset_layout.py

# Convert manifest + metadata to MoRE .npy format
python scripts/convert_manifest_to_more_format.py
```

### 7. Smoke test pretraining (Phase 4)
```bash
sbatch slurm/smoke_test.slurm
```

### 8. Cache teacher ECG embeddings (Phase 5, prerequisite)
```bash
# Pre-compute 128-dim ECG embeddings from frozen MoRE teacher
sbatch slurm/cache_teacher_embeddings.slurm
# Outputs: data/processed/teacher_ecg_embeddings.npy  (N×128 float32)
#          data/processed/teacher_ecg_study_ids.npy   (N,)
```

### 9. Distillation training (Phase 5)
```bash
sbatch slurm/distill_train.slurm
# Best checkpoint auto-saved to outputs/distill/student_best_ep{N}_ecgauc{X}.pth
```

### 10. Evaluate student vs teacher probe (Phase 5)
```bash
# Auto-selects best checkpoint; runs on both val and test splits
sbatch slurm/evaluate_student.slurm

# Or specify a checkpoint explicitly:
CKPT=outputs/distill/student_best_ep34_ecgauc0.7043.pth \
  sbatch slurm/evaluate_student.slurm
```

---

## Dual-Head Student Architecture

The student is a MobileNetV3-Small backbone adapted for 1-D single-lead ECG input:

```
Input: (B, 1, 1000)  — Lead I at 100 Hz, 10 s
  └─ MobileNetV3-Small backbone (modified for 1-D)
       └─ 128-dim embedding
            ├─ ecg_classifier  → (B, 10)  ECG rhythm logits    [primary head]
            └─ pulm_classifier → (B, 4)   pulmonary logits     [secondary head]
```

**Training loss:**
```
L = 1.0 × L_ecg_BCE  +  0.5 × L_pulm_BCE  +  0.5 × L_align
```
where `L_align = 1 - mean_cosine_similarity(student_emb, teacher_emb)`.

**ECG rhythm classes (10):** Normal, Sinus bradycardia, Sinus tachycardia, Atrial fibrillation, LBBB, RBBB, ST elevation MI, ST ischemia, AV block, LVH

Labels parsed from `machine_measurements.csv` free-text report fields using regex matching.

**Pulmonary classes (4):** Atelectasis, Cardiomegaly, Edema, Pleural Effusion
Labels from CheXpert annotations, learned indirectly via embedding alignment with the multimodal teacher.

---

## Baseline Results (Run 1 — epoch 34 best checkpoint)

Evaluated on val set. Teacher probe = `MultiOutputClassifier(LogisticRegression)` fit on teacher embeddings from train split — represents the ceiling achievable from teacher embeddings alone.

### ECG Rhythm Head (primary)

| Class | Student AUC | Teacher probe AUC |
|---|---|---|
| Normal | 0.632 | — |
| Sinus bradycardia | 0.932 | — |
| Sinus tachycardia | 0.784 | — |
| Atrial fibrillation | 0.523 | — |
| LBBB | 0.984 | — |
| RBBB | 0.614 | — |
| ST elevation MI | 0.614 | — |
| ST ischemia | 0.704 | — |
| AV block | 0.569 | — |
| LVH | 0.567 | — |
| **MACRO** | **0.7043** | **0.9332** |

### Pulmonary Head (secondary, via distillation)

| Class | Student AUC | Teacher probe AUC |
|---|---|---|
| Atelectasis | 0.484 | 0.6283 |
| Cardiomegaly | 0.463 | 0.6371 |
| Edema | 0.485 | 0.7212 |
| Pleural Effusion | 0.463 | 0.6518 |
| **MACRO** | **0.4735** | **0.6596** |

**Embedding cosine similarity (student vs teacher): 0.5821**

Notes:
- ECG head macro AUC of 0.70 is a strong first-run result given single-lead input only.
- Pulm head near-random (0.47) on run 1 — expected, as the student sees no CXR. Planned ablations: increase `gamma` (alignment weight), try higher-capacity backbone.
- Teacher probe ECG AUC (0.93) reflects full multimodal context (ECG + CXR + text); the relevant ECG-only baseline comparison is TBD.

---

## Folder Structure

```
more-clinical-distill/
├── configs/
│   ├── paths.yaml               # data paths config
│   ├── smoke_test.yaml          # teacher smoke-test config
│   └── distill_config.yaml      # student distillation config
├── data/                        # gitignored — raw downloads and processed arrays
│   ├── mimic-iv-ecg/
│   ├── mimic-cxr-jpg/
│   └── processed/               # MoRE-format .npy files + teacher embedding cache
├── distill/
│   ├── student_model.py         # MobileNetV3-Small dual-head student
│   ├── distill_dataset.py       # Dataset: ECG signal + teacher emb + labels
│   ├── distill_train.py         # Training loop with dual-head loss
│   ├── evaluate_student.py      # Benchmark: student vs teacher linear probe
│   └── cache_teacher_embeddings.py  # Pre-compute teacher ECG embeddings
├── docs/                        # Audit docs and implementation plans
├── logs/                        # SLURM logs (gitignored)
├── manifests/                   # Download lists (gitignored)
├── outputs/                     # Model checkpoints (gitignored)
├── preprocessing/               # Original MoRE preprocessing scripts
├── scripts/
│   ├── convert_manifest_to_more_format.py  # Build train/val/test .npy splits
│   └── ...
├── slurm/
│   ├── distill_train.slurm
│   ├── evaluate_student.slurm
│   ├── cache_teacher_embeddings.slurm
│   └── ...
├── utils/                       # Original MoRE utilities
├── .env.example                 # Credential template (copy to .env, never commit)
├── pretrain_multimodel.py
└── requirements.txt
```

---

## Original MoRE README

### Please Cite this work as:
@article{thapa2024more,
  title={MoRE: Multi-Modal Contrastive Pre-training with Transformers on X-Rays, ECGs, and Diagnostic Report},
  author={Thapa, Samrajya and Howlader, Koushik and Bhattacharjee, Subhankar and others},
  journal={arXiv preprint arXiv:2410.16239},
  year={2024}
}

# MoRE: MultiModal Contrastive Pretraining of X-ray, ECG, and Report

![MoRE Framework](./diagramMultimodal_final.png)

MoRE is a pretraining framework which synergestically aligns Xray, ECG, and Diagnostic Report of same patient with Contrastive Learning. The Clinical Report (Cardiology Report and Radiology Report) are combined together and acts an anchor to align the Xray and ECG in a multimodal space, we show this via Multi-Modal Retrieval by retrieving Xray and ECG data via a single text query (refer to Section 4.6.3 MultiModal Retrieval in Paper), we also adapt TransLRP to show multimodal attention visualization to provide explanation of multimodal input for diagnosis (refer to section 4.6.3 Gradient Based LRP attention visualization). MoRE beats baseline GLoRIA, MedKLIP in Mimic IV Xray dataset on 4 labels (Atelectasis, Cardiomegaly, Edema, Effusion) and beats baselines in PtbXL ECG dataset for superclass labels. MoRE outperforms its baselines in Zero-shot classification as well showcasing its strong representation learning capability. MoRE also utilizes PEFT LoRA strategy to fine-tune the LLM during pre-training effectively only training 0.6% of original parameters of the LLM significantly reducing training time. 

## Setting up the Environment

1. **Create a virtual environment**:
   ```bash
   python -m venv myenv
   ```

2. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Pre-Train MoRE

1. **Download the required datasets** from Physionet (datasets are not attached due to credential requirements for data signing).
   
2. **Add the datasets** to the appropriate folder.

3. **Preprocess the data** (preprocessing code is included).

4. **Run the pretraining script**:
   ```bash
   python pretrain_multimodel.py
   ```
   Add arguments as needed; default settings are provided.

## Fine-tune in Mimic/Chexpert

1. **Ensure that the pre-trained model is saved**.

2. **Run the fine-tuning script**:
   ```bash
   python multimodal_infer.py
   ```
   Make sure to change the data paths and model paths as needed.

## Zero-Shot Classification

1. **Run the zero-shot classification script**:
   ```bash
   python zero_shot_xray/ecg_more.py
   ```
   Update data paths or parameters as necessary.

## Retrieval Tasks

1. **Check the `xray_ecg_retrieval.ipynb` notebook** for an example of multimodal retrieval.

2. **Run the X-ray retrieval script**:
   ```bash
   python xray_retrieval.py
   ```

## t-SNE Plot

1. **Check the `tnse_plot.ipynb` notebook** for an example of a t-SNE plot of features.

## Model Weights

Link: https://drive.google.com/file/d/1BB9dT6iYihqJarD5qX0bdnfYhiwhBgmH/view?usp=share_link 
Change layer names, drop any weights if extra as needed through pytorch 
