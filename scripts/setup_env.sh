#!/usr/bin/env bash
# ============================================================
# scripts/setup_env.sh
# Creates a dedicated conda env "more_kd" for this project.
#
# deepship_xares2 is incompatible (torch 2.8, transformers 5.x,
# wfdb missing). MoRE requires torch 2.0 / transformers 4.35.
#
# Usage:
#   bash scripts/setup_env.sh
#
# After activation, verify with:
#   python -c "import torch, timm, wfdb, peft, transformers, info_nce"
# ============================================================

set -euo pipefail

ENV_NAME="more_kd"
PYTHON_VERSION="3.10"

echo "Creating conda env '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
conda create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}"

PYTHON_BIN="$(conda run -n ${ENV_NAME} which python)"
echo "Python: ${PYTHON_BIN}"

echo "Installing PyTorch 2.0 + CUDA 11.8..."
conda run -n "${ENV_NAME}" pip install \
    torch==2.0.0+cu118 \
    torchvision==0.15.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

echo "Installing MoRE requirements..."
conda run -n "${ENV_NAME}" pip install \
    timm==0.9.12 \
    transformers==4.35.2 \
    peft==0.8.2 \
    wfdb==3.2.0 \
    info_nce_pytorch==0.1.4 \
    scikit-learn==1.3.2 \
    opencv-python==4.8.1.78 \
    lmdb==1.4.1 \
    h5py==3.9.0 \
    scipy==1.11.4 \
    pandas==2.0.3 \
    numpy==1.24.4 \
    Pillow==10.0.1 \
    PyYAML==6.0.1 \
    tqdm==4.66.1

echo ""
echo "Done. Activate with:  conda activate ${ENV_NAME}"
echo "Verify with:"
echo "  python -c \"import torch, timm, wfdb, peft, transformers, info_nce\""
