#!/bin/bash
#SBATCH --job-name=more_smoke
#SBATCH --output=logs/smoke_%j.log
#SBATCH --error=logs/smoke_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00

# ============================================================
# slurm/smoke_test.slurm
#
# Runs the full MoRE smoke test:
#   1. Build 50-sample .npy subset from manifest
#   2. Run forward pass validation script
#
# Usage:
#   sbatch slurm/smoke_test.slurm
#
# Requires:
#   - conda env more_kd installed (run scripts/setup_env.sh first)
#   - Data downloaded (slurm/download_subset.slurm with --subset 500+)
#   - MIMIC metadata CSVs present
# ============================================================

set -euo pipefail

REPO_DIR="/scratch/user/dshekar/more-clinical-distill"
PYTHON_BIN="/scratch/user/dshekar/.conda/envs/more_kd/bin/python"
SMOKE_CFG="${REPO_DIR}/configs/smoke_test.yaml"
PATHS_CFG="${REPO_DIR}/configs/paths.yaml"

echo "================================================================"
echo "MoRE Smoke Test SLURM Job"
echo "================================================================"
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      ${SLURM_NODELIST}"
echo "GPU:       ${SLURM_GPUS_ON_NODE:-1}"
echo "Start:     $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "Repo dir:  ${REPO_DIR}"
echo "Python:    ${PYTHON_BIN}"
echo "================================================================"

mkdir -p "${REPO_DIR}/logs/smoke_test"
mkdir -p "${REPO_DIR}/outputs/smoke_test"

module load WebProxy 2>/dev/null || true
export http_proxy="${http_proxy:-http://10.73.132.63:8080}"
export https_proxy="${https_proxy:-http://10.73.132.63:8080}"

export HF_HOME="/scratch/user/dshekar/hf_home"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
mkdir -p "${TRANSFORMERS_CACHE}"

# Point MoRE at RoBERTa model (update if placed elsewhere)
export MORE_ROBERTA_PATH="${REPO_DIR}/data/RoBERTa-base-PM-M3/RoBERTa-base-PM-M3-hf"

cd "${REPO_DIR}"

# ── Step 1: build smoke-test .npy ────────────────────────────────────
echo ""
echo "Step 1/2: Building smoke-test dataset (50 subjects)..."
"${PYTHON_BIN}" scripts/build_smoke_test_data.py \
    --n 50 \
    --config "${PATHS_CFG}"

# ── Step 2: forward pass validation ──────────────────────────────────
echo ""
echo "Step 2/2: Running forward pass smoke test..."
"${PYTHON_BIN}" scripts/run_smoke_test.py \
    --config "${PATHS_CFG}" \
    --smoke-config "${SMOKE_CFG}"

echo ""
echo "================================================================"
echo "Smoke test finished: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "================================================================"
