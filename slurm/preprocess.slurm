#!/bin/bash
#SBATCH --job-name=more_preprocess
#SBATCH --output=logs/preprocess_%j.log
#SBATCH --error=logs/preprocess_%j.err
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=04:00:00

# ============================================================
# slurm/preprocess.slurm
#
# Runs the Phase 3 preprocessing pipeline (CPU-only).
# Builds more_train/val/test.npy from manifest + MIMIC metadata.
#
# Usage:
#   sbatch slurm/preprocess.slurm
#
#   # Smoke-test (first 500 subjects):
#   SUBSET=500 sbatch slurm/preprocess.slurm
# ============================================================

set -euo pipefail

REPO_DIR="/scratch/user/dshekar/more-clinical-distill"
PYTHON_BIN="/scratch/user/dshekar/.conda/envs/more_kd/bin/python"
SUBSET="${SUBSET:-}"

echo "================================================================"
echo "MoRE Preprocessing SLURM Job"
echo "================================================================"
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      ${SLURM_NODELIST}"
echo "Start:     $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "Repo dir:  ${REPO_DIR}"
echo "Python:    ${PYTHON_BIN}"
echo "Subset:    ${SUBSET:-all}"
echo "================================================================"

mkdir -p "${REPO_DIR}/logs"

module load WebProxy 2>/dev/null || true
export http_proxy="${http_proxy:-http://10.73.132.63:8080}"
export https_proxy="${https_proxy:-http://10.73.132.63:8080}"

cd "${REPO_DIR}"

export PYTHON_BIN="${PYTHON_BIN}"

if [ -n "${SUBSET}" ]; then
    bash scripts/run_preprocessing_pipeline.sh "${SUBSET}"
else
    bash scripts/run_preprocessing_pipeline.sh
fi

echo ""
echo "================================================================"
echo "Job finished: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "================================================================"
