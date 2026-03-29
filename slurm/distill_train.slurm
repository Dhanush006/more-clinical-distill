#!/bin/bash
#SBATCH --job-name=more_distill
#SBATCH --output=logs/distill_%j.log
#SBATCH --error=logs/distill_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

# ============================================================
# slurm/distill_train.slurm
#
# Runs the student distillation training loop.
#
# Prerequisites:
#   - Smoke test passed (slurm/smoke_test.slurm)
#   - Teacher weights at outputs/teacher/best_multimodel.pth
#   - Teacher embeddings cached at outputs/teacher/ecg_embeddings_train.npy
#     (run distill/cache_teacher_embeddings.py first)
#   - more_train.npy, more_val.npy built (Phase 3)
#
# Usage:
#   sbatch slurm/distill_train.slurm
# ============================================================

set -euo pipefail

REPO_DIR="/scratch/user/dshekar/more-clinical-distill"
PYTHON_BIN="/scratch/user/dshekar/.conda/envs/deepship_xares2/bin/python"
DISTILL_CFG="${REPO_DIR}/configs/distill_config.yaml"

echo "================================================================"
echo "MoRE Knowledge Distillation Training"
echo "================================================================"
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      ${SLURM_NODELIST}"
echo "GPU:       ${SLURM_GPUS_ON_NODE:-1}"
echo "Start:     $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "================================================================"

mkdir -p "${REPO_DIR}/logs/distill"
mkdir -p "${REPO_DIR}/outputs/distill"

module load WebProxy 2>/dev/null || true
export http_proxy="${http_proxy:-http://10.73.132.63:8080}"
export https_proxy="${https_proxy:-http://10.73.132.63:8080}"

export HF_HOME="/scratch/user/dshekar/hf_home"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export MORE_ROBERTA_PATH="${REPO_DIR}/data/RoBERTa-base-PM-M3/RoBERTa-base-PM-M3-hf"

cd "${REPO_DIR}"

echo ""
echo "Running distillation training..."
"${PYTHON_BIN}" distill/distill_train.py --config "${DISTILL_CFG}"

echo ""
echo "================================================================"
echo "Distillation finished: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "================================================================"
