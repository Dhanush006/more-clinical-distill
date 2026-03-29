#!/usr/bin/env bash
#SBATCH --job-name=more_cache_emb
#SBATCH --output=/scratch/user/dshekar/more-clinical-distill/logs/cache_embeddings_%j.log
#SBATCH --error=/scratch/user/dshekar/more-clinical-distill/logs/cache_embeddings_%j.log
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

# ============================================================
# slurm/cache_embeddings.slurm
#
# Run teacher ECG encoder on all downloaded ECG studies and
# cache 128-dim embeddings to data/processed/.
#
# Usage:
#   sbatch slurm/cache_embeddings.slurm
#
#   # Smoke-test with 500 studies:
#   SUBSET=500 sbatch slurm/cache_embeddings.slurm
# ============================================================

set -euo pipefail

REPO_DIR="/scratch/user/dshekar/more-clinical-distill"
SUBSET="${SUBSET:-0}"

# Prefer more_kd env, fall back to deepship_xares2 (has torch+wfdb+timm)
PYTHON_BIN="/scratch/user/dshekar/.conda/envs/deepship_xares2/bin/python"
if [ -f "/scratch/user/dshekar/.conda/envs/more_kd/bin/python" ]; then
    PYTHON_BIN="/scratch/user/dshekar/.conda/envs/more_kd/bin/python"
fi

echo "================================================================"
echo "MoRE Teacher Embedding Cache Job"
echo "================================================================"
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      ${SLURM_NODELIST}"
echo "Start:     $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "Python:    ${PYTHON_BIN}"
echo "Subset:    ${SUBSET:-all}"
echo "================================================================"

cd "${REPO_DIR}"

module load WebProxy 2>/dev/null || true

SUBSET_ARG=""
if [ "${SUBSET}" -gt 0 ] 2>/dev/null; then
    SUBSET_ARG="--subset ${SUBSET}"
fi

"${PYTHON_BIN}" distill/cache_teacher_embeddings.py \
    --config configs/distill_config.yaml \
    --batch_size 128 \
    ${SUBSET_ARG}

echo ""
echo "================================================================"
echo "Embedding cache complete: $(date '+%Y-%m-%d %H:%M:%S %Z')"
ls -lh data/processed/teacher_ecg_embeddings.npy 2>/dev/null || true
echo "================================================================"
