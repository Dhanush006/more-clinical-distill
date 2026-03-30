#!/bin/bash
#SBATCH --job-name=more_eval
#SBATCH --output=logs/eval_%j.log
#SBATCH --error=logs/eval_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

# ============================================================
# slurm/evaluate_student.slurm
#
# Benchmark: student model vs teacher-embedding linear probe.
# Reports AUC-ROC for both ECG rhythm head and pulmonary head.
#
# Prerequisites:
#   - distill_train job completed (outputs/distill/student_best_*.pth)
#   - Teacher embeddings cached (data/processed/teacher_ecg_embeddings.npy)
#   - more_test.npy present
#
# Usage:
#   # Auto-find best checkpoint:
#   sbatch slurm/evaluate_student.slurm
#
#   # Specify checkpoint explicitly:
#   CKPT=outputs/distill/student_best_ep42_ecgauc0.7830.pth \
#     sbatch slurm/evaluate_student.slurm
# ============================================================

set -euo pipefail

REPO_DIR="/scratch/user/dshekar/more-clinical-distill"
PYTHON_BIN="/scratch/user/dshekar/.conda/envs/deepship_xares2/bin/python"
DISTILL_CFG="${REPO_DIR}/configs/distill_config.yaml"
CKPT_DIR="${REPO_DIR}/outputs/distill"

echo "================================================================"
echo "MoRE Student Model Evaluation"
echo "================================================================"
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      ${SLURM_NODELIST}"
echo "Start:     $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "================================================================"

mkdir -p "${REPO_DIR}/logs"

module load WebProxy 2>/dev/null || true
export http_proxy="${http_proxy:-http://10.73.132.63:8080}"
export https_proxy="${https_proxy:-http://10.73.132.63:8080}"

export HF_HOME="/scratch/user/dshekar/hf_home"

cd "${REPO_DIR}"

# Auto-select checkpoint if not provided
if [ -z "${CKPT:-}" ]; then
    # Pick the best checkpoint by most recent modification time among student_best_*.pth
    CKPT=$(ls -t "${CKPT_DIR}"/student_best_*.pth 2>/dev/null | head -1 || true)
    if [ -z "${CKPT}" ]; then
        echo "ERROR: No student_best_*.pth found in ${CKPT_DIR}"
        echo "  Run distill_train first, or pass CKPT=<path> explicitly."
        exit 1
    fi
fi

echo ""
echo "Checkpoint: ${CKPT}"
echo ""
echo "Running evaluation on test set..."
"${PYTHON_BIN}" distill/evaluate_student.py \
    --config "${DISTILL_CFG}" \
    --checkpoint "${CKPT}" \
    --split test

echo ""
echo "Running evaluation on val set..."
"${PYTHON_BIN}" distill/evaluate_student.py \
    --config "${DISTILL_CFG}" \
    --checkpoint "${CKPT}" \
    --split val

echo ""
echo "================================================================"
echo "Evaluation finished: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "================================================================"
