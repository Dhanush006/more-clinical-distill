#!/usr/bin/env bash
#SBATCH --job-name=distill_ablation
#SBATCH --output=/scratch/user/dshekar/more-clinical-distill/logs/ablations/%x_%j.log
#SBATCH --error=/scratch/user/dshekar/more-clinical-distill/logs/ablations/%x_%j.log
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

# ============================================================
# slurm/run_ablation.slurm
#
# Parametric distillation ablation job. Driven by ABLATION_CONFIG.
#
# Usage (from scripts/submit_ablations.sh):
#   ABLATION_CONFIG=configs/ablations/A2_lossgate.yaml \
#   sbatch --job-name=A2_lossgate slurm/run_ablation.slurm
#
# Or manually:
#   ABLATION_CONFIG=configs/ablations/A0_baseline.yaml sbatch slurm/run_ablation.slurm
# ============================================================

set -euo pipefail

REPO_DIR="/scratch/user/dshekar/more-clinical-distill"
CONFIG="${ABLATION_CONFIG:-configs/distill_config.yaml}"
PYTHON_BIN="/scratch/user/dshekar/.conda/envs/deepship_xares2/bin/python"

echo "================================================================"
echo "Distillation Ablation Run"
echo "================================================================"
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      ${SLURM_NODELIST}"
echo "Config:    ${CONFIG}"
echo "Start:     $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "================================================================"

cd "${REPO_DIR}"

# Create log dir for ablations if it doesn't exist
mkdir -p logs/ablations

module load WebProxy 2>/dev/null || true

"${PYTHON_BIN}" distill/distill_train.py --config "${CONFIG}"

echo ""
echo "================================================================"
echo "Ablation complete: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "================================================================"
