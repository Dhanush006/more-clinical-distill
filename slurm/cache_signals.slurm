#!/usr/bin/env bash
#SBATCH --job-name=cache_signals
#SBATCH --output=/scratch/user/dshekar/more-clinical-distill/logs/cache_signals_%j.log
#SBATCH --error=/scratch/user/dshekar/more-clinical-distill/logs/cache_signals_%j.log
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=01:00:00

# Pre-cache all 49,076 ECGs to a single (N, 1000, 12) float32 .npy
# Eliminates per-batch wfdb.rdsamp + scipy.resample → ~30x epoch speedup.

set -euo pipefail

REPO_DIR="/scratch/user/dshekar/more-clinical-distill"
PYTHON_BIN="/scratch/user/dshekar/.conda/envs/deepship_xares2/bin/python"

cd "${REPO_DIR}"

echo "================================================================"
echo "ECG Signal Cache Build"
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      ${SLURM_NODELIST}"
echo "Start:     $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "================================================================"

"${PYTHON_BIN}" scripts/cache_ecg_signals.py --workers 16

echo ""
echo "================================================================"
echo "Cache build complete: $(date '+%Y-%m-%d %H:%M:%S %Z')"
ls -lh data/processed/ecg_signals_100hz.npy data/processed/ecg_signal_ids.npy
echo "================================================================"
