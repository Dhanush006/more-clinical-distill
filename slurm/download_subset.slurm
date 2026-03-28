#!/bin/bash
#SBATCH --job-name=more_download
#SBATCH --output=logs/download_%j.log
#SBATCH --error=logs/download_%j.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --time=12:00:00

# ============================================================
# slurm/download_subset.slurm
#
# Downloads MIMIC-IV-ECG and MIMIC-CXR-JPG files from PhysioNet.
# Credentials are read from .env at REPO_DIR/.env — NEVER
# hardcoded here. This file is safe to commit.
#
# Usage:
#   # Build download lists first:
#   python scripts/build_download_lists.py
#
#   # Submit:
#   sbatch slurm/download_subset.slurm
#
#   # For a smoke-test subset (500 subjects):
#   python scripts/build_download_lists.py --subset 500
#   sbatch slurm/download_subset.slurm
# ============================================================

set -euo pipefail

REPO_DIR="/scratch/user/dshekar/more-clinical-distill"
ENV_FILE="${REPO_DIR}/.env"

echo "================================================================"
echo "MoRE PhysioNet Download Job"
echo "================================================================"
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Node:         ${SLURM_NODELIST}"
echo "Partition:    ${SLURM_JOB_PARTITION}"
echo "Start:        $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "Host:         $(hostname)"
echo "Repo dir:     ${REPO_DIR}"
echo "================================================================"

# ── Load credentials from .env ────────────────────────────────────────
if [ ! -f "${ENV_FILE}" ]; then
    echo "ERROR: .env not found at ${ENV_FILE}"
    echo "  Copy .env.example to .env and fill in credentials."
    exit 1
fi

# Source silently — do not echo any credential values
set -a
# shellcheck source=/dev/null
source "${ENV_FILE}"
set +a

# ── Verify credentials loaded (existence only, never print values) ────
if [ -z "${PHYSIONET_USER:-}" ] || [ -z "${PHYSIONET_PASS:-}" ]; then
    echo "ERROR: PHYSIONET_USER or PHYSIONET_PASS not set in .env"
    exit 1
fi
echo "Credentials loaded: PHYSIONET_USER is set (value hidden)"

# ── Load web proxy (required on Grace for outbound HTTP) ─────────────
module load WebProxy 2>/dev/null || echo "WebProxy module not available — skipping"
export http_proxy="${http_proxy:-http://10.73.132.63:8080}"
export https_proxy="${https_proxy:-http://10.73.132.63:8080}"
echo "Proxy: ${http_proxy}"

# ── Ensure logs dir exists ────────────────────────────────────────────
mkdir -p "${REPO_DIR}/logs"
mkdir -p "${REPO_DIR}/manifests"

# ── Run download ──────────────────────────────────────────────────────
cd "${REPO_DIR}"

echo ""
echo "Running download_subset.sh..."
bash scripts/download_subset.sh

echo ""
echo "================================================================"
echo "Job finished: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "================================================================"
