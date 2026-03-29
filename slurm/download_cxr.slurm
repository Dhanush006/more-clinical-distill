#!/bin/bash
#SBATCH --job-name=more_cxr_dl
#SBATCH --output=logs/download_cxr_%j.log
#SBATCH --error=logs/download_cxr_%j.err
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --time=24:00:00

# ============================================================
# slurm/download_cxr.slurm
#
# Downloads MIMIC-CXR-JPG files + CXR metadata CSVs.
# Requires separate DUA approval at:
#   https://physionet.org/content/mimic-cxr-jpg/2.1.0/
#
# Usage:
#   sbatch slurm/download_cxr.slurm
# ============================================================

set -euo pipefail

REPO_DIR="/scratch/user/dshekar/more-clinical-distill"
ENV_FILE="${REPO_DIR}/.env"

echo "================================================================"
echo "MoRE CXR Download Job"
echo "================================================================"
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      ${SLURM_NODELIST}"
echo "Start:     $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "================================================================"

set -a; source "${ENV_FILE}"; set +a
if [ -z "${PHYSIONET_USER:-}" ] || [ -z "${PHYSIONET_PASS:-}" ]; then
    echo "ERROR: Credentials not set in .env"
    exit 1
fi
echo "Credentials loaded (values hidden)"

module load WebProxy 2>/dev/null || echo "WebProxy not available"
export http_proxy="${http_proxy:-http://10.73.132.63:8080}"
export https_proxy="${https_proxy:-http://10.73.132.63:8080}"

mkdir -p "${REPO_DIR}/logs"
cd "${REPO_DIR}"

# ── Step 1: (Re-)attempt CXR metadata CSVs ───────────────────────────
echo ""
echo "Step 1/2: Downloading CXR metadata CSVs..."
REPO_DIR="${REPO_DIR}" bash scripts/download_metadata_csvs.sh

# ── Step 2: Download CXR .jpg files ──────────────────────────────────
echo ""
echo "Step 2/2: Downloading CXR (.jpg) files..."
CXR_LIST="${REPO_DIR}/manifests/cxr_download_list.txt"
CXR_DIR="${REPO_DIR}/data/mimic-cxr-jpg"
mkdir -p "${CXR_DIR}"

if [ ! -f "${CXR_LIST}" ]; then
    echo "ERROR: CXR download list not found: ${CXR_LIST}"
    exit 1
fi

CXR_COUNT=$(wc -l < "${CXR_LIST}")
echo "  URLs in list: ${CXR_COUNT}"
echo "  Start: $(date '+%Y-%m-%d %H:%M:%S')"

START=$(date +%s)
wget \
    --user="${PHYSIONET_USER}" \
    --password="${PHYSIONET_PASS}" \
    -nv \
    -nc \
    -x \
    --no-check-certificate \
    --directory-prefix="${CXR_DIR}" \
    --input-file="${CXR_LIST}" \
    2>&1 | grep -v "^--" || true

END=$(date +%s)
CXR_JPG=$(find "${CXR_DIR}" -name "*.jpg" 2>/dev/null | wc -l)
CXR_SIZE=$(du -sh "${CXR_DIR}" 2>/dev/null | cut -f1)

echo ""
echo "================================================================"
echo "CXR Download Summary"
echo "================================================================"
echo "  .jpg files: ${CXR_JPG}"
echo "  Total size: ${CXR_SIZE}"
echo "  Duration:   $((END - START))s"
echo "  Finish:     $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "================================================================"
echo ""
echo "NEXT STEP: Run preprocessing pipeline:"
echo "  sbatch slurm/preprocess.slurm"
