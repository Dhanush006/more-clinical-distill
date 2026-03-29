#!/bin/bash
#SBATCH --job-name=more_ecg_dl
#SBATCH --output=logs/download_ecg_%j.log
#SBATCH --error=logs/download_ecg_%j.err
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --time=12:00:00

# ============================================================
# slurm/download_ecg.slurm
#
# Downloads MIMIC-IV-ECG files + ECG metadata CSVs.
# CXR download is separate (requires MIMIC-CXR-JPG DUA approval).
#
# Usage:
#   sbatch slurm/download_ecg.slurm
# ============================================================

set -euo pipefail

REPO_DIR="/scratch/user/dshekar/more-clinical-distill"
ENV_FILE="${REPO_DIR}/.env"

echo "================================================================"
echo "MoRE ECG-Only Download Job"
echo "================================================================"
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      ${SLURM_NODELIST}"
echo "Start:     $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "Repo dir:  ${REPO_DIR}"
echo "================================================================"

# ── Load credentials ──────────────────────────────────────────────────
if [ ! -f "${ENV_FILE}" ]; then
    echo "ERROR: .env not found at ${ENV_FILE}"
    exit 1
fi
set -a; source "${ENV_FILE}"; set +a
if [ -z "${PHYSIONET_USER:-}" ] || [ -z "${PHYSIONET_PASS:-}" ]; then
    echo "ERROR: PHYSIONET_USER or PHYSIONET_PASS not set in .env"
    exit 1
fi
echo "Credentials loaded (values hidden)"

# ── Load web proxy ────────────────────────────────────────────────────
module load WebProxy 2>/dev/null || echo "WebProxy module not available"
export http_proxy="${http_proxy:-http://10.73.132.63:8080}"
export https_proxy="${https_proxy:-http://10.73.132.63:8080}"
echo "Proxy: ${http_proxy}"

mkdir -p "${REPO_DIR}/logs" "${REPO_DIR}/manifests"

cd "${REPO_DIR}"

# ── Step 1: Download metadata CSVs ────────────────────────────────────
echo ""
echo "Step 1/2: Downloading metadata CSVs..."
REPO_DIR="${REPO_DIR}" bash scripts/download_metadata_csvs.sh

# ── Step 2: Download ECG .hea + .dat files ────────────────────────────
echo ""
echo "Step 2/2: Downloading ECG (.hea + .dat) files..."
echo "  ECG list: manifests/ecg_download_list.txt"
echo "  ECG dest: data/mimic-iv-ecg/"

ECG_LIST="${REPO_DIR}/manifests/ecg_download_list.txt"
ECG_DIR="${REPO_DIR}/data/mimic-iv-ecg"
mkdir -p "${ECG_DIR}"

if [ ! -f "${ECG_LIST}" ]; then
    echo "ERROR: ECG download list not found: ${ECG_LIST}"
    echo "  Run: python scripts/build_download_lists.py"
    exit 1
fi

ECG_COUNT=$(wc -l < "${ECG_LIST}")
echo "  URLs in list: ${ECG_COUNT}"
echo "  Start: $(date '+%Y-%m-%d %H:%M:%S')"

START=$(date +%s)
wget \
    --user="${PHYSIONET_USER}" \
    --password="${PHYSIONET_PASS}" \
    -nv \
    -nc \
    -x \
    --no-check-certificate \
    --directory-prefix="${ECG_DIR}" \
    --input-file="${ECG_LIST}" \
    2>&1 | grep -v "^--" || true

END=$(date +%s)
echo "  ECG download done in $((END - START))s"

# ── Summary ───────────────────────────────────────────────────────────
ECG_HEA=$(find "${ECG_DIR}" -name "*.hea" 2>/dev/null | wc -l)
ECG_DAT=$(find "${ECG_DIR}" -name "*.dat" 2>/dev/null | wc -l)
ECG_SIZE=$(du -sh "${ECG_DIR}" 2>/dev/null | cut -f1)

echo ""
echo "================================================================"
echo "ECG Download Summary"
echo "================================================================"
echo "  .hea files: ${ECG_HEA}"
echo "  .dat files: ${ECG_DAT}"
echo "  Total size: ${ECG_SIZE}"
echo "  Finish:     $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "================================================================"
echo ""
echo "NEXT STEP: Once MIMIC-CXR-JPG DUA is approved, run:"
echo "  sbatch slurm/download_cxr.slurm"
