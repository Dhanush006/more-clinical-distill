#!/usr/bin/env bash
# ============================================================
# scripts/download_subset.sh
#
# Downloads ECG and CXR files from PhysioNet using pre-built
# URL lists produced by build_download_lists.py.
#
# Reads credentials from .env (PHYSIONET_USER, PHYSIONET_PASS).
# Credentials are NEVER printed or logged.
#
# Usage:
#   # Load env and run directly (interactive):
#   set -a && source .env && set +a
#   bash scripts/download_subset.sh
#
#   # Or let slurm/download_subset.slurm handle it (recommended).
#
# Options (env vars, all optional):
#   ECG_LIST   path to ecg_download_list.txt  (default: manifests/ecg_download_list.txt)
#   CXR_LIST   path to cxr_download_list.txt  (default: manifests/cxr_download_list.txt)
#   ECG_DIR    ECG destination root           (default: data/mimic-iv-ecg)
#   CXR_DIR    CXR destination root           (default: data/mimic-cxr-jpg)
#   WGET_JOBS  parallel wget connections       (default: 4)
# ============================================================

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

ECG_LIST="${ECG_LIST:-${REPO_DIR}/manifests/ecg_download_list.txt}"
CXR_LIST="${CXR_LIST:-${REPO_DIR}/manifests/cxr_download_list.txt}"
ECG_DIR="${ECG_DIR:-${REPO_DIR}/data/mimic-iv-ecg}"
CXR_DIR="${CXR_DIR:-${REPO_DIR}/data/mimic-cxr-jpg}"
WGET_JOBS="${WGET_JOBS:-4}"

# ── Credential check (fail fast, never echo values) ──────────────────
if [ -z "${PHYSIONET_USER:-}" ] || [ -z "${PHYSIONET_PASS:-}" ]; then
    echo "ERROR: PHYSIONET_USER and PHYSIONET_PASS must be set."
    echo "  Copy .env.example to .env, fill in values, then:"
    echo "  set -a && source .env && set +a"
    exit 1
fi

# ── Preflight ─────────────────────────────────────────────────────────
for LIST in "${ECG_LIST}" "${CXR_LIST}"; do
    if [ ! -f "${LIST}" ]; then
        echo "ERROR: download list not found: ${LIST}"
        echo "  Run:  python scripts/build_download_lists.py"
        exit 1
    fi
done

mkdir -p "${ECG_DIR}" "${CXR_DIR}"

ECG_COUNT=$(wc -l < "${ECG_LIST}")
CXR_COUNT=$(wc -l < "${CXR_LIST}")
echo "================================================"
echo "PhysioNet Download"
echo "================================================"
echo "  ECG list:  ${ECG_LIST}  (${ECG_COUNT} URLs)"
echo "  CXR list:  ${CXR_LIST}  (${CXR_COUNT} URLs)"
echo "  ECG dest:  ${ECG_DIR}"
echo "  CXR dest:  ${CXR_DIR}"
echo "  Parallel:  ${WGET_JOBS}"
echo "  Start:     $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================"

# wget helper: mirrors PhysioNet directory structure, skips existing files
# -nv: non-verbose (no URL printing, hides credentials from logs)
# -nc: skip already-downloaded files
# -x : create directory hierarchy
# --no-check-certificate: Grace cluster sometimes has proxy SSL issues
wget_download() {
    local list_file="$1"
    local dest_dir="$2"
    wget \
        --user="${PHYSIONET_USER}" \
        --password="${PHYSIONET_PASS}" \
        -nv \
        -nc \
        -x \
        --no-check-certificate \
        --directory-prefix="${dest_dir}" \
        --input-file="${list_file}" \
        2>&1 | grep -v "^--" || true  # filter wget progress noise
}

# ── Download ECG files ────────────────────────────────────────────────
echo ""
echo "Downloading ECG (.hea + .dat) files..."
START_ECG=$(date +%s)
wget_download "${ECG_LIST}" "${ECG_DIR}"
END_ECG=$(date +%s)
echo "ECG download done in $((END_ECG - START_ECG))s"

# Quick count check
ECG_HEA=$(find "${ECG_DIR}" -name "*.hea" 2>/dev/null | wc -l)
ECG_DAT=$(find "${ECG_DIR}" -name "*.dat" 2>/dev/null | wc -l)
echo "  .hea files found: ${ECG_HEA}"
echo "  .dat files found: ${ECG_DAT}"

# ── Download CXR files ────────────────────────────────────────────────
echo ""
echo "Downloading CXR (.jpg) files..."
START_CXR=$(date +%s)
wget_download "${CXR_LIST}" "${CXR_DIR}"
END_CXR=$(date +%s)
echo "CXR download done in $((END_CXR - START_CXR))s"

CXR_JPG=$(find "${CXR_DIR}" -name "*.jpg" 2>/dev/null | wc -l)
echo "  .jpg files found: ${CXR_JPG}"

echo ""
echo "================================================"
echo "Download complete: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Total ECG: ${ECG_HEA} .hea  ${ECG_DAT} .dat"
echo "Total CXR: ${CXR_JPG} .jpg"
echo "================================================"
