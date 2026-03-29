#!/usr/bin/env bash
# ============================================================
# scripts/download_metadata_csvs.sh
#
# Downloads small MIMIC metadata CSVs directly to the paths
# expected by configs/paths.yaml — NOT using wget -x (mirror),
# so each file lands at exactly its target location.
#
# Files downloaded:
#   MIMIC-CXR-JPG:
#     mimic-cxr-2.0.0-chexpert.csv
#     mimic-cxr-2.0.0-split.csv
#     mimic-cxr-2.0.0-metadata.csv
#   MIMIC-IV-ECG:
#     machine_measurements.csv
#     record_list.csv
#
# Reads credentials from env: PHYSIONET_USER, PHYSIONET_PASS
# ============================================================

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
CXR_META_DIR="${REPO_DIR}/data/mimic-cxr-jpg"
ECG_META_DIR="${REPO_DIR}/data/mimic-iv-ecg"

MIMIC_CXR_BASE="https://physionet.org/files/mimic-cxr-jpg/2.0.0"
MIMIC_ECG_BASE="https://physionet.org/files/mimic-iv-ecg/1.0"

if [ -z "${PHYSIONET_USER:-}" ] || [ -z "${PHYSIONET_PASS:-}" ]; then
    echo "ERROR: PHYSIONET_USER and PHYSIONET_PASS must be set."
    exit 1
fi

mkdir -p "${CXR_META_DIR}" "${ECG_META_DIR}"

wget_file() {
    local url="$1"
    local dest="$2"
    if [ -f "${dest}" ]; then
        echo "  SKIP (exists): $(basename "${dest}")"
        return 0
    fi
    echo "  Downloading: $(basename "${dest}")"
    wget \
        --user="${PHYSIONET_USER}" \
        --password="${PHYSIONET_PASS}" \
        -nv \
        --no-check-certificate \
        -O "${dest}" \
        "${url}" 2>&1 | grep -v "^--" || true
    echo "  Done: $(basename "${dest}")"
}

echo "=============================================="
echo "Downloading MIMIC metadata CSVs"
echo "=============================================="

# ── MIMIC-IV-ECG metadata (accessible, no .gz — download directly) ───
echo ""
echo "MIMIC-IV-ECG:"
wget_file "${MIMIC_ECG_BASE}/machine_measurements.csv" \
          "${ECG_META_DIR}/machine_measurements.csv"

wget_file "${MIMIC_ECG_BASE}/record_list.csv" \
          "${ECG_META_DIR}/record_list.csv"

# ── MIMIC-CXR-JPG metadata ────────────────────────────────────────────
# NOTE: Requires separate DUA approval at physionet.org/content/mimic-cxr-jpg/
# Skip gracefully if not yet granted (403). Re-run after DUA approval.
echo ""
echo "MIMIC-CXR-JPG (skipping if 403 — requires separate DUA approval):"
for csv_name in "mimic-cxr-2.0.0-chexpert.csv" "mimic-cxr-2.0.0-split.csv" "mimic-cxr-2.0.0-metadata.csv"; do
    dest="${CXR_META_DIR}/${csv_name}"
    if [ -f "${dest}" ]; then
        echo "  SKIP (exists): ${csv_name}"
        continue
    fi
    http_code=$(wget --user="${PHYSIONET_USER}" --password="${PHYSIONET_PASS}" \
        --no-check-certificate --spider -S \
        "${MIMIC_CXR_BASE}/${csv_name}" 2>&1 | grep "HTTP/" | tail -1 | awk '{print $2}')
    if [ "${http_code}" = "200" ]; then
        echo "  Downloading: ${csv_name}"
        wget --user="${PHYSIONET_USER}" --password="${PHYSIONET_PASS}" \
            --no-check-certificate -nv -O "${dest}" \
            "${MIMIC_CXR_BASE}/${csv_name}" 2>&1 | grep -v "^--" || true
    else
        echo "  SKIP (HTTP ${http_code}): ${csv_name}  — complete DUA at physionet.org/content/mimic-cxr-jpg/"
    fi
done

echo ""
echo "=============================================="
echo "Metadata CSV download complete"
ls -lh "${CXR_META_DIR}"/*.csv 2>/dev/null || true
ls -lh "${ECG_META_DIR}"/*.csv 2>/dev/null || true
echo "=============================================="
