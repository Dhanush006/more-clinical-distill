#!/usr/bin/env bash
# ============================================================
# scripts/download_metadata_csvs.sh
#
# Downloads small MIMIC metadata CSVs directly to the paths
# expected by configs/paths.yaml — NOT using wget -x (mirror),
# so each file lands at exactly its target location.
#
# Files downloaded:
#   MIMIC-IV-ECG:
#     machine_measurements.csv   (~175MB)
#     record_list.csv            (~68MB)
#   MIMIC-CXR-JPG (2.1.0, requires separate DUA):
#     mimic-cxr-2.0.0-chexpert.csv   (decompressed from .gz)
#     mimic-cxr-2.0.0-split.csv
#     mimic-cxr-2.0.0-metadata.csv
#
# Reads credentials from env: PHYSIONET_USER, PHYSIONET_PASS
# ============================================================

set -uo pipefail   # NOTE: no -e; we handle wget errors per-call

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
CXR_META_DIR="${REPO_DIR}/data/mimic-cxr-jpg"
ECG_META_DIR="${REPO_DIR}/data/mimic-iv-ecg"

# MIMIC-CXR-JPG 2.1.0 — file names inside still carry 2.0.0 label
MIMIC_CXR_BASE="https://physionet.org/files/mimic-cxr-jpg/2.1.0"
MIMIC_ECG_BASE="https://physionet.org/files/mimic-iv-ecg/1.0"

if [ -z "${PHYSIONET_USER:-}" ] || [ -z "${PHYSIONET_PASS:-}" ]; then
    echo "ERROR: PHYSIONET_USER and PHYSIONET_PASS must be set."
    exit 1
fi

mkdir -p "${CXR_META_DIR}" "${ECG_META_DIR}"

# wget_file URL DEST
# Downloads URL to DEST; skips if already present; never crashes caller.
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
        "${url}" 2>&1 | grep -v "^--" || { echo "  WARNING: wget failed for $(basename "${dest}")"; rm -f "${dest}"; return 0; }
    echo "  Done: $(basename "${dest}")  ($(du -h "${dest}" | cut -f1))"
}

# wget_check URL — returns HTTP status code without crashing on error
wget_check() {
    local url="$1"
    wget --user="${PHYSIONET_USER}" --password="${PHYSIONET_PASS}" \
        --no-check-certificate --spider -S \
        "${url}" 2>&1 | grep "HTTP/" | tail -1 | awk '{print $2}' || echo "000"
}

echo "=============================================="
echo "Downloading MIMIC metadata CSVs"
echo "=============================================="

# ── MIMIC-IV-ECG metadata ─────────────────────────────────────────────
echo ""
echo "MIMIC-IV-ECG:"
wget_file "${MIMIC_ECG_BASE}/machine_measurements.csv" \
          "${ECG_META_DIR}/machine_measurements.csv"
wget_file "${MIMIC_ECG_BASE}/record_list.csv" \
          "${ECG_META_DIR}/record_list.csv"

# ── MIMIC-CXR-JPG metadata (2.1.0, .gz on server, decompress locally) ─
# Skip gracefully if DUA not yet approved (403) — re-run after approval.
echo ""
echo "MIMIC-CXR-JPG (2.1.0):"
for csv_base in "mimic-cxr-2.0.0-chexpert" "mimic-cxr-2.0.0-split" "mimic-cxr-2.0.0-metadata"; do
    csv_dest="${CXR_META_DIR}/${csv_base}.csv"
    gz_dest="${CXR_META_DIR}/${csv_base}.csv.gz"

    if [ -f "${csv_dest}" ]; then
        echo "  SKIP (exists): ${csv_base}.csv"
        continue
    fi

    # probe without crashing
    http_code=$(wget_check "${MIMIC_CXR_BASE}/${csv_base}.csv.gz")

    if [ "${http_code}" = "200" ]; then
        wget_file "${MIMIC_CXR_BASE}/${csv_base}.csv.gz" "${gz_dest}"
        if [ -f "${gz_dest}" ]; then
            echo "  Decompressing ${csv_base}.csv..."
            gunzip -k "${gz_dest}"
            echo "  Decompressed: ${csv_base}.csv  ($(du -h "${csv_dest}" | cut -f1))"
        fi
    else
        echo "  SKIP (HTTP ${http_code}): ${csv_base}.csv — complete DUA at physionet.org/content/mimic-cxr-jpg/"
    fi
done

echo ""
echo "=============================================="
echo "Metadata CSV download complete"
ls -lh "${CXR_META_DIR}"/*.csv 2>/dev/null || echo "  (no CXR CSVs yet)"
ls -lh "${ECG_META_DIR}"/*.csv 2>/dev/null || echo "  (no ECG CSVs yet)"
echo "=============================================="
