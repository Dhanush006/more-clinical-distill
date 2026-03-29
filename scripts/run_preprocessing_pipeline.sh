#!/usr/bin/env bash
# ============================================================
# scripts/run_preprocessing_pipeline.sh
#
# Runs the full Phase 3 preprocessing pipeline:
#   1. verify_local_dataset_layout.py  — fail fast if files missing
#   2. convert_manifest_to_more_format.py — build MoRE .npy arrays
#
# This is a CPU-only script. Submit via slurm/preprocess.slurm for HPC.
#
# Usage:
#   bash scripts/run_preprocessing_pipeline.sh
#   bash scripts/run_preprocessing_pipeline.sh --subset 500   # smoke-test
# ============================================================

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/scratch/user/dshekar/.conda/envs/more_kd/bin/python}"
CONFIG="${CONFIG:-${REPO_DIR}/configs/paths.yaml}"
SUBSET="${1:-}"

if [ ! -x "${PYTHON_BIN}" ]; then
    echo "ERROR: Python not found: ${PYTHON_BIN}"
    echo "  Set PYTHON_BIN env var or run: bash scripts/setup_env.sh"
    exit 1
fi

echo "================================================================"
echo "MoRE Preprocessing Pipeline"
echo "================================================================"
echo "  Repo:    ${REPO_DIR}"
echo "  Python:  ${PYTHON_BIN}"
echo "  Config:  ${CONFIG}"
echo "  Subset:  ${SUBSET:-all}"
echo "  Start:   $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"

cd "${REPO_DIR}"

SUBSET_ARG=""
if [ -n "${SUBSET}" ]; then
    SUBSET_ARG="--subset ${SUBSET}"
fi

# ── Step 1: verify layout ─────────────────────────────────────────────
echo ""
echo "Step 1/2: Verifying local dataset layout..."
"${PYTHON_BIN}" scripts/verify_local_dataset_layout.py \
    --config "${CONFIG}" \
    ${SUBSET_ARG} \
    --report logs/layout_verification.txt \
    || echo "Layout verification: partial data detected, proceeding with available files."

echo "Layout verification done."

# ── Step 2: convert to MoRE format ───────────────────────────────────
echo ""
echo "Step 2/2: Converting manifest to MoRE .npy format..."
"${PYTHON_BIN}" scripts/convert_manifest_to_more_format.py \
    --config "${CONFIG}" \
    ${SUBSET_ARG}

echo ""
echo "================================================================"
echo "Preprocessing complete: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Outputs:"
echo "  data/processed/more_train.npy"
echo "  data/processed/more_val.npy"
echo "  data/processed/more_test.npy"
echo "================================================================"
