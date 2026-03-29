#!/usr/bin/env bash
# Download MoRE teacher model weights from Google Drive.
# Run this on a LOGIN NODE (not via sbatch) — compute nodes lack pip access.
# Uses gdown; falls back to wget with cookie bypass.
set -uo pipefail

DEST="/scratch/user/dshekar/more-clinical-distill/outputs/more_pretrained.pth"
FILE_ID="1BB9dT6iYihqJarD5qX0bdnfYhiwhBgmH"

if [ -f "${DEST}" ] && [ -s "${DEST}" ]; then
    echo "Teacher weights already exist: $(ls -lh "${DEST}" | awk '{print $5, $NF}')"
    exit 0
fi

mkdir -p "$(dirname "${DEST}")"

# Find a usable Python
PYTHON_BIN=""
for p in \
    "/scratch/user/dshekar/.conda/envs/more_kd/bin/python" \
    "/scratch/user/dshekar/.conda/envs/deepship_xares2/bin/python" \
    "$(which python3 2>/dev/null)" \
    "$(which python 2>/dev/null)"; do
    [ -n "$p" ] && [ -f "$p" ] && PYTHON_BIN="$p" && break
done

if [ -n "${PYTHON_BIN}" ]; then
    echo "Using Python: ${PYTHON_BIN}"
    "${PYTHON_BIN}" -c "import gdown" 2>/dev/null || \
        "${PYTHON_BIN}" -m pip install -q gdown
    "${PYTHON_BIN}" -m gdown "https://drive.google.com/uc?id=${FILE_ID}" -O "${DEST}"
    if [ -f "${DEST}" ] && [ -s "${DEST}" ]; then
        echo "Download complete: $(ls -lh "${DEST}" | awk '{print $5}')"
        exit 0
    fi
fi

# Fallback: direct wget with cookie confirmation
echo "gdown unavailable or failed, trying wget..."
wget --no-check-certificate \
     "https://drive.google.com/uc?export=download&id=${FILE_ID}&confirm=t" \
     -O "${DEST}"

if [ -f "${DEST}" ] && [ -s "${DEST}" ]; then
    echo "Download complete: $(ls -lh "${DEST}" | awk '{print $5}')"
else
    rm -f "${DEST}"
    echo ""
    echo "ERROR: Automated download failed (likely no internet on this node)."
    echo "Download on a machine with internet access and scp here:"
    echo "  pip install gdown"
    echo "  gdown '${FILE_ID}' -O more_pretrained.pth"
    echo "  scp more_pretrained.pth dshekar@grace.hprc.tamu.edu:${DEST}"
    exit 1
fi
