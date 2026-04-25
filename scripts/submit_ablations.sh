#!/usr/bin/env bash
# ============================================================
# scripts/submit_ablations.sh
#
# Fan-out: submits all 6 ablation training jobs.
# Each job reads its config from configs/ablations/.
#
# Usage:
#   bash scripts/submit_ablations.sh            # submit all
#   bash scripts/submit_ablations.sh A2_lossgate  # submit one by name
# ============================================================

set -euo pipefail

REPO_DIR="/scratch/user/dshekar/more-clinical-distill"
cd "${REPO_DIR}"

mkdir -p logs/ablations

declare -A ABLATIONS=(
    [A0_baseline]="configs/ablations/A0_baseline.yaml"
    [A1_uniform]="configs/ablations/A1_uniform.yaml"
    [A2_lossgate]="configs/ablations/A2_lossgate.yaml"
    [A3_ecgonly]="configs/ablations/A3_ecgonly.yaml"
    [B1_lead2]="configs/ablations/B1_lead2.yaml"
    [B2_v2]="configs/ablations/B2_v2.yaml"
)

FILTER="${1:-}"

for NAME in "${!ABLATIONS[@]}"; do
    if [[ -n "${FILTER}" && "${NAME}" != "${FILTER}" ]]; then
        continue
    fi
    CONFIG="${ABLATIONS[$NAME]}"
    JOB_ID=$(ABLATION_CONFIG="${CONFIG}" sbatch --job-name="${NAME}" slurm/run_ablation.slurm | awk '{print $NF}')
    echo "Submitted ${NAME} → job ${JOB_ID}  (config: ${CONFIG})"
done

echo ""
echo "Monitor with: squeue -u ${USER}"
echo "Logs at:      ${REPO_DIR}/logs/ablations/"
