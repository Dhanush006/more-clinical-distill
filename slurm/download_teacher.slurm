#!/usr/bin/env bash
#SBATCH --job-name=more_teacher_dl
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/user/dshekar/more-clinical-distill/logs/download_teacher_%j.log
#SBATCH --error=/scratch/user/dshekar/more-clinical-distill/logs/download_teacher_%j.log

cd /scratch/user/dshekar/more-clinical-distill
bash scripts/download_teacher_weights.sh
