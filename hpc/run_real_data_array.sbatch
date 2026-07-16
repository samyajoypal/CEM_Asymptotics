#!/usr/bin/env bash
#SBATCH --job-name=cem-real
#SBATCH --partition=cpuidle
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=results/logs/real_%A_%a.out
#SBATCH --error=results/logs/real_%A_%a.err

set -euo pipefail
: "${SLURM_SUBMIT_DIR:?Submit this job from the CEM_JRSSB project root}"
cd "$SLURM_SUBMIT_DIR"
module load python/3.11

row=$(awk -F, -v line="$((SLURM_ARRAY_TASK_ID + 2))" 'NR==line {print}' hpc/manifests/real_data_tasks.csv)
IFS=, read -r dataset p mode start replications output <<< "$row"

export PYTHONPATH=src
export MPLCONFIGDIR="$HOME/.cache/cem-matplotlib"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$MPLCONFIGDIR"

.venv311/bin/python scripts/run_real_data_bootstrap_chunk.py \
  --dataset "$dataset" --p "$p" --mode "$mode" --start "$start" \
  --replications "$replications" --output "$output"
