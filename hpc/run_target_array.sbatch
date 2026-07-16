#!/usr/bin/env bash
#SBATCH --job-name=cem-target
#SBATCH --partition=epyc-768
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=results/logs/target_%A_%a.out
#SBATCH --error=results/logs/target_%A_%a.err

set -euo pipefail
: "${SLURM_SUBMIT_DIR:?Submit this job from the CEM_JRSSB project root}"
cd "$SLURM_SUBMIT_DIR"
module load python/3.11

row=$(awk -F, -v line="$((SLURM_ARRAY_TASK_ID + 2))" 'NR==line {print}' hpc/manifests/targets.csv)
IFS=, read -r scenario reference_size replications output_dir <<< "$row"

export PYTHONPATH=src
export MPLCONFIGDIR="${TMPDIR:-/tmp}/cem-mpl-${SLURM_JOB_ID}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

.venv311/bin/python scripts/compute_cml_target.py \
  --scenario "$scenario" \
  --reference-size "$reference_size" \
  --replications "$replications" \
  --output-dir "$output_dir"
