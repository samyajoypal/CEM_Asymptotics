#!/usr/bin/env bash
#SBATCH --job-name=cem-sim
#SBATCH --partition=epyc-768
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=results/logs/simulation_%A_%a.out
#SBATCH --error=results/logs/simulation_%A_%a.err

set -euo pipefail
: "${SLURM_SUBMIT_DIR:?Submit this job from the CEM_JRSSB project root}"
cd "$SLURM_SUBMIT_DIR"
module load python/3.11

manifest="${MANIFEST:-hpc/manifests/simulation_tasks.csv}"
row=$(awk -F, -v line="$((SLURM_ARRAY_TASK_ID + 2))" 'NR==line {print}' "$manifest")
IFS=, read -r experiment scenario sample_size start replications target output <<< "$row"

export PYTHONPATH=src
export MPLCONFIGDIR="${TMPDIR:-/tmp}/cem-mpl-${SLURM_JOB_ID}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

.venv311/bin/python scripts/run_simulation_chunk.py \
  --scenario "$scenario" \
  --sample-size "$sample_size" \
  --start "$start" \
  --replications "$replications" \
  --target "$target" \
  --output "$output"
