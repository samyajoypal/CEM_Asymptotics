#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p results/logs results/targets/confirmatory results/raw/confirmatory

target_count=$(($(wc -l < hpc/manifests/targets.csv) - 1))
simulation_count=$(($(wc -l < hpc/manifests/simulation_tasks.csv) - 1))

target_job=$(sbatch --parsable --array="0-$((target_count - 1))%4" hpc/run_target_array.sbatch)
echo "Submitted target array: $target_job"

simulation_job=$(sbatch --parsable --dependency="afterok:${target_job}" \
  --array="0-$((simulation_count - 1))%20" hpc/run_simulation_array.sbatch)
echo "Submitted simulation array: $simulation_job (depends on $target_job)"
