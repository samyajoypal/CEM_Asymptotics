"""Expand the frozen design into deterministic Slurm task manifests."""

from __future__ import annotations

import fnmatch
import json
from pathlib import Path

import pandas as pd

from cem_inference.scenarios import scenario_registry


def main():
    project = Path(__file__).resolve().parents[1]
    design = json.loads((project / "configs/simulation_design.json").read_text())
    scenario_ids = sorted(scenario_registry())
    target_rows = []
    for scenario in scenario_ids:
        repetitions = (
            design["target"]["p10_replications"]
            if "_p10_" in scenario
            else design["target"]["replications"]
        )
        target_rows.append(
            {
                "scenario": scenario,
                "reference_size": design["target"]["reference_size"],
                "replications": repetitions,
                "output_dir": "results/targets/confirmatory_v2",
            }
        )

    simulation_rows = []
    chunk_size = 50
    for experiment in design["experiments"]:
        selected = sorted(
            {
                scenario
                for scenario in scenario_ids
                for pattern in experiment["scenario_patterns"]
                if fnmatch.fnmatch(scenario, pattern)
            }
        )
        for scenario in selected:
            for sample_size in experiment["sample_sizes"]:
                total = experiment["replications"]
                for start in range(0, total, chunk_size):
                    repetitions = min(chunk_size, total - start)
                    simulation_rows.append(
                        {
                            "experiment": experiment["name"],
                            "scenario": scenario,
                            "sample_size": sample_size,
                            "start": start,
                            "replications": repetitions,
                            "target": f"results/targets/confirmatory_v2/{scenario}_summary.csv",
                            "output": (
                                f"results/raw/confirmatory_v2/{experiment['name']}/"
                                f"{scenario}_n{sample_size}_start{start:04d}.csv"
                            ),
                        }
                    )

    manifest_dir = project / "hpc/manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(target_rows).to_csv(manifest_dir / "targets.csv", index=False)
    pd.DataFrame(simulation_rows).to_csv(
        manifest_dir / "simulation_tasks.csv", index=False
    )
    print(f"Target tasks: {len(target_rows)}")
    print(f"Simulation tasks: {len(simulation_rows)}")
    print(f"Primary replications: {sum(row['replications'] for row in simulation_rows)}")


if __name__ == "__main__":
    main()
