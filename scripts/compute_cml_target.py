"""Compute independently replicated CML targets for one frozen scenario."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cem_inference import approximate_cml_target
from cem_inference.scenarios import get_scenario


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--reference-size", type=int, default=100_000)
    parser.add_argument("--replications", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--output-dir", type=Path, default=Path("results/targets"))
    args = parser.parse_args()

    specification = get_scenario(args.scenario)
    study = approximate_cml_target(
        specification,
        reference_size=args.reference_size,
        replications=args.replications,
        seed=args.seed,
        fit_kwargs={"max_iter": 160, "tol": 1e-6, "n_jobs": 1},
    )
    model_dimension = study.estimates.shape[1]
    names = [f"coordinate_{index}" for index in range(model_dimension)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    estimates_path = args.output_dir / f"{args.scenario}_estimates.csv"
    summary_path = args.output_dir / f"{args.scenario}_summary.csv"
    estimate_frame = pd.DataFrame(study.estimates, columns=names).assign(
        objective=study.objectives
    )
    estimate_frame = pd.concat(
        [estimate_frame, pd.DataFrame(study.fit_diagnostics).add_prefix("fit_")],
        axis=1,
    )
    estimate_frame.to_csv(estimates_path, index=False)
    pd.DataFrame(
        {
            "scenario": args.scenario,
            "coordinate": np.arange(model_dimension),
            "coordinate_name": names,
            "target": study.mean,
            "target_mcse": study.monte_carlo_standard_error,
            "reference_size": args.reference_size,
            "successful_references": len(study.estimates),
            "failed_references": len(study.failures),
        }
    ).to_csv(summary_path, index=False)
    print(pd.read_csv(summary_path).to_string(index=False))
    if study.failures:
        print("\nFailures:")
        print("\n".join(study.failures))
    print(f"\nSaved {estimates_path}\nSaved {summary_path}")


if __name__ == "__main__":
    main()
