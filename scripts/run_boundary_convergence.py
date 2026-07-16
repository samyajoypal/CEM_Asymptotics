"""Controlled convergence experiment for the kernel boundary estimator.

For g(x; theta)=theta-x at theta=0 with X standard normal, the one-parameter
boundary curvature equals phi(0). This isolates the estimator from fitting and
tests the h^2 bias and (nh)^-1 variance predicted by the theory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from cem_inference import epanechnikov


def run_experiment(
    sample_sizes: list[int], replications: int, seed: int, multiplier: float
) -> pd.DataFrame:
    seed_sequence = np.random.SeedSequence(seed)
    child_seeds = seed_sequence.spawn(len(sample_sizes) * replications)
    target = norm.pdf(0.0)
    rows = []
    seed_index = 0
    for sample_size in sample_sizes:
        bandwidth = multiplier * sample_size ** -0.2
        for replication in range(replications):
            rng = np.random.default_rng(child_seeds[seed_index])
            seed_index += 1
            observations = rng.normal(size=sample_size)
            estimate = np.mean(epanechnikov(-observations / bandwidth)) / bandwidth
            rows.append(
                {
                    "n": sample_size,
                    "replication": replication,
                    "bandwidth": bandwidth,
                    "estimate": estimate,
                    "target": target,
                    "error": estimate - target,
                    "squared_error": (estimate - target) ** 2,
                }
            )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-sizes", nargs="+", type=int, default=[250, 1000, 4000, 16000])
    parser.add_argument("--replications", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--multiplier", type=float, default=1.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/processed/boundary_convergence.csv"),
    )
    args = parser.parse_args()
    result = run_experiment(
        args.sample_sizes, args.replications, args.seed, args.multiplier
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    summary = result.groupby("n").agg(
        bias=("error", "mean"),
        empirical_sd=("estimate", "std"),
        rmse=("squared_error", lambda values: np.sqrt(np.mean(values))),
        bandwidth=("bandwidth", "first"),
    )
    summary_path = args.output.with_name(f"{args.output.stem}_summary.csv")
    summary.reset_index().to_csv(summary_path, index=False)
    print(summary.to_string())
    print(f"\nSaved {args.output}")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
