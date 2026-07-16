"""Run one reproducible local or global bootstrap chunk for Dry Bean data."""

from __future__ import annotations

import argparse
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from cem_inference import NormalAdapter, StudentTAdapter, fit_fmvmm_hard, optimize_local_cml
from run_real_data_screen import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--p", type=int, default=2)
    parser.add_argument("--mode", choices=("local", "global"), required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--replications", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    raw, _ = load_dataset(args.dataset)
    observations = PCA(n_components=args.p, svd_solver="full").fit_transform(
        StandardScaler().fit_transform(raw)
    )
    adapters = [NormalAdapter(args.p), StudentTAdapter(args.p)]
    fit = fit_fmvmm_hard(observations, adapters, max_iter=160, tol=1e-6)
    rows = []
    for replication in range(args.start, args.start + args.replications):
        seed = np.random.SeedSequence([
            args.seed, zlib.crc32(args.dataset.encode()), replication,
            0 if args.mode == "local" else 1,
        ])
        rng = np.random.default_rng(seed)
        sample = observations[rng.integers(0, len(observations), len(observations))]
        started = time.perf_counter()
        try:
            if args.mode == "local":
                result = optimize_local_cml(
                    sample, fit.model, fit.coordinates,
                    trust_radius=0.75, max_iter=250,
                )
                if not result.success:
                    raise RuntimeError(
                        f"local optimization: {result.message}; "
                        f"hit_boundary={result.hit_trust_boundary}"
                    )
                coordinates = result.coordinates
                objective = result.objective
                metadata = {"hit_trust_boundary": result.hit_trust_boundary}
            else:
                refit = fit_fmvmm_hard(sample, adapters, max_iter=160, tol=1e-6)
                coordinates = refit.coordinates
                objective = refit.classification_objective
                metadata = {
                    "fit_initialization": refit.initialization,
                    "fit_objective_gap": refit.objective_gap,
                    "fit_minimum_weight": refit.weights.min(),
                    "fit_selected_admissible": refit.selected_admissible,
                }
            for coordinate, value in enumerate(coordinates):
                rows.append({
                    "dataset": args.dataset, "p": args.p, "mode": args.mode,
                    "replication": replication, "status": "success",
                    "coordinate": coordinate,
                    "coordinate_name": fit.model.coordinate_names[coordinate],
                    "estimate": value, "objective": objective,
                    "seconds": time.perf_counter() - started, **metadata,
                })
        except Exception as error:
            rows.append({
                "dataset": args.dataset, "p": args.p, "mode": args.mode,
                "replication": replication, "status": "failed",
                "error_message": f"{type(error).__name__}: {error}",
                "seconds": time.perf_counter() - started,
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output, index=False)
    print(frame.groupby("status").replication.nunique())
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
