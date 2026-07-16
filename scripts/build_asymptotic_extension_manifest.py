"""Build the post-confirmatory large-n convergence extension."""

from pathlib import Path

import pandas as pd


def main():
    project = Path(__file__).resolve().parents[1]
    cells = [
        ("nt_p2_strong_balanced", 8000),
        ("nsn_p2_strong_balanced", 8000),
        ("ntsn_p2_moderate_balanced", 4000),
        ("ntsn_p5_moderate_balanced", 4000),
    ]
    rows = []
    for scenario, sample_size in cells:
        for start in range(0, 500, 50):
            rows.append({
                "experiment": "asymptotic_extension",
                "scenario": scenario,
                "sample_size": sample_size,
                "start": start,
                "replications": 50,
                "target": f"results/targets/confirmatory_v2/{scenario}_summary.csv",
                "output": (
                    "results/raw/asymptotic_extension/"
                    f"{scenario}_n{sample_size}_start{start:04d}.csv"
                ),
            })
    path = project / "hpc/manifests/asymptotic_extension.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Wrote {len(rows)} tasks and {sum(row['replications'] for row in rows)} replications to {path}")


if __name__ == "__main__":
    main()
