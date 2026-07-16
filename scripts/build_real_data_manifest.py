"""Build the prespecified Dry Bean bootstrap manifest."""

from pathlib import Path
import pandas as pd


def main():
    rows = []
    for dataset in ("dry_bean_dermason_seker", "dry_bean_cali_barbunya"):
        for mode, total in (("local", 500), ("global", 200)):
            for start in range(0, total, 50):
                rows.append({
                    "dataset": dataset, "p": 2, "mode": mode,
                    "start": start, "replications": 50,
                    "output": (
                        f"results/raw/real_data/{dataset}/"
                        f"{mode}_start{start:04d}.csv"
                    ),
                })
    path = Path(__file__).resolve().parents[1] / "hpc/manifests/real_data_tasks.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Wrote {len(rows)} real-data tasks")


if __name__ == "__main__":
    main()
