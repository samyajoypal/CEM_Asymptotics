"""Build compact LaTeX tables for the supplementary material."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "appendix" / "empirical_tables.tex"


def esc(value: object) -> str:
    return str(value).replace("_", r"\_\allowbreak{}")


def performance_rows(path: Path, label: str) -> list[str]:
    frame = pd.read_csv(path)
    rows = [rf"\multicolumn{{7}}{{l}}{{\textit{{{label}}}}}\\"]
    for _, row in frame.iterrows():
        method = "Corrected" if row.method == "corrected_stabilized" else "Naive"
        rows.append(
            f"{esc(row.scenario)} & {int(row.n)} & {method} & "
            f"{int(row.usable_replications)} & {row.median_se_to_sd:.3f} & "
            f"{row.median_coverage:.3f} & {row.stabilization_frequency:.3f} \\\\"
        )
    return rows


def parameter_table(path: Path, title: str, label: str) -> list[str]:
    frame = pd.read_csv(path)
    rows = [
        r"\begingroup\scriptsize\setlength{\tabcolsep}{2pt}",
        r"\begin{longtable}{p{0.30\linewidth}rrrrrr}",
        rf"\caption{{{title}}}\label{{{label}}}\\",
        r"\toprule",
        r"Coordinate & Estimate & Naive SE & Corrected SE & Local SD & Naive/SD & Corrected/SD\\",
        r"\midrule\endfirsthead",
        r"\toprule",
        r"Coordinate & Estimate & Naive SE & Corrected SE & Local SD & Naive/SD & Corrected/SD\\",
        r"\midrule\endhead",
    ]
    for _, row in frame.iterrows():
        rows.append(
            f"{esc(row.coordinate_name)} & {row.estimate:.3f} & {row.naive_se:.3f} & "
            f"{row.stabilized_se:.3f} & {row.local_bootstrap_sd:.3f} & "
            f"{row.naive_to_local_bootstrap:.3f} & "
            f"{row.stabilized_to_local_bootstrap:.3f} \\\\"
        )
    rows.extend([r"\bottomrule", r"\end{longtable}", r"\endgroup"])
    return rows


def main() -> None:
    confirmatory = ROOT / "results/processed/confirmatory_v2/tables"
    extension = ROOT / "results/processed/asymptotic_extension/tables"
    real = ROOT / "results/processed/real_data/final"
    rows = [
        r"\begingroup\scriptsize\setlength{\tabcolsep}{2pt}",
        r"\begin{longtable}{p{0.27\linewidth}rrlrrr}",
        r"\caption{Complete scenario-level simulation summary. SE/SD and coverage are coordinate medians; stabilization is the fraction of usable fits for which the eigenvalue constraint was active.}\label{tab:s-full-performance}\\",
        r"\toprule",
        r"Scenario & $n$ & Method & Usable & SE/SD & Coverage & Stabilization\\",
        r"\midrule\endfirsthead",
        r"\toprule",
        r"Scenario & $n$ & Method & Usable & SE/SD & Coverage & Stabilization\\",
        r"\midrule\endhead",
    ]
    rows += performance_rows(confirmatory / "scenario_performance.csv", "Primary study")
    rows += performance_rows(extension / "scenario_performance.csv", "Large-sample extension")
    rows += [r"\bottomrule", r"\end{longtable}", r"\endgroup", ""]

    failures = pd.read_csv(confirmatory / "fit_failure_rates.csv").merge(
        pd.read_csv(confirmatory / "raw_curvature_failures.csv"),
        on=["scenario", "n"], how="left",
    )
    rows += [
        r"\begingroup\scriptsize\setlength{\tabcolsep}{3pt}",
        r"\begin{longtable}{p{0.43\linewidth}rrrr}",
        r"\caption{Complete primary-study failure accounting. Raw curvature failures count replications in which at least one bandwidth produced a nonpositive unmodified curvature.}\label{tab:s-failures}\\",
        r"\toprule",
        r"Scenario & $n$ & Planned & Fit failures & Raw failures\\",
        r"\midrule\endfirsthead",
        r"\toprule",
        r"Scenario & $n$ & Planned & Fit failures & Raw failures\\",
        r"\midrule\endhead",
    ]
    for _, row in failures.iterrows():
        rows.append(
            f"{esc(row.scenario)} & {int(row.n)} & {int(row.planned_replications)} & "
            f"{int(row.fit_failures)} & {int(row.replications_with_any_raw_failure)} \\\\"
        )
    rows += [r"\bottomrule", r"\end{longtable}", r"\endgroup", ""]

    screen = pd.read_csv(ROOT / "results/processed/real_data/screen.csv")
    screen = screen[["dataset", "family", "p", "n", "status",
                     "ari_external_labels", "minimum_kernel_support",
                     "median_se_inflation", "boundary_shrinkage_factor"]]
    rows += [
        r"\begingroup\scriptsize\setlength{\tabcolsep}{2pt}",
        r"\begin{longtable}{p{0.22\linewidth}lrrrrrr}",
        r"\caption{Complete prespecified real-data screen. Dashes denote fits that failed before a valid covariance summary was obtained.}\label{tab:s-screen}\\",
        r"\toprule",
        r"Data & Family & $p$ & $n$ & ARI & Support & Inflation & Shrinkage\\",
        r"\midrule\endfirsthead",
        r"\toprule",
        r"Data & Family & $p$ & $n$ & ARI & Support & Inflation & Shrinkage\\",
        r"\midrule\endhead",
    ]
    for _, row in screen.iterrows():
        if row.status == "success":
            metrics = (f"{row.ari_external_labels:.3f} & "
                       f"{int(row.minimum_kernel_support)} & "
                       f"{row.median_se_inflation:.3f} & "
                       f"{row.boundary_shrinkage_factor:.3f}")
        else:
            metrics = r"-- & -- & -- & --"
        family = "Normal--$t$" if row.family == "normal_t" else "Normal--SN"
        rows.append(
            f"{esc(row.dataset)} & {family} & {int(row.p)} & {int(row.n)} & "
            f"{metrics} \\\\"
        )
    rows += [r"\bottomrule", r"\end{longtable}", r"\endgroup", ""]

    rows += parameter_table(
        real / "dry_bean_dermason_seker_parameters.csv",
        "Parameter-level results for Dermason--Seker.",
        "tab:s-dermason-parameters",
    )
    rows += [""]
    rows += parameter_table(
        real / "dry_bean_cali_barbunya_parameters.csv",
        "Parameter-level results for Cali--Barbunya.",
        "tab:s-cali-parameters",
    )
    OUT.write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
