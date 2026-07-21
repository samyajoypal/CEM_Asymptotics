"""Build compact LaTeX tables for the supplementary material."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "appendix" / "empirical_tables.tex"


def esc(value: object) -> str:
    return str(value).replace("_", r"\_\allowbreak{}")


def scenario_name(value: str) -> str:
    parts = value.split("_")
    family = {"nt": "Normal--Student-$t$", "nsn": "Normal--skew-normal",
              "ntsn": "Normal--Student-$t$--skew-normal"}[parts[0]]
    dimension = parts[1].removeprefix("p")
    regime = {"moderate": "moderate overlap", "strong": "strong overlap",
              "separated": "well separated"}[parts[2]]
    balance = "imbalanced weights" if "imbalanced" in parts else "balanced weights"
    return f"{family}; dimension {dimension}; {regime}; {balance}"


def dataset_name(value: str) -> str:
    return {"breast_cancer": "Breast cancer", "wine": "Wine",
            "digits_0_1": "Handwritten digits 0 and 1",
            "dry_bean_cali_barbunya": "Dry Bean: Cali--Barbunya",
            "dry_bean_dermason_seker": "Dry Bean: Dermason--Seker",
            "dry_bean_horoz_sira": "Dry Bean: Horoz--Sira"}.get(value, value)


def coordinate_name(value: str) -> str:
    if value.startswith("weight_logit"):
        return r"Log odds of component weights, $\log(\pi_1/\pi_2)$"
    prefix, index = value.rsplit("_coordinate_", 1)
    component = "Gaussian component" if "component_1_normal" in prefix else r"Student-$t$ component"
    labels = {"0": "location, principal component 1",
              "1": "location, principal component 2",
              "2": "log-Cholesky diagonal 1",
              "3": "Cholesky off-diagonal",
              "4": "log-Cholesky diagonal 2",
              "5": r"log degrees-of-freedom excess, $\log(\nu-2)$"}
    return f"{component}: {labels.get(index, 'parameter ' + index)}"


def performance_rows(path: Path, label: str) -> list[str]:
    frame = pd.read_csv(path)
    rows = [rf"\multicolumn{{7}}{{l}}{{\textit{{{label}}}}}\\"]
    for _, row in frame.iterrows():
        method = "Corrected" if row.method == "corrected_stabilized" else "Naive"
        rows.append(
            f"{scenario_name(row.scenario)} & {int(row.n)} & {method} & "
            f"{int(row.usable_replications)} & {row.median_se_to_sd:.3f} & "
            f"{row.median_coverage:.3f} & {row.stabilization_frequency:.3f} \\\\"
        )
    return rows


def parameter_table(path: Path, title: str, label: str) -> list[str]:
    frame = pd.read_csv(path)
    rows = [
        r"\begingroup\scriptsize\setlength{\tabcolsep}{2pt}",
        r"\begin{longtable}{p{0.30\linewidth}rrrrrr}",
        rf"\caption{{{title} Here s.e. denotes standard error, s.d. denotes standard deviation, and each ratio divides the analytic standard error by the local-bootstrap standard deviation.}}\label{{{label}}}\\",
        r"\toprule",
        r"Parameter & Estimate & Naive s.e. & Corrected s.e. & Bootstrap s.d. & Naive ratio & Corrected ratio\\",
        r"\midrule\endfirsthead",
        r"\toprule",
        r"Parameter & Estimate & Naive s.e. & Corrected s.e. & Bootstrap s.d. & Naive ratio & Corrected ratio\\",
        r"\midrule\endhead",
    ]
    for _, row in frame.iterrows():
        rows.append(
            f"{coordinate_name(row.coordinate_name)} & {row.estimate:.3f} & {row.naive_se:.3f} & "
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
        r"\caption{Complete scenario-level simulation summary. The standard-error ratio is the estimated standard error divided by the empirical standard deviation; coverage and ratios are coordinate medians. Stabilization is the fraction of usable fits for which the eigenvalue constraint was active.}\label{tab:s-full-performance}\\",
        r"\toprule",
        r"Scenario & Sample size & Method & Fits & S.e. ratio & Coverage & Stabilization\\",
        r"\midrule\endfirsthead",
        r"\toprule",
        r"Scenario & Sample size & Method & Fits & S.e. ratio & Coverage & Stabilization\\",
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
            f"{scenario_name(row.scenario)} & {int(row.n)} & {int(row.planned_replications)} & "
            f"{int(row.fit_failures)} & {int(row.replications_with_any_raw_failure)} \\\\"
        )
    rows += [r"\bottomrule", r"\end{longtable}", r"\endgroup", ""]

    screen = pd.read_csv(ROOT / "results/processed/real_data/screen.csv")
    screen = screen[["dataset", "family", "p", "n", "status",
                     "ari_external_labels", "minimum_kernel_support",
                     "median_se_inflation", "boundary_shrinkage_factor"]]
    rows += [
        r"\begingroup\tiny\setlength{\tabcolsep}{1pt}",
        r"\begin{longtable}{p{0.22\linewidth}p{0.15\linewidth}rrrrrr}",
        r"\caption{Complete prespecified real-data screen. The adjusted Rand index measures agreement with external labels; kernel count is the number of observations in the boundary tube; standard-error inflation is the corrected-to-naive median ratio; stabilization is the retained boundary-curvature fraction. Dashes denote failed covariance summaries.}\label{tab:s-screen}\\",
        r"\toprule",
        r"Data set & Family & Dimension & Sample size & Rand index & Kernel count & S.e. inflation & Stabilization\\",
        r"\midrule\endfirsthead",
        r"\toprule",
        r"Data set & Family & Dimension & Sample size & Rand index & Kernel count & S.e. inflation & Stabilization\\",
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
        family = ("Normal--Student-$t$" if row.family == "normal_t"
                  else "Normal--skew-normal")
        rows.append(
            f"{dataset_name(row.dataset)} & {family} & {int(row.p)} & {int(row.n)} & "
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
