# simulation/tables.py

from pathlib import Path

import pandas as pd

import numpy as np


# ============================================================
# Numeric formatting
# ============================================================

def fmt(x, digits=3):
    """
    Safe numeric formatter.
    """

    if pd.isna(x):

        return ""

    if np.abs(x) > 1e4:

        return f"{x:.2e}"

    return f"{x:.{digits}f}"


# ============================================================
# Clean parameter labels
# ============================================================

# ============================================================
# Family-specific parameter labels
# Notation matches the manuscript exactly:
#
#   Normal-t  : rho, mu_1, vech(L_1), mu_2, vech(L_2), lambda
#   Normal-SN : rho, mu_1, vech(L_1), xi_2, vech(L_2), alpha_2
#
# Subscript convention: first index = component, second = coordinate
# e.g. mu_{1,1} is the first coordinate of the component-1 mean.
# ============================================================

# --------------------------------------------------------
# Shared parameters (identical meaning in both families)
# --------------------------------------------------------

_SHARED_LABELS = {

    # log-odds mixing proportion: pi_1 = exp(rho)/(1+exp(rho))
    "rho":   r"$\rho$",

    # Component-1 Gaussian mean (boldsymbol mu_1 in manuscript)
    "mu1_1": r"$\mu_{1,1}$",
    "mu1_2": r"$\mu_{1,2}$",

    # Cholesky entries of Sigma_1 = L_1 L_1^T
    "l11_1": r"$\ell_{11,1}$",
    "l21_1": r"$\ell_{21,1}$",
    "l22_1": r"$\ell_{22,1}$",

    # Cholesky entries of Sigma_2 / Omega_2
    "l11_2": r"$\ell_{11,2}$",
    "l21_2": r"$\ell_{21,2}$",
    "l22_2": r"$\ell_{22,2}$",
}

# --------------------------------------------------------
# Normal-t specific parameters
# --------------------------------------------------------

PARAMETER_LABELS_NORMAL_T = {

    **_SHARED_LABELS,

    # Component-2 Student-t mean
    "mu2_1": r"$\mu_{2,1}$",
    "mu2_2": r"$\mu_{2,2}$",

    # Degrees-of-freedom: nu = 2 + exp(lambda)
    "lambda": r"$\lambda$",
}

# --------------------------------------------------------
# Normal-SN specific parameters
# --------------------------------------------------------

PARAMETER_LABELS_NORMAL_SN = {

    **_SHARED_LABELS,

    # Component-2 skew-normal location
    "xi2_1":  r"$\xi_{2,1}$",
    "xi2_2":  r"$\xi_{2,2}$",

    # Skewness vector alpha_2 in manuscript
    "alpha1": r"$\alpha_{2,1}$",
    "alpha2": r"$\alpha_{2,2}$",
}

# --------------------------------------------------------
# Canonical parameter ordering
# --------------------------------------------------------

PARAM_ORDER_NORMAL_T = [
    "rho",
    "mu1_1", "mu1_2",
    "l11_1", "l21_1", "l22_1",
    "mu2_1", "mu2_2",
    "l11_2", "l21_2", "l22_2",
    "lambda",
]

PARAM_ORDER_NORMAL_SN = [
    "rho",
    "mu1_1", "mu1_2",
    "l11_1", "l21_1", "l22_1",
    "xi2_1", "xi2_2",
    "l11_2", "l21_2", "l22_2",
    "alpha1", "alpha2",
]


def get_param_labels(family):
    """
    Return the parameter-label dict and ordering list
    for a given family string.
    """

    if family == "normal_t":
        return PARAMETER_LABELS_NORMAL_T, PARAM_ORDER_NORMAL_T

    elif family == "normal_skewnormal":
        return PARAMETER_LABELS_NORMAL_SN, PARAM_ORDER_NORMAL_SN

    else:
        # Fallback: return shared labels
        return _SHARED_LABELS, list(_SHARED_LABELS.keys())


# ============================================================
# Publication summary table
# ============================================================

def build_publication_table(
    summary_df,
    family=None,
):
    """
    Construct publication-ready summary table, pivoted by sample size n.
    Rows: Parameter, then Metric.
    Columns: Sample sizes.

    Parameters
    ----------
    summary_df : DataFrame
    family : str or None
        If provided, use family-specific parameter labels and ordering.
    """

    # Resolve family-specific labels and ordering
    if family is None:
        # Try to infer from data
        families = summary_df["family"].unique() if "family" in summary_df.columns else []
        family = families[0] if len(families) == 1 else None

    param_labels, ordered_params = get_param_labels(family)

    grouped = summary_df.groupby(["parameter", "n"])

    data_dict = {}

    for (param, n), g in grouped:
        row = g.iloc[0]

        bias = fmt(row["bias"])
        bias_cml = fmt(row["bias_cml"]) if "bias_cml" in row else ""
        rmse = fmt(row["rmse"])
        se = fmt(row["mean_bc_se"])
        cov = fmt(100 * row["bc_coverage"]) + r"\%" if pd.notna(row["bc_coverage"]) else ""
        cov_cml = fmt(100 * row["bc_coverage_cml"]) + r"\%" if "bc_coverage_cml" in row and pd.notna(row["bc_coverage_cml"]) else ""
        time_s = fmt(row.get("mean_compute_time", np.nan), digits=1)
        conv = fmt(100 * row.get("convergence_rate", np.nan), digits=1) + r"\%" if pd.notna(row.get("convergence_rate", np.nan)) else ""

        data_dict[(param, n)] = {
            "Bias": bias,
            "Bias (CML)": bias_cml,
            "RMSE": rmse,
            "BCSE": se,
            "Cov. Prob.": cov,
            "Cov. Prob. (CML)": cov_cml,
            "Time (s)": time_s,
            "Conv": conv,
        }

    metrics = ["Bias", "Bias (CML)", "RMSE", "BCSE", "Cov. Prob.", "Cov. Prob. (CML)", "Time (s)", "Conv"]

    unique_ns = sorted(summary_df["n"].unique())

    final_rows = []

    present_params = summary_df["parameter"].unique()
    sorted_params = [p for p in ordered_params if p in present_params]

    for param in sorted_params:
        param_label = param_labels.get(param, param)

        for i, metric in enumerate(metrics):
            display_param = param_label if i == 0 else ""
            row_dict = {
                "Parameter": display_param,
                "Metric": metric,
            }
            for n in unique_ns:
                key = (param, n)
                if key in data_dict:
                    row_dict[f"$n={n}$"] = data_dict[key][metric]
                else:
                    row_dict[f"$n={n}$"] = ""

            final_rows.append(row_dict)

    table_df = pd.DataFrame(final_rows)
    return table_df


# ============================================================
# Export CSV
# ============================================================

def export_csv_table(
    df,
    output_path,
):
    """
    Export CSV summary table.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(

        parents=True,

        exist_ok=True,
    )

    df.to_csv(

        output_path,

        index=False,
    )


# ============================================================
# Export LaTeX
# ============================================================

def export_latex_table(
    df,
    output_path,
    caption=None,
    label=None,
):
    """
    Export LaTeX table.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(

        parents=True,

        exist_ok=True,
    )

    latex = df.to_latex(

        index=False,

        escape=False,

        longtable=True,

        multicolumn=True,

        bold_rows=False,
    )

    # --------------------------------------------------------
    # Add caption and label
    # --------------------------------------------------------

    if caption is not None:

        latex = latex.replace(

            r"\begin{longtable}",

            (
                r"\begin{longtable}" "\n"
                rf"\caption{{{caption}}}" "\n"
                rf"\label{{{label}}}"
            )
        )

    with open(output_path, "w") as f:

        f.write(latex)


# ============================================================
# Split tables by family
# ============================================================

def export_family_tables(
    summary_df,
    output_dir,
):
    """
    Export separate tables for each family and scenario pair.
    """

    output_dir = Path(output_dir)

    output_dir.mkdir(

        parents=True,

        exist_ok=True,
    )

    grouped = summary_df.groupby(["family", "scenario_name"])

    for (family, scenario_name), sub_df in grouped:

        table_df = build_publication_table(
            sub_df,
            family=family,
        )

        prefix = f"{family}_{scenario_name}"

        # ----------------------------------------------------
        # CSV
        # ----------------------------------------------------

        export_csv_table(

            table_df,

            output_dir / f"{prefix}_summary.csv",
        )

        # ----------------------------------------------------
        # LaTeX
        # ----------------------------------------------------
        
        scenario_clean = scenario_name.replace("_", " ").title()
        family_clean = family.replace("_", "-").title()

        export_latex_table(

            table_df,

            output_dir / f"{prefix}_summary.tex",

            caption=(
                f"Simulation results for "
                f"{family_clean} mixtures under {scenario_clean} scenario."
            ),

            label=(
                f"tab:{prefix}_simulation"
            ),
        )


# ============================================================
# Full combined table
# ============================================================

def export_full_summary_table(
    summary_df,
    output_dir,
):
    """
    Export raw simulation summary to CSV.
    """

    output_dir = Path(output_dir)

    output_dir.mkdir(

        parents=True,

        exist_ok=True,
    )

    export_csv_table(

        summary_df,

        output_dir / "full_summary.csv",
    )

