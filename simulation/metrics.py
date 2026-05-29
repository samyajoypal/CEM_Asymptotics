# simulation/metrics.py

import numpy as np
import pandas as pd


# ============================================================
# Basic metrics
# ============================================================

def compute_bias(
    estimates,
    truth,
):
    """
    Monte Carlo bias.
    """
    diff = estimates - truth

    diff = diff[np.isfinite(diff)]

    if len(diff) == 0:

        return np.nan

    return np.mean(diff)


def compute_rmse(
    estimates,
    truth,
):
    """
    Monte Carlo RMSE.
    """

    err = (estimates - truth) ** 2

    err = err[np.isfinite(err)]

    if len(err) == 0:

        return np.nan

    return np.sqrt(
        np.mean(err)
    )

def compute_empirical_sd(
    estimates,
):
    """
    Empirical standard deviation.
    """

    if len(estimates) <= 1:

        return np.nan

    estimates = estimates[
        np.isfinite(estimates)
    ]

    if len(estimates) <= 1:

        return np.nan

    return np.std(
        estimates,
        ddof=1,
    )


def compute_mean_se(
    ses,
):
    """
    Average estimated standard error.
    """
    ses = ses[np.isfinite(ses)]

    if len(ses) == 0:

        return np.nan

    return np.mean(ses)


def compute_coverage(
    lower,
    upper,
    truth,
):
    """
    Empirical confidence interval coverage.
    """

    covered = (

        (truth >= lower)
        &
        (truth <= upper)
    )

    covered = covered[np.isfinite(covered)]

    if len(covered) == 0:

        return np.nan

    return np.mean(covered)


# ============================================================
# Confidence intervals
# ============================================================

def build_confidence_intervals(
    estimates,
    ses,
    z=1.959963984540054,
):
    """
    Wald confidence intervals.
    """

    lower = estimates - z * ses

    upper = estimates + z * ses

    return lower, upper


# ============================================================
# Summarize one parameter
# ============================================================

def summarize_parameter(
    estimates,
    truth,
    cml_target,
    naive_ses,
    bc_ses,
):
    """
    Build summary statistics for one parameter.
    """

    # --------------------------------------------------------
    # Remove invalid entries
    # --------------------------------------------------------

    valid_mask = (

    np.isfinite(estimates)
    &
    np.isfinite(naive_ses)
    &
    np.isfinite(bc_ses)
    &
    (naive_ses > 0)
    &
    (bc_ses > 0)
)

    estimates = estimates[valid_mask]

    naive_ses = naive_ses[valid_mask]

    bc_ses = bc_ses[valid_mask]

    if len(estimates) == 0:

        return {

            "bias": np.nan,

            "rmse": np.nan,

            "empirical_sd": np.nan,

            "mean_naive_se": np.nan,

            "mean_bc_se": np.nan,

            "naive_coverage": np.nan,

            "bc_coverage": np.nan,
        }

    # --------------------------------------------------------
    # Confidence intervals
    # --------------------------------------------------------

    naive_lower, naive_upper = (

        build_confidence_intervals(

            estimates,

            naive_ses,
        )
    )

    bc_lower, bc_upper = (

        build_confidence_intervals(

            estimates,

            bc_ses,
        )
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    out = {

        "bias":
            compute_bias(
                estimates,
                truth,
            ),

        "rmse":
            compute_rmse(
                estimates,
                truth,
            ),

        "empirical_sd":
            compute_empirical_sd(
                estimates,
            ),

        "mean_naive_se":
            compute_mean_se(
                naive_ses,
            ),

        "mean_bc_se":
            compute_mean_se(
                bc_ses,
            ),

        "naive_coverage":
            compute_coverage(

                naive_lower,

                naive_upper,

                truth,
            ),

        "bc_coverage":
            compute_coverage(

                bc_lower,

                bc_upper,

                truth,
            ),
            
        "bias_cml":
            compute_bias(
                estimates,
                cml_target,
            ) if np.isfinite(cml_target) else np.nan,
            
        "naive_coverage_cml":
            compute_coverage(
                naive_lower,
                naive_upper,
                cml_target,
            ) if np.isfinite(cml_target) else np.nan,
            
        "bc_coverage_cml":
            compute_coverage(
                bc_lower,
                bc_upper,
                cml_target,
            ) if np.isfinite(cml_target) else np.nan,
    }

    return out


# ============================================================
# Main summary routine
# ============================================================

def summarize_results(
    raw_results,
):
    """
    Summarize Monte Carlo simulation results.
    """

    # --------------------------------------------------------
    # Calculate group-level metrics
    # --------------------------------------------------------

    group_metrics = {}
    group_cols = [
        "family",
        "scenario_name",
        "n",
    ]

    if "status" in raw_results.columns:
        for keys, g in raw_results.groupby(group_cols):
            n_total = g["replication"].nunique()
            n_success = g[g["status"] == "success"]["replication"].nunique()
            conv_rate = n_success / n_total if n_total > 0 else 0.0
            if "compute_time" in g.columns:
                mean_time = g.drop_duplicates(subset=["replication"])["compute_time"].mean()
            else:
                mean_time = np.nan
            group_metrics[keys] = {
                "convergence_rate": conv_rate,
                "mean_compute_time": mean_time,
            }

    # --------------------------------------------------------
    # Keep successful runs only
    # --------------------------------------------------------

    if "status" in raw_results.columns:

        raw_results = raw_results[
            raw_results["status"] == "success"
        ].copy()

    rows = []

    # ========================================================
    # Group by:
    #   family
    #   scenario
    #   sample size
    # ========================================================

    group_cols = [

        "family",

        "scenario_name",

        "n",
    ]

    grouped = raw_results.groupby(
        group_cols
    )

    # ========================================================
    # Process each simulation setting
    # ========================================================

    for keys, g in grouped:

        family, scenario_name, n = keys

        # ----------------------------------------------------
        # Parameter columns ONLY from theta_
        # ----------------------------------------------------

        theta_cols = [

            c for c in g.columns

            if c.startswith("theta_")
        ]

        theta_cols = sorted(theta_cols)

        # ----------------------------------------------------
        # Loop through parameters
        # ----------------------------------------------------

        for theta_col in theta_cols:

            parameter = theta_col.replace(
                "theta_",
                "",
            )

            true_col = f"true_{parameter}"

            cml_col = f"cml_target_{parameter}"

            naive_se_col = (
                f"naive_se_{parameter}"
            )

            bc_se_col = (
                f"bc_se_{parameter}"
            )

            # ------------------------------------------------
            # Skip if required columns missing
            # ------------------------------------------------

            required_cols = [

                theta_col,

                true_col,
                
                cml_col,

                naive_se_col,

                bc_se_col,
            ]

            if not all(
                c in g.columns
                for c in required_cols
            ):

                continue

            # ------------------------------------------------
            # Extract values
            # ------------------------------------------------

            estimates = pd.to_numeric(

                g[theta_col],

                errors="coerce",
            ).values

            truths = pd.to_numeric(

                g[true_col],

                errors="coerce",
            ).values
            
            cml_targets = pd.to_numeric(

                g[cml_col],

                errors="coerce",
            ).values

            naive_ses = pd.to_numeric(

                g[naive_se_col],

                errors="coerce",
            ).values

            bc_ses = pd.to_numeric(

                g[bc_se_col],

                errors="coerce",
            ).values

            # ------------------------------------------------
            # Remove NaNs
            # ------------------------------------------------

            valid_mask = (

                np.isfinite(estimates)
                &
                np.isfinite(truths)
                &
                np.isfinite(naive_ses)
                &
                np.isfinite(bc_ses)
            )

            estimates = estimates[valid_mask]

            truths = truths[valid_mask]
            
            cml_targets = cml_targets[valid_mask]

            naive_ses = naive_ses[valid_mask]

            bc_ses = bc_ses[valid_mask]

            # ------------------------------------------------
            # Skip empty groups
            # ------------------------------------------------

            if len(estimates) == 0:

                continue

            # ------------------------------------------------
            # Truth
            # ------------------------------------------------

            truth = float(truths[0])
            
            cml_target = float(cml_targets[0])

            # ------------------------------------------------
            # Metrics
            # ------------------------------------------------

            metrics = summarize_parameter(

                estimates=estimates,

                truth=truth,
                
                cml_target=cml_target,

                naive_ses=naive_ses,

                bc_ses=bc_ses,
            )

            # ------------------------------------------------
            # Store row
            # ------------------------------------------------

            row = {

                "family":
                    family,

                "scenario_name":
                    scenario_name,

                "n":
                    n,

                "parameter":
                    parameter,

                "truth":
                    truth,
            }

            if keys in group_metrics:
                row.update(group_metrics[keys])

            row.update(metrics)

            rows.append(row)

    summary_df = pd.DataFrame(rows)

    return summary_df


# ============================================================
# Monte Carlo diagnostics
# ============================================================

def simulation_diagnostics(
    raw_results,
):
    """
    Basic simulation diagnostics.
    """

    diagnostics = {}

    # --------------------------------------------------------
    # Status counts
    # --------------------------------------------------------

    if "status" in raw_results.columns:

        status_counts = raw_results[
            "status"
        ].value_counts()

        diagnostics["status_counts"] = (

            status_counts.to_dict()
        )

    # --------------------------------------------------------
    # Success counts
    # --------------------------------------------------------

    if "status" in raw_results.columns:

        diagnostics["n_success"] = int(

            np.sum(
                raw_results["status"] == "success"
            )
        )

    diagnostics["n_total"] = len(
        raw_results
    )

    # --------------------------------------------------------
    # Success rate
    # --------------------------------------------------------

    if diagnostics["n_total"] > 0:

        diagnostics["success_rate"] = (

            diagnostics.get(
                "n_success",
                diagnostics["n_total"],
            )
            /
            diagnostics["n_total"]
        )

    return diagnostics


# ============================================================
# Pretty print diagnostics
# ============================================================

def print_simulation_diagnostics(
    diagnostics,
):
    """
    Pretty-print diagnostics.
    """

    print("\nSimulation Diagnostics")
    print("=" * 60)

    for k, v in diagnostics.items():

        print(f"{k}: {v}")

    print("=" * 60)
