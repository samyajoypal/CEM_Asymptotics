# simulation/diagnostics.py

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Basic simulation diagnostics
# ============================================================

def compute_basic_diagnostics(
    raw_results,
):
    """
    Compute overall simulation diagnostics.

    Parameters
    ----------
    raw_results : DataFrame

    Returns
    -------
    dict
    """

    diagnostics = {}

    # --------------------------------------------------------
    # Total runs
    # --------------------------------------------------------

    diagnostics["n_total"] = len(
        raw_results
    )

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

        diagnostics["n_success"] = int(

            np.sum(
                raw_results["status"] == "success"
            )
        )

        diagnostics["n_failed"] = int(

            np.sum(
                raw_results["status"] != "success"
            )
        )

        if diagnostics["n_total"] > 0:

            diagnostics["success_rate"] = (

                diagnostics["n_success"]
                /
                diagnostics["n_total"]
            )

    return diagnostics


# ============================================================
# Failure analysis
# ============================================================

def analyze_failures(
    raw_results,
):
    """
    Analyze failed replications.

    Returns
    -------
    DataFrame
    """

    if "status" not in raw_results.columns:

        return pd.DataFrame()

    failed = raw_results[
        raw_results["status"] != "success"
    ].copy()

    if len(failed) == 0:

        return pd.DataFrame()

    # --------------------------------------------------------
    # Keep unique failures
    # --------------------------------------------------------

    keep_cols = [

        "family",

        "scenario_name",

        "n",

        "replication",

        "error",
    ]

    keep_cols = [

        c for c in keep_cols

        if c in failed.columns
    ]

    failed = failed[
        keep_cols
    ].drop_duplicates()

    return failed


# ============================================================
# Parameter trajectory diagnostics
# ============================================================

def parameter_range_diagnostics(
    raw_results,
):
    """
    Check for exploding parameter estimates.

    Returns
    -------
    DataFrame
    """

    rows = []

    parameter_cols = [

        c for c in raw_results.columns

        if c.startswith("theta_")
    ]

    grouped = raw_results.groupby([

        "family",

        "scenario_name",

        "n",
    ])

    for keys, gdf in grouped:

        family, scenario_name, n = keys

        for param in parameter_cols:

            vals = gdf[param].dropna().values

            if len(vals) == 0:

                continue

            rows.append({

                "family":
                    family,

                "scenario_name":
                    scenario_name,

                "n":
                    n,

                "parameter":
                    param,

                "min":
                    np.min(vals),

                "max":
                    np.max(vals),

                "mean":
                    np.mean(vals),

                "sd":
                    np.std(vals),
            })

    return pd.DataFrame(rows)


# ============================================================
# Standard error diagnostics
# ============================================================

def standard_error_diagnostics(
    raw_results,
):
    """
    Diagnostics for SE stability.

    Returns
    -------
    DataFrame
    """

    rows = []

    naive_cols = [

        c for c in raw_results.columns

        if c.startswith("naive_se_")
    ]

    bc_cols = [

        c for c in raw_results.columns

        if c.startswith("bc_se_")
    ]

    all_cols = naive_cols + bc_cols

    grouped = raw_results.groupby([

        "family",

        "scenario_name",

        "n",
    ])

    for keys, gdf in grouped:

        family, scenario_name, n = keys

        for col in all_cols:

            vals = gdf[col].dropna().values

            if len(vals) == 0:

                continue

            rows.append({

                "family":
                    family,

                "scenario_name":
                    scenario_name,

                "n":
                    n,

                "quantity":
                    col,

                "min":
                    np.min(vals),

                "max":
                    np.max(vals),

                "mean":
                    np.mean(vals),

                "sd":
                    np.std(vals),
            })

    return pd.DataFrame(rows)


# ============================================================
# Covariance stability diagnostics
# ============================================================

def covariance_stability_checks(
    raw_results,
):
    """
    Detect unstable covariance estimates.

    Returns
    -------
    DataFrame
    """

    rows = []

    se_cols = [

        c for c in raw_results.columns

        if (
            c.startswith("naive_se_")
            or
            c.startswith("bc_se_")
        )
    ]

    grouped = raw_results.groupby([

        "family",

        "scenario_name",

        "n",
    ])

    for keys, gdf in grouped:

        family, scenario_name, n = keys

        for col in se_cols:

            vals = gdf[col].dropna().values

            if len(vals) == 0:

                continue

            rows.append({

                "family":
                    family,

                "scenario_name":
                    scenario_name,

                "n":
                    n,

                "quantity":
                    col,

                "n_large":
                    int(np.sum(vals > 100)),

                "n_small":
                    int(np.sum(vals < 1e-8)),

                "n_nan":
                    int(np.sum(np.isnan(vals))),

                "n_inf":
                    int(np.sum(np.isinf(vals))),
            })

    return pd.DataFrame(rows)


# ============================================================
# Export diagnostics
# ============================================================

def export_diagnostics(
    diagnostics_dict,
    output_dir,
):
    """
    Export diagnostics to CSV.
    """

    output_dir = Path(output_dir)

    output_dir.mkdir(

        parents=True,

        exist_ok=True,
    )

    for name, obj in diagnostics_dict.items():

        path = output_dir / f"{name}.csv"

        # ----------------------------------------------------
        # Dict
        # ----------------------------------------------------

        if isinstance(obj, dict):

            df = pd.DataFrame([

                {
                    "metric": k,
                    "value": str(v),
                }

                for k, v in obj.items()
            ])

            df.to_csv(
                path,
                index=False,
            )

        # ----------------------------------------------------
        # DataFrame
        # ----------------------------------------------------

        elif isinstance(obj, pd.DataFrame):

            obj.to_csv(
                path,
                index=False,
            )


# ============================================================
# Master diagnostics pipeline
# ============================================================

def run_full_diagnostics(
    raw_results,
    output_dir=None,
):
    """
    Run complete diagnostics suite.

    Returns
    -------
    dict
    """

    diagnostics = {

        "basic_diagnostics":
            compute_basic_diagnostics(
                raw_results
            ),

        "failure_analysis":
            analyze_failures(
                raw_results
            ),

        "parameter_ranges":
            parameter_range_diagnostics(
                raw_results
            ),

        "standard_error_diagnostics":
            standard_error_diagnostics(
                raw_results
            ),

        "covariance_stability":
            covariance_stability_checks(
                raw_results
            ),
    }

    # --------------------------------------------------------
    # Export
    # --------------------------------------------------------

    if output_dir is not None:

        export_diagnostics(

            diagnostics,

            output_dir=output_dir,
        )

    return diagnostics


# ============================================================
# Pretty printer
# ============================================================

def print_diagnostics_summary(
    diagnostics,
):
    """
    Pretty-print diagnostics summary.
    """

    print("\n")
    print("=" * 70)
    print("Simulation Diagnostics Summary")
    print("=" * 70)

    basic = diagnostics.get(
        "basic_diagnostics",
        {}
    )

    for k, v in basic.items():

        print(f"{k}: {v}")

    print("=" * 70)
