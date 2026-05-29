# simulation/plots.py

from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.patches import Ellipse

from inference.boundary import (
    build_decision_grid,
    extract_boundary_contour,
)


# ============================================================
# Plot style
# ============================================================

plt.rcParams.update({

    "figure.figsize": (7, 6),

    "font.size": 12,

    "axes.titlesize": 13,

    "axes.labelsize": 12,

    "legend.fontsize": 10,

    "xtick.labelsize": 10,

    "ytick.labelsize": 10,
})

# ============================================================
# Parameters worth plotting
# ============================================================

# ============================================================
# Family-specific parameter labels for plots
# Notation matches the manuscript exactly.
# ============================================================

_PLOT_SHARED_LABELS = {
    "rho":   r"$\rho$",
    "mu1_1": r"$\mu_{1,1}$",
    "mu1_2": r"$\mu_{1,2}$",
    "l11_1": r"$\ell_{11,1}$",
    "l21_1": r"$\ell_{21,1}$",
    "l22_1": r"$\ell_{22,1}$",
    "l11_2": r"$\ell_{11,2}$",
    "l21_2": r"$\ell_{21,2}$",
    "l22_2": r"$\ell_{22,2}$",
}

PLOT_LABELS_NORMAL_T = {
    **_PLOT_SHARED_LABELS,
    "mu2_1": r"$\mu_{2,1}$",
    "mu2_2": r"$\mu_{2,2}$",
    "lambda": r"$\lambda$",
}

PLOT_LABELS_NORMAL_SN = {
    **_PLOT_SHARED_LABELS,
    "xi2_1":  r"$\xi_{2,1}$",
    "xi2_2":  r"$\xi_{2,2}$",
    "alpha1": r"$\alpha_{2,1}$",
    "alpha2": r"$\alpha_{2,2}$",
}

# All known parameter keys across both families
PLOT_PARAMETERS = list({
    **PLOT_LABELS_NORMAL_T,
    **PLOT_LABELS_NORMAL_SN,
}.keys())


def get_plot_label(parameter, family=None):
    """
    Return the mathtext label for a parameter, optionally family-specific.
    Falls back to the raw key name if unknown.
    """

    if family == "normal_t":
        return PLOT_LABELS_NORMAL_T.get(parameter, parameter)

    elif family == "normal_skewnormal":
        return PLOT_LABELS_NORMAL_SN.get(parameter, parameter)

    # No family specified: merge both (shared + union)
    merged = {**PLOT_LABELS_NORMAL_T, **PLOT_LABELS_NORMAL_SN}
    return merged.get(parameter, parameter)

# ============================================================
# Covariance ellipse
# ============================================================

def covariance_ellipse(
    mean,
    cov,
    ax,
    n_std=2.0,
    edgecolor="black",
    linewidth=2,
    linestyle="-",
):
    """
    Draw covariance ellipse.
    """

    eigvals, eigvecs = np.linalg.eigh(cov)

    order = eigvals.argsort()[::-1]

    eigvals = eigvals[order]

    eigvecs = eigvecs[:, order]

    angle = np.degrees(

        np.arctan2(
            eigvecs[1, 0],
            eigvecs[0, 0],
        )
    )

    width = 2 * n_std * np.sqrt(eigvals[0])

    height = 2 * n_std * np.sqrt(eigvals[1])

    ellipse = Ellipse(

        xy=mean,

        width=width,

        height=height,

        angle=angle,

        fill=False,

        edgecolor=edgecolor,

        linewidth=linewidth,

        linestyle=linestyle,
    )

    ax.add_patch(ellipse)


# ============================================================
# Plot simulated data
# ============================================================

def plot_simulated_data(
    X,
    labels,
    save_path=None,
    title=None,
):
    """
    Scatter plot of simulated mixture data.
    """

    fig, ax = plt.subplots()

    unique_labels = np.unique(labels)

    colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
    ]

    for k in unique_labels:

        idx = labels == k

        ax.scatter(

            X[idx, 0],

            X[idx, 1],

            s=15,

            alpha=0.7,

            color=colors[k],

            label=f"Component {k+1}",
        )

    ax.set_xlabel(r"$x_1$")

    ax.set_ylabel(r"$x_2$")

    if title is not None:

        ax.set_title(title)

    ax.legend()

    ax.grid(alpha=0.3)

    if save_path is not None:

        plt.savefig(

            save_path,

            bbox_inches="tight",

            dpi=300,
        )

    plt.close()


# ============================================================
# Plot decision boundary
# ============================================================

def plot_decision_boundary(
    X,
    family,
    pi_hat,
    alpha_hat,
    z_hat=None,
    save_path=None,
    title=None,
):
    """
    Plot heterogeneous CEM decision boundary.
    """

    fig, ax = plt.subplots()

    # --------------------------------------------------------
    # Plot observations
    # --------------------------------------------------------

    if z_hat is None:

        ax.scatter(

            X[:, 0],

            X[:, 1],

            s=12,

            alpha=0.7,
        )

    else:

        colors = [
    "#1f77b4",
    "#d62728",
]

        for k in [0, 1]:

            idx = z_hat == k

            ax.scatter(

                X[idx, 0],

                X[idx, 1],

                s=14,

                alpha=0.7,

                color=colors[k],

                label=f"Cluster {k+1}",
            )

    # --------------------------------------------------------
    # Build decision grid
    # --------------------------------------------------------

    grid_object = build_decision_grid(

        X=X,

        family=family,

        pi_hat=pi_hat,

        alpha_hat=alpha_hat,
    )

    # --------------------------------------------------------
    # Extract contour
    # --------------------------------------------------------

    contour = extract_boundary_contour(
        grid_object
    )

    # --------------------------------------------------------
    # Plot contour
    # --------------------------------------------------------

    if contour is not None:

        ax.plot(

            contour[:, 0],

            contour[:, 1],

            color="black",

            linewidth=2.5,

            label="Decision boundary",
        )

    # --------------------------------------------------------
    # Plot covariance ellipses
    # --------------------------------------------------------

    # Component 1

    mu1, Sigma1 = alpha_hat[0][:2]

    covariance_ellipse(

        mean=mu1,

        cov=Sigma1,

        ax=ax,

        edgecolor="tab:blue",
    )

    # Component 2

    mu2 = alpha_hat[1][0]

    Sigma2 = alpha_hat[1][1]

    covariance_ellipse(

        mean=mu2,

        cov=Sigma2,

        ax=ax,

        edgecolor="tab:orange",
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    ax.set_xlabel(r"$x_1$")

    ax.set_ylabel(r"$x_2$")

    if title is not None:

        ax.set_title(title)

    ax.legend()

    ax.grid(alpha=0.3)

    if save_path is not None:

        plt.savefig(

            save_path,

            bbox_inches="tight",

            dpi=300,
        )

    plt.close()


# ============================================================
# Bias plot
# ============================================================

def plot_bias_curves(
    summary_df,
    parameter,
    save_path=None,
):
    """
    Plot empirical bias versus sample size.
    """

    fig, ax = plt.subplots()

    grouped = summary_df[
    (summary_df["parameter"] == parameter)
    &
    np.isfinite(summary_df["bias"])
]

    for (family, scenario_name), gdf in grouped.groupby(
        ["family", "scenario_name"]
    ):

        ax.plot(

            gdf["n"],

            gdf["bias"],

            marker="o",

            linewidth=2,

            label=f"{family}: {scenario_name}",
        )

    ax.axhline(

        0.0,

        linestyle="--",

        color="black",
    )

    ax.set_xlabel("Sample size $n$")

    ax.set_ylabel("Bias")

    param_label = get_plot_label(parameter)
    ax.set_title(
        f"Bias of {param_label}"
    )

    ax.legend()

    ax.grid(alpha=0.3)

    if save_path is not None:

        plt.savefig(

            save_path,

            bbox_inches="tight",

            dpi=300,
        )
    plt.tight_layout()
    plt.close()


# ============================================================
# RMSE plot
# ============================================================

def plot_rmse_curves(
    summary_df,
    parameter,
    save_path=None,
):
    """
    Plot RMSE versus sample size.
    """

    fig, ax = plt.subplots()

    grouped = summary_df[
    (summary_df["parameter"] == parameter)
    &
    np.isfinite(summary_df["rmse"])
]

    for (family, scenario_name), gdf in grouped.groupby(
        ["family", "scenario_name"]
    ):

        ax.plot(

            gdf["n"],

            gdf["rmse"],

            marker="o",

            linewidth=2,

            label=f"{family}: {scenario_name}",
        )

    ax.set_xlabel("Sample size $n$")

    ax.set_ylabel("RMSE")

    param_label = get_plot_label(parameter)
    ax.set_title(
        f"RMSE of {param_label}"
    )

    ax.legend()

    ax.grid(alpha=0.3)

    if save_path is not None:

        plt.savefig(

            save_path,

            bbox_inches="tight",

            dpi=300,
        )
    plt.tight_layout()
    plt.close()


# ============================================================
# Coverage comparison
# ============================================================

def plot_coverage_comparison(
    summary_df,
    parameter,
    save_path=None,
):
    """
    Compare naive vs boundary-corrected coverage.
    """

    fig, ax = plt.subplots()

    df = summary_df[
    (summary_df["parameter"] == parameter)
    &
    np.isfinite(summary_df["naive_coverage"])
    &
    np.isfinite(summary_df["bc_coverage"])
]

    for (family, scenario_name), gdf in df.groupby(
        ["family", "scenario_name"]
    ):

        ax.plot(

            gdf["n"],

            gdf["naive_coverage"],

            marker="o",

            linestyle="--",

            linewidth=2,

            label=f"{family} {scenario_name} (naive)",
        )
        ax.plot(

            gdf["n"],

            gdf["bc_coverage"],

            marker="s",

            linewidth=2,

            label=f"{family} {scenario_name} (boundary)",
        )

    ax.axhline(

        0.95,

        linestyle=":",

        color="black",

        label="Nominal 95%",
    )

    ax.set_xlabel("Sample size $n$")

    ax.set_ylabel("Coverage probability")

    ax.set_ylim(0, 1.00)

    param_label = get_plot_label(parameter)
    ax.set_title(
        f"Coverage of {param_label}"
    )

    ax.legend()

    ax.grid(alpha=0.3)

    if save_path is not None:

        plt.savefig(

            save_path,

            bbox_inches="tight",

            dpi=300,
        )
    plt.tight_layout()
    plt.close()


# ============================================================
# SD vs SE convergence plot
# ============================================================

def plot_sd_vs_se(
    summary_df,
    parameter,
    save_path=None,
):
    """
    Plot empirical SD approaching estimated SE asymptotically.
    """

    fig, ax = plt.subplots()

    df = summary_df[
        (summary_df["parameter"] == parameter)
        &
        np.isfinite(summary_df["empirical_sd"])
        &
        np.isfinite(summary_df["mean_bc_se"])
    ]

    for (family, scenario_name), gdf in df.groupby(
        ["family", "scenario_name"]
    ):
        
        # Plot empirical SD
        ax.plot(
            gdf["n"],
            gdf["empirical_sd"],
            marker="o",
            linestyle="-",
            linewidth=2,
            label=f"{family} {scenario_name} (Empirical SD)",
        )
        
        # Plot mean BC SE
        ax.plot(
            gdf["n"],
            gdf["mean_bc_se"],
            marker="s",
            linestyle="--",
            linewidth=2,
            label=f"{family} {scenario_name} (Mean BC SE)",
        )

    ax.set_xlabel("Sample size $n$")
    ax.set_ylabel("Standard Deviation / Error")

    param_label = get_plot_label(parameter)
    ax.set_title(
        f"SD vs SE for {param_label}"
    )

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(alpha=0.3)

    if save_path is not None:
        plt.savefig(
            save_path,
            bbox_inches="tight",
            dpi=300,
        )
    plt.tight_layout()
    plt.close()



# ============================================================
# Boundary geometry visualization
# ============================================================

def plot_boundary_heatmap(
    grid_object,
    contour,
    save_path=None,
    title=None,
):
    """
    Plot decision function heatmap and boundary.
    """

    GX = grid_object["GX"]

    GY = grid_object["GY"]

    G = grid_object["G"]

    fig, ax = plt.subplots()

    heat = ax.contourf(

        GX,

        GY,

        G,

        levels=40,

        cmap="coolwarm",

        alpha=0.85,
    )

    plt.colorbar(
        heat,
        ax=ax,
    )

    if contour is not None:

        ax.plot(

            contour[:, 0],

            contour[:, 1],

            color="black",

            linewidth=3,
        )

    ax.set_xlabel(r"$x_1$")

    ax.set_ylabel(r"$x_2$")

    if title is not None:

        ax.set_title(title)

    ax.grid(alpha=0.2)

    if save_path is not None:

        plt.savefig(

            save_path,

            bbox_inches="tight",

            dpi=300,
        )

    plt.close()


# ============================================================
# Batch plotting
# ============================================================

def generate_all_plots(
    raw_df,
    summary_df,
    output_dir,
):
    """
    Generate all simulation plots.
    """

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Parameter plots
    # --------------------------------------------------------

    parameters = [

    p for p in PLOT_PARAMETERS

    if p in summary_df["parameter"].unique()
]

    for param in parameters:

        # ----------------------------------------------------
        # Bias
        # ----------------------------------------------------

        plot_bias_curves(

            summary_df,

            parameter=param,

            save_path=(
                output_dir
                / f"bias_{param}.png"
            ),
        )

        # ----------------------------------------------------
        # RMSE
        # ----------------------------------------------------

        plot_rmse_curves(

            summary_df,

            parameter=param,

            save_path=(
                output_dir
                / f"rmse_{param}.png"
            ),
        )

        # ----------------------------------------------------
        # Coverage
        # ----------------------------------------------------

        plot_coverage_comparison(

            summary_df,

            parameter=param,

            save_path=(
                output_dir
                / f"coverage_{param}.png"
            ),
        )

        # ----------------------------------------------------
        # SD vs SE
        # ----------------------------------------------------

        plot_sd_vs_se(

            summary_df,

            parameter=param,

            save_path=(
                output_dir
                / f"sd_vs_se_{param}.png"
            ),
        )
