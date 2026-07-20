"""Generate publication figures from the locked JRSS B result files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from cem_inference import NormalAdapter, StudentTAdapter, fit_fmvmm_hard
from run_real_data_screen import load_dataset


ROOT = Path(__file__).resolve().parents[1]
BLUE, ORANGE, GREEN, RED = "#276FBF", "#D95F02", "#2A9D8F", "#B23A48"


def style() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.labelsize": 9, "axes.titlesize": 10,
        "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 160, "savefig.dpi": 300,
    })


def save(fig: plt.Figure, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def scenario_label(value: str) -> str:
    parts = value.split("_")
    family = {"nt": "N--t", "nsn": "N--SN", "ntsn": "N--t--SN"}[parts[0]]
    dimension = parts[1].replace("p", "$p=$")
    regime = parts[2].replace("moderate", "moderate").replace("separated", "separated")
    balance = "imbal." if "imbalanced" in parts else "bal."
    return f"{family}, {dimension}, {regime}, {balance}"


def simulation_calibration(out: Path) -> None:
    perf = pd.read_csv(ROOT / "results/processed/confirmatory_v2/tables/scenario_performance.csv")
    wide = perf.pivot_table(index=["scenario", "n"], columns="method",
                            values=["median_se_to_sd", "median_coverage"]).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.15))
    family = wide[("scenario", "")].str.split("_").str[0]
    colors = family.map({"nt": BLUE, "nsn": ORANGE, "ntsn": GREEN})
    sizes = np.where(wide[("n", "")].to_numpy() >= 2000, 35, 20)
    panels = [
        ("median_se_to_sd", "Naive SE / empirical SD", "Corrected SE / empirical SD", 1.0),
        ("median_coverage", "Naive 95% coverage", "Corrected 95% coverage", 0.95),
    ]
    for ax, (metric, xlabel, ylabel, target) in zip(axes, panels):
        x = wide[(metric, "naive")].to_numpy()
        y = wide[(metric, "corrected_stabilized")].to_numpy()
        lo = min(x.min(), y.min(), target) - .04
        hi = max(x.max(), y.max(), target) + .04
        ax.plot([lo, hi], [lo, hi], color="0.65", lw=1, ls="--")
        ax.axhline(target, color="0.15", lw=.8, ls=":")
        ax.axvline(target, color="0.15", lw=.8, ls=":")
        ax.scatter(x, y, c=colors, s=sizes, alpha=.82, edgecolor="white", linewidth=.35)
        ax.set(xlabel=xlabel, ylabel=ylabel, xlim=(lo, hi), ylim=(lo, hi))
        ax.grid(alpha=.18)
    handles = [mpl.lines.Line2D([], [], marker="o", linestyle="", color=c, label=l)
               for c, l in [(BLUE, "Normal--Student-$t$"), (ORANGE, "Normal--skew-normal"),
                            (GREEN, "three components")]]
    axes[1].legend(handles=handles, loc="lower right", frameon=False)
    fig.tight_layout()
    save(fig, out, "simulation_calibration")


def simulation_profiles(out: Path) -> None:
    perf = pd.read_csv(ROOT / "results/processed/confirmatory_v2/tables/scenario_performance.csv")
    keep = perf[perf.scenario.str.match(r"^(nt|nsn)_p2_(separated|moderate|strong)_balanced$")]
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.3), sharey="row")
    for col, prefix in enumerate(("nt", "nsn")):
        sub = keep[keep.scenario.str.startswith(prefix + "_")]
        for regime, marker in [("separated", "o"), ("moderate", "s"), ("strong", "^")]:
            cell = sub[sub.scenario.str.contains(f"_{regime}_")]
            for method, color, ls in [("naive", "0.35", "--"),
                                      ("corrected_stabilized", BLUE, "-")]:
                d = cell[cell.method.eq(method)].sort_values("n")
                axes[0, col].plot(d.n, d.median_se_to_sd, marker=marker, color=color,
                                  ls=ls, label=f"{regime}, {'corrected' if method != 'naive' else 'naive'}")
                axes[1, col].plot(d.n, d.median_coverage, marker=marker, color=color, ls=ls)
        axes[0, col].axhline(1, color="0.15", lw=.8, ls=":")
        axes[1, col].axhline(.95, color="0.15", lw=.8, ls=":")
        axes[0, col].set_title("Normal--Student-$t$" if prefix == "nt" else "Normal--skew-normal")
        axes[1, col].set_xlabel("Sample size")
        axes[0, col].grid(alpha=.18); axes[1, col].grid(alpha=.18)
    axes[0, 0].set_ylabel("Median SE / empirical SD")
    axes[1, 0].set_ylabel("Median 95% coverage")
    method_handles = [mpl.lines.Line2D([], [], color="0.35", ls="--", label="Naive"),
                      mpl.lines.Line2D([], [], color=BLUE, label="Corrected")]
    regime_handles = [mpl.lines.Line2D([], [], marker=m, color="0.25", linestyle="", label=r.title())
                      for r, m in [("separated", "o"), ("moderate", "s"), ("strong", "^")]]
    axes[0, 1].legend(handles=method_handles + regime_handles, frameon=False, ncol=2, loc="lower right")
    fig.tight_layout()
    save(fig, out, "simulation_profiles")


def extension_and_boundary(out: Path) -> None:
    ext = pd.read_csv(ROOT / "results/processed/asymptotic_extension/tables/scenario_performance.csv")
    wide = ext.pivot(index=["scenario", "n"], columns="method", values="median_coverage").reset_index()
    conv = pd.read_csv(ROOT / "results/processed/boundary_convergence_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.15))
    y = np.arange(len(wide))
    axes[0].plot(wide.naive, y, "o", color="0.4", label="Naive")
    axes[0].plot(wide.corrected_stabilized, y, "o", color=BLUE, label="Corrected")
    for i in y:
        axes[0].plot([wide.naive.iloc[i], wide.corrected_stabilized.iloc[i]], [i, i], color="0.75")
    axes[0].axvline(.95, color="0.15", ls=":", lw=.8)
    axes[0].set_yticks(y, [scenario_label(s) for s in wide.scenario])
    axes[0].set_xlabel("Median 95% coverage")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="x", alpha=.18)
    axes[1].loglog(conv.n, conv.rmse, "o-", color=GREEN, label="RMSE")
    scale = conv.rmse.iloc[0] * np.sqrt(conv.n.iloc[0] / conv.n)
    axes[1].loglog(conv.n, scale, ls="--", color="0.45", label=r"$n^{-1/2}$ reference")
    axes[1].set(xlabel="Sample size", ylabel="Boundary-functional RMSE")
    axes[1].grid(alpha=.18, which="both")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    save(fig, out, "extension_boundary_convergence")


def bandwidth_and_failures(out: Path) -> None:
    full = pd.read_csv(ROOT / "results/processed/confirmatory_v2/summary.csv")
    d = full[full.method.eq("corrected_stabilized")].groupby(
        ["scenario", "n", "bandwidth_multiplier"], as_index=False
    ).agg(coverage=("coverage_95", "median"), se_ratio=("se_to_empirical_sd", "median"))
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.15))
    for metric, ax, target, ylabel in [("coverage", axes[0], .95, "Median 95% coverage"),
                                       ("se_ratio", axes[1], 1., "Median SE / empirical SD")]:
        for _, cell in d.groupby(["scenario", "n"]):
            prefix = cell.scenario.iloc[0].split("_")[0]
            ax.plot(cell.bandwidth_multiplier, cell[metric], color={"nt": BLUE, "nsn": ORANGE,
                    "ntsn": GREEN}[prefix], alpha=.25, lw=.8)
        med = d.groupby("bandwidth_multiplier")[metric].median()
        ax.plot(med.index, med.values, "o-", color="black", lw=1.6, label="Across-cell median")
        ax.axhline(target, color="0.2", ls=":", lw=.8)
        ax.set(xlabel="Bandwidth multiplier", ylabel=ylabel, xticks=[.75, 1, 1.5])
        ax.grid(alpha=.18); ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, out, "bandwidth_sensitivity")


def application_fit(name: str):
    raw, labels = load_dataset(name)
    obs = PCA(n_components=2, svd_solver="full").fit_transform(StandardScaler().fit_transform(raw))
    fit = fit_fmvmm_hard(obs, [NormalAdapter(2), StudentTAdapter(2)], max_iter=160, tol=1e-6)
    return obs, labels, fit


def application_geometry(out: Path) -> None:
    datasets = [("dry_bean_dermason_seker", "Dermason--Seker"),
                ("dry_bean_cali_barbunya", "Cali--Barbunya")]
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.2))
    for ax, (name, title) in zip(axes, datasets):
        obs, labels, fit = application_fit(name)
        unique = np.unique(labels)
        for lab, color in zip(unique, (BLUE, ORANGE)):
            mask = labels == lab
            ax.scatter(obs[mask, 0], obs[mask, 1], s=5, alpha=.28, color=color, label=lab.title(), rasterized=True)
        xlim, ylim = np.quantile(obs[:, 0], [.005, .995]), np.quantile(obs[:, 1], [.005, .995])
        gx, gy = np.meshgrid(np.linspace(*xlim, 240), np.linspace(*ylim, 240))
        grid = np.c_[gx.ravel(), gy.ravel()]
        scores = fit.model.component_scores(grid, fit.coordinates)
        contrast = (scores[:, 0] - scores[:, 1]).reshape(gx.shape)
        ax.contour(gx, gy, contrast, levels=[0], colors="black", linewidths=1.5)
        ax.set(title=title, xlabel="Standardized PC1", ylabel="Standardized PC2")
        ax.legend(frameon=False, markerscale=2)
    fig.tight_layout()
    save(fig, out, "application_boundaries")


def application_diagnostics(out: Path) -> None:
    real = ROOT / "results/processed/real_data/final"
    files = [("dry_bean_dermason_seker_parameters.csv", "Dermason--Seker"),
             ("dry_bean_cali_barbunya_parameters.csv", "Cali--Barbunya")]
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.15), sharey=True)
    for ax, (file, title) in zip(axes, files):
        d = pd.read_csv(real / file)
        x = np.arange(len(d))
        ax.plot(x, d.naive_to_local_bootstrap, "o", ms=4, color="0.45", label="Naive")
        ax.plot(x, d.stabilized_to_local_bootstrap, "o", ms=4, color=BLUE, label="Corrected")
        ax.axhline(1, color="0.15", ls=":", lw=.8)
        ax.set(title=title, xlabel="Unconstrained parameter coordinate", xticks=x)
        ax.grid(axis="y", alpha=.18)
    axes[0].set_ylabel("Analytic SE / local-bootstrap SD")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    save(fig, out, "application_se_calibration")


def basin_diagnostics(out: Path) -> None:
    datasets = [("dry_bean_dermason_seker", "Dermason--Seker"),
                ("dry_bean_cali_barbunya", "Cali--Barbunya")]
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.0), sharey=True)
    for ax, (name, title) in zip(axes, datasets):
        _, _, fit = application_fit(name)
        files = sorted((ROOT / "results/raw/real_data" / name).glob("global_*.csv"))
        frame = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        success = frame[frame.status.eq("success") & frame.coordinate.notna()]
        wide = success.pivot(index="replication", columns="coordinate", values="estimate")
        wide = wide.reindex(columns=range(fit.model.parameter_dimension))
        dist = np.max(np.abs(wide.to_numpy() - fit.coordinates), axis=1)
        ax.hist(dist, bins=np.linspace(0, max(3, np.quantile(dist, .99)), 24), color=GREEN, alpha=.8)
        ax.axvline(.75, color=RED, lw=1.3, ls="--", label="Local-basin threshold")
        ax.set(title=title, xlabel="Maximum coordinate distance")
        ax.legend(frameon=False)
    axes[0].set_ylabel("Unrestricted bootstrap refits")
    fig.tight_layout()
    save(fig, out, "application_basin_diagnostics")


def asymptotic_evidence(out: Path) -> None:
    """Direct empirical checks of rate and the Gaussian approximation."""
    diag = pd.read_csv(ROOT / "results/processed/confirmatory_v2/normality_diagnostics.csv")
    summ = pd.read_csv(ROOT / "results/processed/confirmatory_v2/summary.csv")
    summ = summ[(summ.method == "naive") & summ.n.isin([500, 2000])]
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 6.0))
    probs = np.array([.01,.025,.05,.10,.25,.50,.75,.90,.95,.975,.99])
    theo = stats.norm.ppf(probs)
    qcols = [f"q{int(1000*p):03d}" for p in probs]
    for n, color in [(500, ORANGE), (2000, BLUE)]:
        q = diag.loc[diag.n.eq(n), qcols].median().to_numpy(float)
        axes[0,0].plot(theo, q, "o-", ms=3, color=color, label=f"n={n}")
    axes[0,0].plot(theo, theo, color="0.2", ls=":", lw=1)
    axes[0,0].set(xlabel="Standard-normal quantile", ylabel="Median empirical quantile",
                  title="Standardized root-n errors")
    axes[0,0].legend(frameon=False)
    vals=[diag.loc[diag.n.eq(n),'ks_distance'].dropna() for n in sorted(diag.n.unique())]
    axes[0,1].boxplot(vals, tick_labels=[str(n) for n in sorted(diag.n.unique())], showfliers=False)
    axes[0,1].set(xlabel="Sample size", ylabel="KS distance", title="Coordinate-level Gaussian distance")
    med=summ.groupby(['scenario','n'],as_index=False).agg(rmse=('rmse','median'),bias=('bias',lambda x: np.median(np.abs(x))))
    for scenario,g in med.groupby('scenario'):
        if len(g)>1:
            axes[1,0].plot(g.n, np.sqrt(g.n)*g.rmse, color=BLUE, alpha=.25)
            axes[1,1].plot(g.n, np.sqrt(g.n)*g.bias, color=GREEN, alpha=.25)
    axes[1,0].set(xscale='log', xlabel="Sample size", ylabel=r"Median $\sqrt{n}\,\mathrm{RMSE}$", title="Root-n rate diagnostic")
    axes[1,1].set(xscale='log', xlabel="Sample size", ylabel=r"Median $\sqrt{n}\,|\mathrm{bias}|$", title="Centeredness diagnostic")
    for ax in axes.ravel(): ax.grid(alpha=.15)
    fig.tight_layout()
    save(fig, out, "asymptotic_evidence")


def coordinate_atlas(out: Path) -> None:
    """Plot every reported coordinate, retaining the breadth of the old supplement."""
    d = pd.read_csv(ROOT / "results/processed/confirmatory_v2/summary.csv")
    central = d[(d.method.eq("naive")) | ((d.method.eq("corrected_stabilized")) & d.bandwidth_multiplier.eq(1.0))].copy()
    scenarios = sorted(central.scenario.unique())
    labels = {s: s.replace("moderate_balanced", "mod").replace("moderate_imbalanced", "imb")
              .replace("separated_balanced", "sep").replace("strong_balanced", "strong") for s in scenarios}
    specs = [
        ("bias", "Bias relative to the CML target", "coordinate_bias_atlas", False),
        ("rmse", "RMSE relative to the CML target", "coordinate_rmse_atlas", False),
        ("coverage_95", "Empirical 95% coverage", "coordinate_coverage_atlas", True),
        ("se_to_empirical_sd", "Estimated SE / empirical SD", "coordinate_se_atlas", True),
    ]
    colors = {500: ORANGE, 1000: GREEN, 2000: BLUE}
    for metric, ylabel, name, compare_methods in specs:
        fig, axes = plt.subplots(6, 3, figsize=(8.0, 11.2), squeeze=False)
        for ax, scenario in zip(axes.ravel(), scenarios):
            g = central[central.scenario.eq(scenario)]
            methods = ["naive", "corrected_stabilized"] if compare_methods else ["naive"]
            for method in methods:
                for n, z in g[g.method.eq(method)].groupby("n"):
                    z = z.sort_values("coordinate")
                    ax.plot(np.arange(len(z)), z[metric], color=colors.get(int(n), "0.4"),
                            lw=.65, alpha=.85, ls="-" if method == "corrected_stabilized" else ":")
            ref = 0 if metric == "bias" else (0.95 if metric == "coverage_95" else (1 if metric == "se_to_empirical_sd" else None))
            if ref is not None: ax.axhline(ref, color="0.15", lw=.55, ls="--")
            ax.set_title(labels[scenario], fontsize=7)
            ax.tick_params(labelsize=6); ax.grid(alpha=.10)
        for ax in axes[-1, :]: ax.set_xlabel("Unconstrained coordinate", fontsize=7)
        for ax in axes[:, 0]: ax.set_ylabel(ylabel, fontsize=7)
        handles=[plt.Line2D([0],[0],color=c,lw=1,label=f"n={n}") for n,c in colors.items()]
        if compare_methods:
            handles += [plt.Line2D([0],[0],color="0.2",ls=":",label="Naive"),
                        plt.Line2D([0],[0],color="0.2",ls="-",label="Corrected")]
        fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, fontsize=7)
        fig.tight_layout(rect=(0,0,1,.975))
        save(fig, out, name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "paper/figures")
    args = parser.parse_args()
    style()
    simulation_calibration(args.output)
    simulation_profiles(args.output)
    extension_and_boundary(args.output)
    bandwidth_and_failures(args.output)
    application_geometry(args.output)
    application_diagnostics(args.output)
    basin_diagnostics(args.output)
    asymptotic_evidence(args.output)
    coordinate_atlas(args.output)


if __name__ == "__main__":
    main()
