# main.py

from pathlib import Path

import pandas as pd

from simulation.scenarios import (
    get_scenario_grid,
    print_scenario_summary,
)

from simulation.monte_carlo import (
    run_monte_carlo,
)

from simulation.metrics import (
    summarize_results,
    simulation_diagnostics,
    print_simulation_diagnostics,
)

from simulation.tables import (
    export_full_summary_table,
    export_family_tables,
)

from simulation.plots import (
    generate_all_plots,
)


# ============================================================
# Main simulation pipeline
# ============================================================

def main():
    """
    Main simulation driver for:

    Boundary-corrected inference for
    heterogeneous CEM estimators.
    """

    # ========================================================
    # Project directories
    # ========================================================

    project_dir = Path(__file__).resolve().parent

    results_dir = project_dir / "results"

    raw_dir = results_dir / "raw"

    summary_dir = results_dir / "summary"

    tables_dir = results_dir / "tables"

    plots_dir = results_dir / "plots"

    diagnostics_dir = results_dir / "diagnostics"

    # --------------------------------------------------------
    # Create folders
    # --------------------------------------------------------

    for d in [

        results_dir,

        raw_dir,

        summary_dir,

        tables_dir,

        plots_dir,

        diagnostics_dir,
    ]:

        d.mkdir(

            parents=True,

            exist_ok=True,
        )

    # ========================================================
    # Simulation settings
    # ========================================================

    n_replications = 100

    sample_sizes = [

        5000,
        10000,
        15000,
        20000,
    ]

    random_seed = 12345

    # --------------------------------------------------------
    # Family options
    # --------------------------------------------------------

    family_pair = "all"

    # ========================================================
    # Load simulation grid
    # ========================================================

    scenario_grid = get_scenario_grid(

        family_pair=family_pair,

        sample_sizes=sample_sizes,
    )

    print("\n")
    print("=" * 70)
    print("Boundary-Corrected Heterogeneous CEM Simulation")
    print("=" * 70)

    print_scenario_summary(
        scenario_grid
    )

    print(f"\nReplications: {n_replications}")

    # ========================================================
    # Run Monte Carlo simulations
    # ========================================================

    print("\nRunning Monte Carlo simulations...\n")

    raw_results = run_monte_carlo(

        scenario_grid=scenario_grid,

        n_replications=n_replications,

        random_seed=random_seed,

        n_jobs=-1,
    )

    # ========================================================
    # Save raw results
    # ========================================================

    raw_path = (

        raw_dir
        / "raw_simulation_results.csv"
    )

    raw_results.to_csv(

        raw_path,

        index=False,
    )

    print(f"\nRaw results saved to:\n{raw_path}")

    # ========================================================
    # Diagnostics
    # ========================================================

    diagnostics = simulation_diagnostics(
        raw_results
    )

    print_simulation_diagnostics(
        diagnostics
    )

    # --------------------------------------------------------
    # Save diagnostics
    # --------------------------------------------------------

    diagnostics_df = pd.DataFrame([

        {
            "metric": k,
            "value": str(v),
        }

        for k, v in diagnostics.items()
    ])

    diagnostics_path = (

        diagnostics_dir
        / "simulation_diagnostics.csv"
    )

    diagnostics_df.to_csv(

        diagnostics_path,

        index=False,
    )

    # ========================================================
    # Summarize results
    # ========================================================

    print("\nSummarizing simulation results...\n")

    summary_results = summarize_results(
        raw_results
    )

    # --------------------------------------------------------
    # Save summary CSV
    # --------------------------------------------------------

    summary_path = (

        summary_dir
        / "summary_results.csv"
    )

    summary_results.to_csv(

        summary_path,

        index=False,
    )

    print(f"Summary results saved to:\n{summary_path}")

    # ========================================================
    # Export publication tables
    # ========================================================

    print("\nExporting publication tables...\n")

    export_full_summary_table(

        summary_results,

        output_dir=tables_dir,
    )

    export_family_tables(

        summary_results,

        output_dir=tables_dir,
    )

    print(f"Tables exported to:\n{tables_dir}")

    # ========================================================
    # Generate plots
    # ========================================================

    print("\nGenerating plots...\n")

    generate_all_plots(

        raw_df=raw_results,

        summary_df=summary_results,

        output_dir=plots_dir,
    )

    print(f"Plots saved to:\n{plots_dir}")

    # ========================================================
    # Compact preview
    # ========================================================

    print("\n")
    print("=" * 70)
    print("Simulation Summary Preview")
    print("=" * 70)

    with pd.option_context(

        "display.max_columns", None,

        "display.width", 160,
    ):

        print(
            summary_results.head(20)
        )

    print("\n")
    print("=" * 70)
    print("Simulation pipeline completed successfully.")
    print("=" * 70)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()
