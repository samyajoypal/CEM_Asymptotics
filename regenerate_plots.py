import pandas as pd
from pathlib import Path
from simulation.plots import generate_all_plots

project_dir = Path(__file__).resolve().parent
results_dir = project_dir / "results"
raw_path = results_dir / "raw" / "raw_simulation_results.csv"
summary_path = results_dir / "summary" / "summary_results.csv"
plots_dir = results_dir / "plots"

raw_results = pd.read_csv(raw_path)
summary_results = pd.read_csv(summary_path)

print("Regenerating plots...")
generate_all_plots(raw_df=raw_results, summary_df=summary_results, output_dir=plots_dir)
print("Done!")
