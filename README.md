# Classification EM Boundary Correction Reproducibility

This repository contains the full source code required to reproduce the Monte Carlo simulations, tables, plots, and real-data applications presented in our manuscript on the boundary-corrected Classification Expectation-Maximization (CEM) algorithm.

## Prerequisites

This codebase requires Python 3.10+ and the `fmvmm` package. 

To install the necessary dependencies, you can create a virtual environment and run:

```bash
pip install -r requirements.txt
```

## Repository Structure

- `main.py`: The primary entry point for running the large-scale Monte Carlo simulation study. It systematically evaluates different mixture families (`normal_t` and `normal_skewnormal`) across varying overlap scenarios.
- `regenerate_plots.py`: A utility script that consumes the raw outputs from `results/raw` and regenerates the high-quality PDF plots and LaTeX tables presented in the manuscript.
- `run_real_data.py`: The script for running the real-world application on the Breast Cancer Wisconsin (Diagnostic) dataset, which validates the boundary correction using a non-parametric bootstrap strategy.
- `inference/`: Contains the core statistical methodologies, including the novel sandwich covariance estimator (`sandwich.py`), boundary decomposition logic (`boundary.py`), and analytical score functions (`scores.py`).
- `simulation/`: Contains utilities defining the data-generating scenarios (`scenarios.py`), numerical CML targets (`compute_cml_targets.py` and `cml_targets.json`), and metrics for evaluating finite-sample performance (`metrics.py`).
- `results/`: The directory where all generated data, plots, and tables will be saved during execution.

## Reproducing the Results

### 1. Monte Carlo Simulation Study

To execute the large-scale simulations (this validates coverage, asymptotic consistency, and standard errors relative to the CML targets):

```bash
python main.py
```
*Note: The simulations are parallelized via `joblib`. Running the full benchmark (N=10,000 to N=20,000 across multiple families and overlap regimes) may take significant computational time.*

Once `main.py` is completed, you can format the output into LaTeX tables and publication-ready plots by running:

```bash
python regenerate_plots.py
```

All plots will be output to `results/plots/`, and all tables will be output to `results/tables/`.

### 2. Real Data Application (Breast Cancer Wisconsin)

To run the real-world validation comparing the analytical boundary-corrected standard errors to the non-parametric bootstrap standard deviations:

```bash
python run_real_data.py
```
This script will output the comparison to the terminal and validate that the boundary-corrected estimator flawlessly tracks the empirical bootstrap variability without requiring computationally expensive resampling procedures.
