#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-$project_root/../.venv/bin/python}"

cd "$project_root"
env PYTHONPATH=src:scripts MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/cem-mpl}" \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$python_bin" scripts/make_paper_figures.py
"$python_bin" scripts/build_supplement_tables.py
"$python_bin" scripts/build_jrss_submission.py

cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error jrssb_main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error jrssb_supplement.tex

echo "Built paper/jrssb_main.pdf and paper/jrssb_supplement.pdf"
