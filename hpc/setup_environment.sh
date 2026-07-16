#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
module load python/3.11
python3.11 -m venv .venv311
.venv311/bin/python -m pip install --upgrade pip setuptools wheel
.venv311/bin/python -m pip install -e .
.venv311/bin/python -c 'import cem_inference, fmvmm; print(cem_inference.__version__, fmvmm.__version__)'
