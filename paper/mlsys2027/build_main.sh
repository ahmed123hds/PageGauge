#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
/home/anonymous/page_gauge_env_protocol_v2/bin/python build_evidence_tables.py
/home/anonymous/page_gauge_env_protocol_v2/bin/python build_main_metrics.py
/home/anonymous/page_gauge_env_protocol_v2/bin/python build_regional_cost_table.py
/home/anonymous/page_gauge_env_protocol_v2/bin/python build_synthetic_summary.py
/home/anonymous/page_gauge_env_protocol_v2/bin/python build_replication_summary.py
export TEXINPUTS="$PWD/vendor/style/mlsys2025style:${TEXINPUTS:-}:"
export BSTINPUTS="$PWD/vendor/style/mlsys2025style:${BSTINPUTS:-}:"
export BIBINPUTS="$PWD:${BIBINPUTS:-}:"
mkdir -p build
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
bibtex build/main
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
