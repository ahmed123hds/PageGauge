#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
/home/anonymous/page_gauge_env_protocol_v2/bin/python build_evidence_tables.py
export TEXINPUTS="$PWD/vendor/style/mlsys2025style:${TEXINPUTS:-}:"
mkdir -p build
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build development_results.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build development_results.tex
