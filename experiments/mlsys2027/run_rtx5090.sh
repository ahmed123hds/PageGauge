#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
PG_PYTHON=${PG_PYTHON:-/home/anonymous/page_gauge_env_protocol_v2/bin/python}
export CUDA_HOME=${PG_CUDA_HOME:-/home/anonymous/cuda13_pcaf_fused}
if [[ ! -x "$PG_PYTHON" ]]; then
  printf 'PageGauge Python not found: %s\n' "$PG_PYTHON" >&2
  exit 2
fi
# Calling a virtualenv's Python directly does not activate its bin directory.
# FlashInfer launches ninja by name, so expose the matching build tools too.
export PATH="$(dirname "$PG_PYTHON"):$CUDA_HOME/bin:/usr/lib/wsl/lib:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST=12.0
export FLASHINFER_CUDA_ARCH_LIST=12.0
export TORCH_EXTENSIONS_DIR="$ROOT/build/mlsys2027_step01_extensions"
export MAX_JOBS=2
export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# "check"/"check02" only read files/package metadata. GPU steps check idle
# state before separate workers start. No package installations occur.
if [[ $# -eq 0 ]]; then
  set -- check
fi
exec "$PG_PYTHON" diagnostics/mlsys_rtx5090_entry.py "$@"
