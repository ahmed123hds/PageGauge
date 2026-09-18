#!/usr/bin/env bash
# Build only the native kernel in an isolated venv. No model/dependency upgrades.
set -euo pipefail
SOURCE=/home/anonymous/pagegauge_baselines/KIVI
TARGET=/home/anonymous/pagegauge_baselines/kivi_sm120_env
EXPECTED=876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6
test "$(git -C "$SOURCE" rev-parse HEAD)" = "$EXPECTED"
test -z "$(git -C "$SOURCE" status --porcelain --untracked-files=no)"
if [ ! -f "$TARGET/pyvenv.cfg" ]; then
  /home/anonymous/page_gauge_env_protocol_v2/bin/python -m venv --system-site-packages "$TARGET"
fi
# venv inherits the underlying interpreter, not another venv's site-packages.
# This reviewed .pth adds the existing runtime after this environment's own
# packages. Provision it separately; never install into the shared runtime.
test -f "$TARGET/lib/python3.12/site-packages/pagegauge_shared_runtime.pth"
export PATH="$TARGET/bin:/home/anonymous/page_gauge_env_protocol_v2/bin:/home/anonymous/cuda13_pcaf_fused/bin:/usr/bin:/bin"
export CUDA_HOME=/home/anonymous/cuda13_pcaf_fused
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$TARGET/bin/python" -m pip install --no-deps --no-build-isolation "$SOURCE/quant"
"$TARGET/bin/python" -c 'import torch, kivi_gemv; print("torch", torch.__version__); print("native kernel", kivi_gemv.__file__); print("GPU execution not yet validated")'
