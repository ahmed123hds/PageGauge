#!/usr/bin/env bash
# Isolated NSNQuant build. Run only after controlled timing jobs have ended.
set -euo pipefail
SOURCE=/home/anonymous/pagegauge_baselines/NSNQuant
TARGET=/home/anonymous/pagegauge_baselines/nsn_sm120_env
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
test "$(git -C "$SOURCE" rev-parse HEAD)" = 604db3ca34e8de7026b404048eca58b894769701
test "$(git -C "$SOURCE/3rdparty/fast-hadamard-transform" rev-parse HEAD)" = d1a56eee9d502e67faacf61b7b947180d66b32a0
# Compilation itself uses no GPU, but CPU contention invalidates model timings.
GPU_PIDS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
if [ -n "$GPU_PIDS" ]; then
  echo "GPU work is active; defer baseline compilation." >&2
  exit 3
fi
if [ ! -f "$TARGET/pyvenv.cfg" ]; then
  /home/anonymous/page_gauge_env_protocol_v2/bin/python -m venv "$TARGET"
fi
cp "$SCRIPT_DIR/nsn_shared_runtime.pth" "$TARGET/lib/python3.12/site-packages/pagegauge_shared_runtime.pth"
export PATH="$TARGET/bin:/home/anonymous/page_gauge_env_protocol_v2/bin:/home/anonymous/cuda13_pcaf_fused/bin:/usr/bin:/bin"
export CUDA_HOME=/home/anonymous/cuda13_pcaf_fused
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export FAST_HADAMARD_TRANSFORM_FORCE_BUILD=TRUE FAST_HADAMARD_TRANSFORM_CUDA_ARCH=120
# Upstream's old Torch2.4 pin cannot serve SM120. Share the already-installed
# modern Torch read-only, while keeping model API packages private to this venv.
"$TARGET/bin/python" -m pip install --no-deps transformers==4.48.1 tokenizers==0.21.4
"$TARGET/bin/python" -m pip install --no-deps --no-build-isolation "$SOURCE"
"$TARGET/bin/python" -m pip install --no-deps --no-build-isolation "$SOURCE/3rdparty/fast-hadamard-transform"
"$TARGET/bin/python" "$SCRIPT_DIR/check_nsn_import.py"
