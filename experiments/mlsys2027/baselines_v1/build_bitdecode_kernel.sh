#!/usr/bin/env bash
# Isolated, explicitly labeled SM120 portability build; not native-release performance.
set -euo pipefail
SOURCE=/home/anonymous/pagegauge_baselines/BitDecoding
TARGET=/home/anonymous/pagegauge_baselines/bitdecode_sm120_env
test "$(git -C "$SOURCE" rev-parse HEAD)" = ae0d83630d6292453355ced498db2ac87f56ec62
test "$(git -C "$SOURCE/libs/cutlass" rev-parse HEAD)" = 3fe62887d8dd75700fdaf57f9c181878701b0802
test -f "$TARGET/lib/python3.12/site-packages/pagegauge_shared_runtime.pth"
export PATH="$TARGET/bin:/home/anonymous/page_gauge_env_protocol_v2/bin:/home/anonymous/cuda13_pcaf_fused/bin:/usr/bin:/bin"
export CUDA_HOME=/home/anonymous/cuda13_pcaf_fused
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export BIT_DECODE_CUDA_ARCH=120 TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$TARGET/bin/python" -m pip install --no-deps --no-build-isolation "$SOURCE"
"$TARGET/bin/python" -c 'import torch, bit_decode_cuda; print("native binary", bit_decode_cuda.__file__); print("SM120 GPU execution not yet validated")'
