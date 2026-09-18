#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PATH="/home/anonymous/page_gauge_env_protocol_v2/bin:/home/anonymous/cuda13_pcaf_fused/bin:/usr/lib/wsl/lib:/usr/bin:/bin"
export CUDA_HOME=/home/anonymous/cuda13_pcaf_fused
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST=12.0 FLASHINFER_CUDA_ARCH_LIST=12.0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=2
exec /home/anonymous/page_gauge_env_protocol_v2/bin/python -u "$HERE/$1.py" "${@:2}"
