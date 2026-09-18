#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

export TORCH_CUDA_ARCH_LIST="8.0+PTX"
export FLASHINFER_CUDA_ARCH_LIST="8.0"
export TORCH_EXTENSIONS_DIR="$ROOT/build/torch_extensions_a100_sm80"
export MAX_JOBS=${MAX_JOBS:-2}
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0

OUTPUT=results/a100_cross_gpu_s4_a128_t768/kernel_factorization_smoke.json
python a100_colab/preflight_a100.py \
  --output results/a100_cross_gpu_s4_a128_t768/a100_preflight.json
python scripts/benchmark_flashinfer_page_affine_int8.py \
  --lengths 20480 \
  --representation page_gauge \
  --exact-tail 0 \
  --exact-sink 0 \
  --baseline-fixed-split-pages 256 \
  --candidate-fixed-split-pages 256 \
  --warmup 5 \
  --repeats 20 \
  --cache-scrub-mib 256 \
  --seed 20260861 \
  --output "$OUTPUT"
python a100_colab/validate_kernel_smoke.py \
  --input "$OUTPUT" \
  --output results/a100_cross_gpu_s4_a128_t768/kernel_factorization_gate.json
