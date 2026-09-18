#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="8.0"
export FLASHINFER_CUDA_ARCH_LIST="8.0"
export TORCH_EXTENSIONS_DIR="$ROOT/build/torch_extensions_a100_sm80_cap_probe"
export MAX_JOBS=${MAX_JOBS:-2}
export PYTHONHASHSEED=0

OUTPUT_DIR=results/a100_sm80_kernel_probe
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite $OUTPUT_DIR; remove it or use a clean unpack directory." >&2
  exit 2
fi
mkdir -p "$OUTPUT_DIR"

common=(
  --lengths 22016,22016,22016,22016
  --representation page_gauge
  --exact-tail 768
  --exact-sink 2112
  --baseline-fixed-split-pages 256
  --candidate-fixed-split-pages 200
  --warmup 20
  --repeats 80
  --cache-scrub-mib 256
  --seed 20260861
)

python scripts/benchmark_flashinfer_page_affine_int8.py \
  "${common[@]}" --output "$OUTPUT_DIR/legacy.json"
python a100_colab/run_with_sm80_kernel.py --target micro --cap 2 -- \
  "${common[@]}" --output "$OUTPUT_DIR/cap2.json"
python a100_colab/run_with_sm80_kernel.py --target micro \
  --exact-split-pages 30 -- \
  "${common[@]}" --output "$OUTPUT_DIR/split30.json"
python a100_colab/run_with_sm80_kernel.py --target micro --cap 2 \
  --exact-split-pages 30 -- \
  "${common[@]}" --output "$OUTPUT_DIR/split30_cap2.json"

python a100_colab/analyze_sm80_kernel_probe.py \
  --legacy "$OUTPUT_DIR/legacy.json" \
  --cap2 "$OUTPUT_DIR/cap2.json" \
  --split30 "$OUTPUT_DIR/split30.json" \
  --split30-cap2 "$OUTPUT_DIR/split30_cap2.json" \
  --output "$OUTPUT_DIR/analysis.json"

echo "A100 SM80 launch-cap probe complete: $OUTPUT_DIR/analysis.json"
