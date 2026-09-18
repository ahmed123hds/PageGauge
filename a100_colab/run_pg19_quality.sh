#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="8.0+PTX"
export FLASHINFER_CUDA_ARCH_LIST="8.0"
export TORCH_EXTENSIONS_DIR="$ROOT/build/torch_extensions_a100_sm80"
export MAX_JOBS=${MAX_JOBS:-2}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0

OUTPUT_DIR=results/a100_pg19_s4_a128_t768
BOOK_DIR="$OUTPUT_DIR/books"
mkdir -p "$BOOK_DIR/logs"
python a100_colab/preflight_a100.py --output "$OUTPUT_DIR/a100_preflight.json"

ids=(10146 10321 10356 10762 15562 22424)
inputs=()
for id in "${ids[@]}"; do
  result="$BOOK_DIR/$id.json"
  log="$BOOK_DIR/logs/$id.log"
  if [[ -e "$result" ]]; then
    echo "Refusing to overwrite or mix PG-19 result: $result" >&2
    exit 2
  fi
  python a100_colab/wait_for_idle.py
  set +e
  python diagnostics/benchmark_pg19_external_quality.py \
    --model mistralai/Mistral-7B-v0.3 \
    --batch-size 1 \
    --context 20480 \
    --decode-steps 1536 \
    --exact-tail 768 \
    --exact-prefix-pages 4 \
    --exact-static-suffix-pages 128 \
    --prefill-chunk-tokens 1024 \
    --baseline-split-pages 256 \
    --candidate-split-pages 256 \
    --tail-attention flashinfer_merge \
    --old-value-scale-placement probability \
    --token-source pg19 \
    --pg19-file "data/pg19_external_test_v1/$id.txt" \
    --pg19-object-name "test/$id.txt" \
    --pg19-selection-manifest \
      data/pg19_external_test_v1/PG19_EXTERNAL_CONFIRMATION_MANIFEST.json \
    --disable-attention-cuda-graphs \
    --min-logits-cosine 0.995 \
    --min-top1-agreement 0.99 \
    --min-baseline-hf-cosine 0.999 \
    --output "$result" 2>&1 | tee "$log"
  worker_status=${PIPESTATUS[0]}
  set -e
  if [[ ! -s "$result" ]]; then
    echo "PG-19 worker exited $worker_status without an artifact" >&2
    exit "$worker_status"
  fi
  inputs+=("$result")
done

python diagnostics/aggregate_pg19_external_quality.py \
  --inputs "${inputs[@]}" \
  --selection-manifest \
    data/pg19_external_test_v1/PG19_EXTERNAL_CONFIRMATION_MANIFEST.json \
  --output "$OUTPUT_DIR/aggregate.json"

python a100_colab/validate_a100_pg19.py \
  --inputs "${inputs[@]}" \
  --preflight "$OUTPUT_DIR/a100_preflight.json" \
  --aggregate "$OUTPUT_DIR/aggregate.json" \
  --output "$OUTPUT_DIR/a100_pg19_gate.json"
