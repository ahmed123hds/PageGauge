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

OUTPUT_DIR=results/a100_cross_gpu_s4_a128_t768/fresh_process_williams_candidate_split200
mkdir -p "$OUTPUT_DIR/logs"

python a100_colab/preflight_a100.py \
  --output results/a100_cross_gpu_s4_a128_t768/a100_preflight.json

if [[ ! -s results/a100_cross_gpu_s4_a128_t768/kernel_factorization_gate.json ]]; then
  bash a100_colab/run_kernel_smoke.sh
fi

python - <<'PY'
import json
from pathlib import Path

gate = json.loads(
    Path("results/a100_cross_gpu_s4_a128_t768/kernel_factorization_gate.json")
    .read_text(encoding="utf-8")
)
if gate.get("passed") is not True:
    raise SystemExit("A100 factorization gate did not pass")
PY

backends=(flashinfer_fp16 page_gauge page_gauge flashinfer_fp16 page_gauge flashinfer_fp16 flashinfer_fp16 page_gauge)
seeds=(20260861 20260861 20260861 20260861 20260862 20260862 20260862 20260862)
offsets=(0 0 0 0 94400 94400 94400 94400)

for index in 0 1 2 3 4 5 6 7; do
  backend=${backends[$index]}
  seed=${seeds[$index]}
  offset=${offsets[$index]}
  if [[ "$backend" == page_gauge ]]; then
    prefix=4
    suffix=128
  else
    prefix=0
    suffix=0
  fi
  result="$OUTPUT_DIR/block_${index}_${backend}.json"
  log="$OUTPUT_DIR/logs/block_${index}_${backend}.log"
  if [[ -e "$result" ]]; then
    echo "Refusing to mix or overwrite an existing block: $result" >&2
    echo "Use a clean Colab runtime/unpack directory for the cross-GPU run." >&2
    exit 2
  fi
  python a100_colab/wait_for_idle.py
  echo "=== A100 BLOCK $((index + 1))/8 backend=$backend seed=$seed offset=$offset ==="
  set +e
  python diagnostics/benchmark_sustained_dynamic_graphs.py \
    --backend "$backend" \
    --model mistralai/Mistral-7B-v0.3 \
    --batch-size 4 \
    --context 20480 \
    --decode-steps 1536 \
    --exact-tail 768 \
    --exact-sink-pages "$prefix" \
    --exact-static-suffix-pages "$suffix" \
    --prefill-chunk-tokens 1024 \
    --baseline-split-pages 256 \
    --candidate-split-pages 200 \
    --tail-attention flashinfer_merge \
    --old-value-scale-placement probability \
    --trajectory-mode frozen_hf_teacher_forced \
    --seed "$seed" \
    --token-source wikitext2 \
    --wikitext-zip data/wikitext-2-raw-v1.zip \
    --wikitext-member wikitext-2-raw/wiki.test.raw \
    --token-offset "$offset" \
    --token-stride 23600 \
    --capture-warmups 1 \
    --maximum-graph-banks 16 \
    --warmups 1 \
    --repeats 3 \
    --cache-scrub-mib 256 \
    --min-logits-cosine 0.995 \
    --min-top1-agreement 0.99 \
    --quality-diagnostics-top-k 0 \
    --output "$result" 2>&1 | tee "$log"
  worker_status=${PIPESTATUS[0]}
  set -e
  if [[ ! -s "$result" ]]; then
    echo "worker exited $worker_status without a result artifact" >&2
    exit "$worker_status"
  fi
  echo "worker exit=$worker_status; preserving complete result for reduction"
done

python a100_colab/analyze_a100_cross_gpu.py \
  --run-dir "$OUTPUT_DIR" \
  --preflight results/a100_cross_gpu_s4_a128_t768/a100_preflight.json \
  --kernel-gate results/a100_cross_gpu_s4_a128_t768/kernel_factorization_gate.json \
  --output results/a100_cross_gpu_s4_a128_t768/final_a100_analysis.json
