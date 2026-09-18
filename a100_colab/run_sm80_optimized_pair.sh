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
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0

SELECTION=results/a100_sm80_kernel_probe/analysis.json
if [[ ! -s "$SELECTION" ]]; then
  echo "Run bash a100_colab/run_sm80_kernel_probe.sh first." >&2
  exit 2
fi
read -r cap exact_split recommended < <(python - "$SELECTION" <<'PY'
import json
import sys

p = json.load(open(sys.argv[1], encoding="utf-8"))
print(p.get("selected_cap") or 0, p.get("selected_exact_split_pages") or 0,
      int(p.get("full_model_run_recommended") is True))
PY
)
if [[ "$recommended" != 1 ]]; then
  echo "The microprobe found no safe improvement; refusing an expensive model run." >&2
  exit 2
fi

OUTPUT_DIR=results/a100_sm80_optimized_pair
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite $OUTPUT_DIR; use a clean directory." >&2
  exit 2
fi
mkdir -p "$OUTPUT_DIR"

common=(
  --model mistralai/Mistral-7B-v0.3
  --batch-size 4 --context 20480 --decode-steps 1536
  --exact-tail 768 --prefill-chunk-tokens 1024
  --baseline-split-pages 256 --candidate-split-pages 200
  --tail-attention flashinfer_merge
  --old-value-scale-placement probability
  --trajectory-mode frozen_hf_teacher_forced
  --seed 20260861 --token-source wikitext2
  --wikitext-zip data/wikitext-2-raw-v1.zip
  --wikitext-member wikitext-2-raw/wiki.test.raw
  --token-offset 0 --token-stride 23600
  --capture-warmups 1 --maximum-graph-banks 16
  --warmups 1 --repeats 3 --cache-scrub-mib 256
  --min-logits-cosine 0.995 --min-top1-agreement 0.99
  --quality-diagnostics-top-k 0
)

python a100_colab/wait_for_idle.py
set +e
python diagnostics/benchmark_sustained_dynamic_graphs.py \
  --backend flashinfer_fp16 --exact-sink-pages 0 \
  --exact-static-suffix-pages 0 "${common[@]}" \
  --output "$OUTPUT_DIR/flashinfer.json"
fi_status=$?
set -e
[[ -s "$OUTPUT_DIR/flashinfer.json" ]] || exit "$fi_status"

python a100_colab/wait_for_idle.py
set +e
python a100_colab/run_with_sm80_kernel.py --target sustained \
  --cap "$cap" --exact-split-pages "$exact_split" -- \
  --backend page_gauge --exact-sink-pages 4 \
  --exact-static-suffix-pages 128 "${common[@]}" \
  --output "$OUTPUT_DIR/page_gauge.json"
pg_status=$?
set -e
[[ -s "$OUTPUT_DIR/page_gauge.json" ]] || exit "$pg_status"

python a100_colab/analyze_sm80_optimized_pair.py \
  --flashinfer "$OUTPUT_DIR/flashinfer.json" \
  --page-gauge "$OUTPUT_DIR/page_gauge.json" \
  --selection "$SELECTION" \
  --output "$OUTPUT_DIR/analysis.json"
