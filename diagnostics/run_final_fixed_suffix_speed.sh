#!/usr/bin/env bash
set -euo pipefail

cd /mnt/c/Users/noorg/Ahmed/page_gauge_protocol_v2_rtx5090
export PATH=/home/anonymous/page_gauge_env_protocol_v2/bin:/home/anonymous/cuda13_pcaf_fused/bin:/usr/lib/wsl/lib:/usr/bin:/bin
export CUDA_HOME=/home/anonymous/cuda13_pcaf_fused
export LD_LIBRARY_PATH=/home/anonymous/cuda13_pcaf_fused/lib64
export TORCH_EXTENSIONS_DIR="$PWD/build/torch_extensions_rtx5090"
export TORCH_CUDA_ARCH_LIST=12.0
export MAX_JOBS=4
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0

output_dir=results/final_fixed_suffix_s4_a128_t768/performance/fresh_process_williams_split128_v2
mkdir -p "$output_dir"

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
  result="$output_dir/block_${index}_${backend}.json"
  echo "=== FINAL SPEED BLOCK $((index + 1))/8 backend=$backend seed=$seed offset=$offset ==="
  if [[ -s "$result" ]]; then
    echo "preserving completed result $result"
    continue
  fi
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
    --candidate-split-pages 128 \
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
    --output "$result"
  worker_status=$?
  set -e
  if [[ ! -s "$result" ]]; then
    echo "worker exited $worker_status without a result artifact" >&2
    exit "$worker_status"
  fi
  echo "worker exit=$worker_status; complete artifact preserved for independent performance/quality reduction"
done
