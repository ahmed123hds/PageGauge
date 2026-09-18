#!/usr/bin/env bash
set -euo pipefail

cd /mnt/c/Users/noorg/Ahmed/page_gauge_protocol_v2_rtx5090
export PATH=/home/anonymous/page_gauge_env_protocol_v2/bin:/home/anonymous/cuda13_pcaf_fused/bin:/usr/lib/wsl/lib:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export CUDA_HOME=/home/anonymous/cuda13_pcaf_fused
export LD_LIBRARY_PATH=/home/anonymous/cuda13_pcaf_fused/lib64
export TORCH_EXTENSIONS_DIR=/mnt/c/Users/noorg/Ahmed/page_gauge_protocol_v2_rtx5090/build/torch_extensions_rtx5090
export MAX_JOBS=4

for context in 32768 49152 65536; do
  for split in 0 64 128 256 512; do
    /home/anonymous/page_gauge_env_protocol_v2/bin/python \
      scripts/benchmark_flashinfer_page_affine_int8.py \
      --representation page_gauge \
      --exact-tail 256 \
      --lengths "$context" \
      --baseline-fixed-split-pages "$split" \
      --candidate-fixed-split-pages "$split" \
      --warmup 20 \
      --repeats 80 \
      --cache-scrub-mib 256 \
      --seed 20260816 \
      --output "results/context_sweep_b1/${context}_split_${split}.json"
  done
done
