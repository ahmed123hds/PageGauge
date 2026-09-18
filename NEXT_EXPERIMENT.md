# PageGauge next experiment: strict-quality outlier localization v1

## Question

The systems path, same-backend eager/graph equivalence, cache mutation,
page-finalization, and sustained speed smoke already pass. The remaining
failure is the preregistered worst-row HF-vs-PageGauge logit-cosine gate.

This experiment does **not** change the threshold, quantizer, exact window,
kernel, graph boundary, or timing protocol. It localizes the original
`exact_tail=256, D=512` failures before another representation change is made.

The diagnostic records:

- every `[decode step, request]` cosine, relative-L2, maximum absolute error,
  top-1 match, and reference/candidate logit norm;
- the lowest-cosine rows with token IDs, top logits, top-1 margins, page offset,
  and the number of runtime-generated pages already visible through INT8;
- per-request minima;
- per-page-offset failure counts;
- consecutive below-threshold clusters.

The strict endpoint remains:

```text
minimum logit cosine >= 0.995
and top-1 agreement >= 0.80
```

The new report is posthoc evidence only. Do not use latency from this diagnostic
as the paper timing result.

## Apply the overlay

Extract the supplied overlay ZIP at the existing PageGauge repository root,
allowing it to replace the two existing files and add the new analyzer and this
instruction file.

Expected changed/new files:

```text
diagnostics/benchmark_sustained_dynamic_graphs.py
diagnostics/analyze_quality_outliers.py
tests/test_sustained_dynamic_graph_protocol.py
NEXT_EXPERIMENT.md
pagegauge_quality_outlier_localization_v1.patch
```

## Preflight in WSL

```bash
cd /mnt/c/Users/noorg/Ahmed/page_gauge_protocol_v2_rtx5090

export TORCH_EXTENSIONS_DIR="$PWD/build/torch_extensions_rtx5090"
export MAX_JOBS=4

python -m py_compile \
  diagnostics/benchmark_sustained_dynamic_graphs.py \
  diagnostics/analyze_quality_outliers.py \
  tests/test_sustained_dynamic_graph_protocol.py

pytest -q tests/test_sustained_dynamic_graph_protocol.py
```

The focused suite should pass before the GPU run.

## Run the next GPU experiment

Use a new output directory and the already validated configuration that
reproduced the original six low-cosine rows:

```bash
cd /mnt/c/Users/noorg/Ahmed/page_gauge_protocol_v2_rtx5090

export TORCH_EXTENSIONS_DIR="$PWD/build/torch_extensions_rtx5090"
export MAX_JOBS=4

mkdir -p results/quality_outlier_localization_v1

python diagnostics/benchmark_sustained_dynamic_graphs.py \
  --backend page_gauge \
  --model mistralai/Mistral-7B-v0.3 \
  --batch-size 4 \
  --context 20480 \
  --decode-steps 512 \
  --exact-tail 256 \
  --prefill-chunk-tokens 1024 \
  --candidate-split-pages 256 \
  --tail-attention flashinfer_merge \
  --trajectory-mode frozen_hf_teacher_forced \
  --seed 20260861 \
  --token-source wikitext2 \
  --warmups 0 \
  --repeats 1 \
  --quality-diagnostics-top-k 64 \
  --quality-diagnostics-top-vocab 12 \
  --output results/quality_outlier_localization_v1/page_gauge_tail256_d512_seed20260861.json \
  2>&1 | tee results/quality_outlier_localization_v1/page_gauge_tail256_d512_seed20260861.log
```

A nonzero process exit is expected if the unchanged strict quality gate still
fails. The JSON should still be preserved. Do not rerun merely because the
worker reports `passed=false`.

## Render the compact report

```bash
python diagnostics/analyze_quality_outliers.py \
  --input results/quality_outlier_localization_v1/page_gauge_tail256_d512_seed20260861.json \
  --output results/quality_outlier_localization_v1/page_gauge_tail256_d512_seed20260861_outliers.md \
  --rows 32

sed -n '1,240p' \
  results/quality_outlier_localization_v1/page_gauge_tail256_d512_seed20260861_outliers.md
```

## Bundle only the files needed for review

```bash
cd results/quality_outlier_localization_v1

python -m zipfile -c page_gauge_quality_outlier_results_seed20260861.zip \
  page_gauge_tail256_d512_seed20260861.json \
  page_gauge_tail256_d512_seed20260861_outliers.md \
  page_gauge_tail256_d512_seed20260861.log

sha256sum page_gauge_quality_outlier_results_seed20260861.zip
```

Share that result ZIP.

## Decision after the result

1. **Failures occur before any runtime-generated page becomes INT8**
   (`generated_int8_pages_visible == 0`): focus on the initial-prefix
   representation, centers/scales, and K-vs-V contribution. Increasing the
   recent generated-token tail is not the right remedy.
2. **Failures begin only after generated pages age into INT8**: trace runtime
   page finalization and recurrent quantization first.
3. **Failures concentrate at one page offset**: run a boundary-specific append
   and attention-state trace.
4. **One request contains isolated failures with normal logit norms and large
   top-1 margins**: run targeted layer-by-layer and K-only/V-only attribution
   on those exact rows.
5. **Failures have unusually low logit norms or tiny top-1 margins**: document
   the geometry, but do not weaken the gate. The next experiment still needs to
   determine where the distribution error enters.

Do not run the four-seed publication ABBA grid until this localization is
reviewed.
