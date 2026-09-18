# PageGauge protocol-v2 RTX 5090 validation

This is the replacement runner for the earlier RTX 5090 package. Do not merge
its results with an old output directory. It contains source and tests, not
precompiled GPU binaries.

The default run is intentionally strict: it requires a working
FlashAttention-4 paged-KV comparison and a pretrained Mistral-7B full-decoder
run. If either check is unavailable or incorrect, the runner first preserves
all diagnostics in a ZIP and then exits nonzero.

## Environment and installation

Use native Linux or WSL2 with Python 3.10+, a CUDA-enabled PyTorch build, the
matching CUDA toolkit with `nvcc`, and a driver new enough for that toolkit.
Both GeForce RTX 5090 and RTX PRO 6000 Blackwell should report compute
capability 12.0. Do not force an older CUDA architecture.

```bash
python - <<'PY'
import torch
print("torch", torch.__version__)
print("torch CUDA", torch.version.cuda)
print("GPU", torch.cuda.get_device_name())
print("capability", torch.cuda.get_device_capability())
print("memory GiB", torch.cuda.get_device_properties(0).total_memory / 2**30)
PY
nvcc --version
```

Install dependencies without replacing a suitable GPU-specific PyTorch build:

```bash
python -m pip install --upgrade -r requirements-page-gauge.txt
python -m pip install --upgrade "flash-attn-4[cu13]"
python -m pip check
```

The `cu13` extra is the official recommendation for CUDA 13. For a CUDA 12.x
PyTorch/toolkit, install `flash-attn-4` without that extra. The runner imports
`flash_attn.cute.flash_attn_varlen_func`; merely having the older
`flash-attn` package is not sufficient.

As of 15 August 2026, the official FA4 interface still explicitly rejects
paged KV on SM120 even though non-paged varlen attention supports SM120. Thus a
current RTX 5090 installation may compile successfully and still produce an
`unsupported_architecture` comparison. Preserve that result: replacing it with
contiguous FA4 would answer a different question. A later FA4 release may lift
the restriction, which is why the runner probes the installed implementation
rather than hard-coding failure.

## Publication run

Run from the extracted bundle root, with a new output directory:

```bash
export TORCH_EXTENSIONS_DIR="$PWD/build/torch_extensions_rtx5090"
export MAX_JOBS=4

python scripts/run_page_gauge_validation.py \
  --output-dir results/page_gauge_RTX5090_PROTOCOL_V2
```

The default full-model workload is `mistralai/Mistral-7B-v0.3`, context 16K,
16 decode tokens, three seeds, and both cache conditions. Downloading the model
requires network access on the first run. If it is already cached, add
`--transformer-local-files-only`.

The run can take a while because FlashInfer, the fused append extension, and
FA4 compile device-specific kernels. Successful completion prints an archive
path and SHA256. Return the complete ZIP, even when the command exits nonzero;
the archive contains the exact failure record.

## What protocol v2 changes

1. Every benchmark reports two separate regimes: cache-neutral/cold samples
   preceded by a 256 MiB unrelated GPU scrub, and cache-hot samples preceded by
   an untimed call to the same method.
2. Method order alternates A/B then B/A inside each regime. All scheduler
   selection and headline summaries use cache-neutral timings only.
3. B1/49K selects FP16 and PageGauge split schedulers independently; it cannot
   inherit the previous candidate-after-baseline cache advantage.
4. A CUDA extension fuses RoPE, centered K/V append, and completed-page INT8
   finalization into the cache-update launch. Only the 256-token exact tail is
   retained in an FP16 ring; the baseline uses the matching fused RoPE plus
   FP16 append launch.
5. FA4 is run at the same B1/49K paged-decode shape. Missing, unsupported,
   runtime-failing, or numerically incorrect FA4 is an explicit failed
   comparison, never a silently omitted baseline.
6. The pretrained decoder benchmark measures every transformer layer, LM head,
   cache update, per-token wall latency, GPU latency, and wall tokens/second.
   It separately profiles the first decode token and a page-closing token.

## Decision files

- `SYSTEMS_VALIDATION_STATUS.json`: the first file to inspect;
  `publication_complete` is true only when FA4 and full-model checks pass.
- `FINAL_SUMMARY.json`: B8 ragged multi-seed cache-neutral and cache-hot result.
- `b1_49k/B1_SUMMARY.json`: corrected B1/49K multi-seed result.
- `FROZEN_SELECTION.json` and `b1_49k/FROZEN_SELECTION.json`: independent
  cache-neutral scheduler decisions.
- `page_finalization_overhead.json`: fused append/finalize incremental cost.
- `b1_49k/FA4_COMPARISON.json`: explicit three-way FlashInfer, PageGauge, and
  FA4 comparison or structured failure.
- `transformer/FULL_MODEL.json`: per-seed layer latency, end-to-end latency,
  tokens/second, and logits checks.
- `RUN_CONFIG.json` and `RUNNER_SOURCE_SHA256.json`: exact command parameters
  and hashes of every source file on the measured path.
- `SHA256SUMS.json`: digest manifest for every returned artifact.

## Optional functional smoke

This checks plumbing only and is not paper evidence:

```bash
python scripts/run_page_gauge_validation.py \
  --output-dir results/page_gauge_RTX5090_PROTOCOL_V2_SMOKE \
  --pilot-repeats 20 --repeats 40 --warmup 10 \
  --validation-seeds 20261001 20261002 20261003 \
  --fa4-policy record-only \
  --transformer-policy record-only \
  --transformer-model synthetic-mistral-page-gauge-smoke \
  --transformer-context 4096 --transformer-decode-steps 16 \
  --transformer-warmups 1 --transformer-repeats 2 \
  --transformer-seeds 20261010 20261011 20261012
```
