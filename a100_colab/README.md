# PageGauge A100 / Colab cross-GPU confirmation

This bundle recompiles the unchanged PageGauge attention algebra for NVIDIA
Ampere `sm_80` and runs a matched, backend-exclusive cross-GPU confirmation.
It does not reuse RTX 5090 cubins or extension caches.

`SM80_BUILD_AUDIT.json` records a clean host-side cross-compilation of both
the selected PageGauge FA2 module and the append/finalize extension into
`sm_80` cubins. Those cubins are deliberately not shipped: Colab rebuilds all
extensions from the included, hash-attested source on the assigned A100.

## Hardware contract

The runner accepts a full NVIDIA A100 40GB or 80GB only:

- CUDA compute capability 8.0;
- exactly 108 SMs; and
- at least 38 GiB device memory.

This intentionally rejects A100 MIG slices. Google Colab does not guarantee a
specific accelerator, so select an A100 runtime and let the preflight fail
closed if Colab assigns another GPU.

## Why the A100 schedule differs

The final PageGauge policy is unchanged: Mistral-7B-v0.3, B4, C20480, D1536,
four exact prefix pages, 128 exact original-prefix suffix pages, and a 768-token
exact tail. The quantizer, factorized attention, merge, graph scope, and timing
boundary are unchanged.

The A100 has 108 SMs rather than the RTX 5090's 170. For Hkv=8 and B=4,
FlashInfer's FA2 split-K padded capacity admits at most six chunks per request.
The RTX 5090 candidate split of 128 pages would require ten chunks per request
and is therefore invalid on A100. The baseline remains at split 256
(`ceil(1376 / 256) = 6`). For PageGauge, the architecture-derived minimum safe
split is `ceil(1196 / 6) = 200` pages, which uses all six available chunks per
request instead of the five produced by split 256. This is a preregistered
scheduler-capacity adaptation, not a parameter sweep or attention-math change.

The preliminary B1 old-only kernel smoke is a compile/correctness diagnostic,
not the speed gate. At B1/split256 it exposes only 5 of 27 padded scheduling
slots and can underutilize A100. Publication speed is determined only by the
B4, 1,536-step, eight-fresh-process end-to-end reduction below.

## A100 launch-shape recovery probe

If the unmodified SM80 result is slower than FlashInfer, do not repeat the
eight model blocks.  Run the model-free production-geometry probe first:

```bash
bash a100_colab/run_sm80_kernel_probe.sh
```

It compares the frozen launch against two independently derived Ampere
adaptations and their combination: an exact-wrapper split of 30 pages (six
chunks/request, 192 CTAs instead of 32) and a PageGauge `NUM_MMA_KV=2` cap.
The cap removes the audited 88-byte/thread spill in the SM80 `MMA4`
specialization.  Neither adaptation changes cache bytes, quantization,
attention equations, exact regions, or acceptance thresholds.  The probe uses
B4 and the exact final S4/A128/T768 geometry, but no model weights.

Only when the reducer finds a faster correct variant, run one adjacent
fresh-process end-to-end point gate:

```bash
bash a100_colab/run_sm80_optimized_pair.sh
```

That pair must exceed 1.10x before spending time on a new eight-block
ABBA/BAAB confirmation.  A passing pair is a screening result, not the final
hierarchical-confidence-interval claim.

Once the source-frozen pair passes, run the final Williams confirmation:

```bash
bash a100_colab/run_sm80_optimized_confirmation.sh
```

It requires the exact selection artifact SHA256
`914566c0704735d5d4dddb6df780d377785ecc306ee7f66cfe34532755bd680e`,
runs eight fresh processes in ABBA then BAAB order, and accepts only when both
the cache-neutral wall-clock point estimate and hierarchical 95% lower bound
exceed 1.10x.

## Colab workflow

1. In Colab choose **Runtime > Change runtime type > A100 GPU**.
2. Upload `pagegauge_a100_colab.zip` and `PageGauge_A100_Colab.ipynb`.
3. Open the notebook and run its cells in order.

The setup downloads the pinned public Mistral snapshot into the normal
Hugging Face cache. If Hugging Face requests authentication, set `HF_TOKEN` in
the Colab environment before the setup cell.

The full command-line equivalent after unzipping is:

```bash
bash a100_colab/setup_colab.sh
bash a100_colab/run_kernel_smoke.sh
bash a100_colab/run_cross_gpu_confirmation.sh
```

The final reducer writes
`results/a100_cross_gpu_s4_a128_t768/final_a100_analysis.json` and exits nonzero
unless all implementation gates pass and cache-neutral end-to-end wall speedup
has both point estimate and hierarchical 95% lower bound above 1.10x.

## Optional PG-19 quality confirmation

The upload also includes the six preregistered PG-19 test books and frozen
selection manifest used by the external-corpus quality protocol. After the
performance run completes, the independent A100 quality confirmation is:

```bash
bash a100_colab/run_pg19_quality.sh
```

This sequential B1 run is intentionally outside the performance experiment.
It writes `results/a100_pg19_s4_a128_t768/a100_pg19_gate.json`; it neither
changes nor contributes samples to the eight-block speed result.

## What is measured

- a pure factorization-vs-explicit-reconstruction long-context kernel gate;
- eight fresh Python/CUDA worker processes in ABBA then BAAB order;
- one warmup and three measured 1,536-step samples per block;
- complete decoder-layer device-dynamic graph replays, page planning,
  finalization, embedding, LM head, and GPU argmax;
- same-backend eager/graph/cache/canary/recurrence gates; and
- deterministic served-cache byte accounting.

Model download, prefill, graph capture, cache restoration, scrub, and hot
preconditioning remain outside the measured interval exactly as in the RTX
5090 protocol.
