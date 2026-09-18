# PageGauge INT8: current method and evidence

Status: protocol-v2 implementation validated on the local RTX 3060, 15 August
2026. The strict RTX 5090 publication run is pending. The earlier 5090 archive
is diagnostic only because it mixed cache states in B1, omitted a valid FA4
comparison, and did not measure a pretrained full decoder.

## Contribution that survived testing

PageGauge stores old keys and values as signed INT8 codes with one FP16 scalar
per 16-token page and KV head. A calibration-frozen, per-channel center is
shared by the entire layer/head. The most recent 256 tokens remain FP16.

For page `p`, the representation is

\[
K_p=c_K+s^K_p Z^K_p,\qquad V_p=c_V+s^V_p Z^V_p,
\]

where `Z` is INT8, `c` is token-invariant, and each `s_p` is scalar over the
whole 16 by 128 page/head tile.

The important property is not merely quantization. It is that every affine
term can be moved outside the tensor-core contractions:

\[
\operatorname{softmax}(qK^T)
=\operatorname{softmax}(q(K-c_K)^T),
\]

because `q c_K^T` shifts every logit by the same constant. Within each page,

\[
q(s^K_p Z^K_p)^T=s^K_p(qZ_p^{K\,T}),
\]

and, since softmax probabilities sum to one,

\[
PV=c_V+\sum_p s^V_p(P_pZ^V_p).
\]

The patched FlashInfer FA2 kernel loads the one-byte codes, uses FlashInfer's
existing INT8-to-FP16 register cast, and performs the tensor-core MMA on those
unscaled code values. It scales completed QK fragments by `sK_p`, scales the
page probability fragment by `sV_p` before PV MMA, merges the exact-tail state,
and restores `cV` once. It never performs an elementwise affine reconstruction
of the actual FP16 K or V values in the hot loop. This prototype is therefore
an INT8-storage/factorized-FP16-MMA path, not a native INT8 tensor-core path.

This is the narrow novelty claim. Exact recent/sink windows, channel
normalization, and KV quantization themselves are established techniques.

## Why the speedup is near 1.8x rather than 5x

For a 16 by 128 page/head tile, FP16 K or V uses 4096 bytes. PageGauge uses
2048 INT8 bytes plus one 2-byte scale, or 50.049% of the old-cache bytes. With
the 256-token FP16 tail in the frozen mixed workload, the measured byte
fraction is 0.5101 and the optimistic bandwidth-only ceiling is 1.9604x.

The remaining gap to that ceiling is explained by exact-tail work, softmax,
page-table reads, state merging, output writes, and cache-page finalization.
The representation cannot produce 5x exact dense-attention speedup because it
does not remove enough bytes or arithmetic to permit it.

## Protocol-v2 local screening

Environment: RTX 3060 12 GB (SM86), PyTorch 2.10.0+cu128, FlashInfer 0.6.17,
FA2 tensor-core decode, page size 16, Hq/Hkv/d = 32/8/128. Protocol v2 scrubs
256 MiB of unrelated GPU memory before each cache-neutral sample, primes the
same method before each cache-hot sample, and alternates A/B then B/A within
each regime. Scheduler selection and headline results use only cache-neutral
samples.

The reduced local end-to-end rehearsal used three seeds and 40 samples per
seed. It is a functionality screen, not the five-seed RTX 5090 result:

| Workload | Cache-neutral attention | Cache-neutral inclusive | Cache-hot inclusive |
|---|---:|---:|---:|
| B8 ragged: 49K, 2K, 4K, ..., 14K | 1.8121x | **1.8070x** | 1.8193x |
| B1, 49K | 1.8170x | **1.8097x** | 1.7993x |

The fused append/finalize incremental median was 2.033 microseconds per B8
decode step and 1.344 microseconds at B1. This replaces the old accounting
model that charged completed-page quantization as a separate launch. The
full-decoder implementation now keeps only the 256-token exact FP16 tail in a
ring buffer; completed old pages remain only in the full INT8 cache.

The full-decoder path was also exercised with a randomly initialized one-layer
Mistral-shaped smoke model. It passed logits correctness but reached only
0.9593x cache-neutral tokens/second across three seeds, so it is not positive
serving evidence. At 4K its bounded PageGauge cache used 56.35% of the FP16
cache bytes. A separate 32-token boundary test crossed two page-table
shifts with minimum logits cosine 0.999981 and 32/32 top-1 agreement. The
pretrained 32-layer Mistral-7B run on RTX 5090 remains the decision experiment.

Local screening artifacts:

- `results/page_gauge_int8/protocol_v2_smoke_rtx3060_20260815/FINAL_SUMMARY.json`
- `results/page_gauge_int8/protocol_v2_smoke_rtx3060_20260815/b1_49k/B1_SUMMARY.json`
- `results/page_gauge_int8/transformer_ring_smoke_rtx3060_20260815.json`
- `results/page_gauge_int8/boundary_smoke_rtx3060_v3.json`

## Model-quality evidence

These are eager-attention simulations using calibration-frozen centers, exact
FP32 score accumulation, PageGauge INT8 for old pages, and an exact 256-token
tail. They validate the representation, not the CUDA latency path.

| Model / held-out text | Context | Delta mean NLL | FP16 top-1 agreement | Sampled KL mean |
|---|---:|---:|---:|---:|
| Qwen2.5-1.5B / paper | 1K | +0.00121 | 99.707% | 0.0000595 |
| Qwen2.5-1.5B / code | 4K | +0.00095 | 99.096% | 0.000340 |
| Qwen2.5-0.5B / paper | 2K | +0.00107 | 98.974% | 0.001461 |
| Qwen2.5-0.5B / code | 2K | -0.00108 | 99.560% | 0.000592 |
| TinyLlama-1.1B / code | 2K | +0.00166 | 99.805% | 0.0000123 |

On real post-RoPE Q/K/V captured from all 28 Qwen2.5-1.5B layers at a 4104
token final decode state, the factorized CUDA kernel differs from an explicitly
reconstructed PageGauge cache by only 0.215% p95 relative L2. Quantization, not
the factorization, dominates the remaining error. Preserving the first 16
tokens exactly reduces final-query p95 attention-output error from 6.20% to
0.87%, but did not consistently improve every full-sequence logit metric; it is
kept as a decode-specific ablation and is not claimed as novel.

## Falsified alternatives

- Reconstructing an affine scale/bias for every K/V fragment was slower on the
  ragged workload. Scale factorization is essential.
- Randomized Hadamard gauges did not improve the no-sink real-attention error.
- A page/head affine offset improved p95 error only modestly and requires extra
  QK/PV reductions.
- Clipping the page range by even 10% severely damaged key logits.
- Extra V channel-group scales gave only a small quality improvement and would
  complicate the PV path.

## Prior-art boundary

[InnerQ](https://arxiv.org/html/2602.23200) is the closest work found. It
already uses inner-dimension groups, recent and sink FP16 windows, key-channel
normalization, and fused CUDA dequantization, and reports 2.7x over FP16 on a
Jetson platform. Its paper describes dequantizing cache rows inside fused GEMV;
PageGauge instead chooses a coarser page/head scalar specifically so both K and
V scales commute through QK and PV and no elementwise affine dequantization
remains beyond the code-to-FP16 register cast.
A direct implementation-level comparison is required before making a
state-of-the-art claim; no official InnerQ code link was exposed by the paper
at the time of this audit.

Other required baselines are [SKVQ](https://arxiv.org/abs/2405.06219),
[KIVI](https://github.com/jy-yuan/KIVI), and
[KVQuant](https://github.com/SqueezeAILab/KVQuant). FlashAttention-4 is an
exact dense-attention kernel optimized for Hopper and Blackwell. Its official
`flash-attn-4` package exposes `flash_attn_varlen_func` with a page-table
argument. The protocol probes that exact API at B1/49K. The comparison is
marked failed if the installed FA4 release rejects paged KV on the physical
architecture; a contiguous fallback is not mislabeled as paged decode.

## Reproduce on another GPU

Use the new `bundle/page_gauge_protocol_v2_rtx5090.zip`, not the older portable
archive, and a fresh output directory for every device. FlashInfer 0.6.17 and a
CUDA toolkit with `nvcc` are required. On CUDA 13 install the official FA4
extra with `python -m pip install "flash-attn-4[cu13]"`.

```bash
python scripts/run_page_gauge_validation.py \
  --output-dir results/page_gauge_RTX5090_PROTOCOL_V2
```

The strict default runner verifies and applies the FlashInfer patch, runs CUDA
tests, selects FP16 and INT8 schedulers independently from cache-neutral pilot
data, executes five validation seeds for B8 and B1 in both cache regimes,
measures fused page finalization, requires the matching paged FA4 probe, runs a
pretrained Mistral-7B full decoder, writes raw samples and checksums, and creates
a ZIP archive. It preserves the ZIP before returning a nonzero status for a
missing required comparison.

## Claims that are not yet supported

- No protocol-v2 RTX 5090, A100, or RTX PRO 6000 Blackwell result has been
  returned yet.
- The pretrained full-decoder runner measures every layer, LM head, and cache
  update, but it is not a vLLM/TensorRT-LLM production server and excludes
  prefill, tokenizer, sampling, networking, and request scheduling.
- No LongBench, GSM8K, or task-accuracy matrix has been run with the actual
  cache implementation.
- No direct InnerQ kernel benchmark is available.
- A modern paged FA4 comparison is unsupported until the strict probe succeeds
  on the target GPU. Failure is explicit in `FA4_COMPARISON.json` and makes
  `publication_complete` false.
