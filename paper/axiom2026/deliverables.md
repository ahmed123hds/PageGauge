# Final deliverables

## 1. Three candidate titles

1. **PageGauge: Factoring Affine KV Quantization Through Softmax Attention**
2. **Quantize for the Consumer: Attention-Compatible KV Compression Without Elementwise Reconstruction**
3. **Algebra Before Bits: Pagewise Affine KV Compression for Factorized Attention**

Selected title: **PageGauge: Factoring Affine KV Quantization Through Softmax Attention**.

## 2. Final abstract

Quantization is usually designed as a representation and only later adapted to
the operation that consumes it. We study the opposite direction for
autoregressive attention. PageGauge uses token-independent channel centers and
page/head scalar scales so that affine metadata cancels, commutes, or factors
through QK, softmax, and PV. The resulting kernel consumes historical INT8
codes without materializing elementwise affine-reconstructed FP16 KV, then
combines them with exact FP16 regions by merging online-softmax states. The
transformation is exact in real arithmetic relative to the reconstructed
mixed-precision cache; original-model error is therefore separated from
floating execution error. On Mistral-7B-v0.3 at batch 4, 20,480-token context,
and 1,536 decode steps on an RTX 5090, the final policy stores 7.289 GB of
served KV versus 11.543 GB for FP16, a 36.86% reduction. A six-window,
9,216-logit-vector B1 fixed-policy confirmation obtains minimum PageGauge/FlashInfer
cosine 0.998707 and 99.891% top-1 agreement while 48 runtime-appended pages age
into INT8. Eight fresh-process ABBA/BAAB B4 blocks yield 15.214 versus 17.232 ms
per sustained complete-decoder step, or 1.1326x speedup (within-protocol
hierarchical 95% CI 1.1323-1.1328). The
quality cohort follows a disclosed post-failure policy repair and is not an
untouched held-out selection. These results support a broader design rule:
quantization metadata should be chosen together with the algebra and
invariances of its consuming operator.

## 3. Exact four-page outline

| Main-paper pages | Content | Budget |
|---|---|---:|
| 1.00-1.65 | Introduction, concise related work, thesis, exactly three contributions, Figure 1 | 0.65 page |
| 1.65-3.00 | Affine representation, Proposition 1, K/V derivation, mixed exact/quantized online merge, error decomposition | 1.35 pages |
| 3.00-3.45 | FlashInfer-compatible realization, exact precision layout, byte and Amdahl-style cost models | 0.45 page |
| 3.45-4.70 | Evaluation setup, four questions, Table 1, compact error-propagation result, adaptation disclosure | 1.25 pages |
| 4.70-5.00 | Limitations and conclusion | 0.30 page |

The rendered submission uses exactly four main-paper pages before references;
references and the appendix begin afterward.

## 4. Main figure specification

**Figure 1: Quantization designed for the consuming attention algebra.**

- Left, conventional fused path: low-bit K/V plus per-element affine
  reconstruction in registers, followed by FP16 QK/PV MMA.
- Right, PageGauge path: FP16 MMA on converted code values; page/head key and
  value scales applied around fragments (production V scaling is probability-side); shared key-center cancellation
  and single value-center restoration; online `(m,l,o)` merge with exact FP16
  regions. Label code conversion to arithmetic registers without using the
  unqualified term "dequantization-free."
- Bottom, endpoint cache layout over 1,376 pages/request: exact prefix pages
  0-3; quantized historical pages 4-1151; static exact prefill suffix pages
  1152-1279; 48 runtime-appended pages 1280-1327 aged to INT8; exact recent tail pages
  1328-1375. State 1,196/1,376 logical pages quantized (86.92%) and gross served
  storage 6.788 GiB because code/scale allocation is retained for exact pages.

The figure should communicate the distinction in under 30 seconds: conventional
systems fuse reconstruction with attention; PageGauge constrains metadata so
the affine terms factor through attention.

## 5. Main table populated from final artifacts

| Backend | B4 served KV | KV reduction / KV-byte ratio | B4 sustained ms/step | B4 speedup (95% hierarchical CI) | B1 fidelity | B1 top-1 | B1 recurrence |
|---|---:|---:|---:|---:|---:|---:|---|
| Optimized FlashInfer FP16 | 11.543 GB (10.750 GiB) | reference / 1.000x | 17.2318 | 1.000x | 0.999570 vs HF | 0.999132 vs HF | FP16 KV |
| PageGauge S4/A128/T768 | 7.289 GB (6.788 GiB) | 36.856% / 1.5837x | 15.2141 | 1.1326x [1.1323, 1.1328] | 0.998707 vs FI | 0.998915 vs FI | 48 runtime-appended pages consumed INT8 |

Timing is cache-neutral wall time from four adjacent pairs across two
fixture/seed clusters, with three repeats inside each of eight fresh-process
blocks. The interval is a 50,000-draw hierarchical fixture-then-pair bootstrap.
Fidelity uses the separate six-window, 9,216-logit-vector B1 post-adaptation
fixed-policy confirmation; memory and latency use the B4 performance protocol.
