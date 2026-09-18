# MLP-only graph pilot: do not promote

Evidence: `dense_pair_20260909T135456Z_99e77c36`, B4/C20480/D1536,
historical S4/A128/T768 reference policy, original common eager attention.
One fresh process per backend, full warmup, three repeats; fixed FI/PG order.

| Backend | Median ms/step | Repeat range |
|---|---:|---:|
| FlashInfer FP16 | 21.143862 | 20.150778–21.283330 |
| PageGauge | 28.579978 | 23.432130–28.642581 |

FI/PG point ratio is 0.739814, not a speed win. Keep every repeat. This pilot
does not establish a confidence interval or a causal regression against the
older eager run, which was measured at a different time.

Both monitored workers exited zero. Result, log and telemetry hashes and all
four rounds of 32×1536 graph calls were rechecked. Prior full-recurrence
validation included 192 bitwise MLP comparisons per backend.

## What the records establish

- No allocator retries or OOMs in the recorded final-memory snapshots.
- Wall and CUDA-event elapsed times agree closely. CUDA events include GPU
  idle gaps caused by host dispatch; agreement does not prove GPU saturation.
- Each additional retained graph round adds 2,129,920 allocated bytes for
  either backend. This is a diagnostic retention issue, not evidence that it
  caused the several-millisecond difference. Device/context overhead also grows.
- Ownership telemetry is sampled process ownership, not clock, power or thermal
  telemetry. It cannot rule out frequency changes or other host contention.
- The graph wraps only the MLP. Attention planning, append, projections,
  normalization and inter-module dispatch remain outside that graph. Per-call
  parameter traversal, input copy and shape/stream guards remain timed.

## Next experiment, before another speed claim

Instrument the actual new dispatch path (not the old eager-only profile): one
full recurrence with the last129 steps profiled, FI and PG, preserving the same
fixtures. Separate CPU launch gaps, graph replay/input-copy costs and attention
kernel/merge work. Do not use instrumented timings as acceptance evidence.
Then implement a targeted change supported by that attribution. Release old
per-round graph objects in a new runner version, without altering these frozen
artifacts. Complete-layer integration is a candidate, not an established fix.

The independent optimized A0 result is unaffected; this unsuccessful reference
integration must not be presented as a rerun of A0. No final TEST was used.

## Follow-up: combined attention and MLP graphs

`combined_pair_20260909T165640Z_33c556e7` completed after both full-recurrence
validations. FI median19.179980ms (19.104965–19.442358), PG18.447448ms
(18.211110–19.629452), FI/PG1.039709. The ranges overlap and the point is below
1.1. No promotion to the speed target. `audit_combined_pair.py` revalidated
source/input evidence, actual tokens, monitoring hashes, four complete graph
trajectories and exact reduction (exit0).

The combined implementation also switches back to graph-compatible attention
planning. Consequently this is not an isolated causal estimate of attention
replay versus the preceding non-graph-planning MLP pilot. Historical policies,
math and kernels are unchanged. Capture/preparation remain excluded.

Next integration scope: group stateless input RMSNorm plus separate Q/K/V
projections, and separately output projection/residual plus post-attention
RMSNorm/MLP/residual, while retaining dynamic append/planning outside capture.
Keep the original operation order and weights; do not introduce packed weight
transforms or quantizer changes. This needs multi-input/multi-output stable
buffers and explicit residual lifetimes, so the current single-tensor MLP
primitive is insufficient. Qualify these boundaries against eager outputs
before full-model measurement. Do not launch another identical combined pilot
without a concrete implementation change or predeclared replication purpose.

## Wider layer segments: positive development pilot

After full-recurrence validation, the first timing attempt completed warmup but
failed during second cache restoration. The separate v2 runner collects old
driver/graph cycles after releasing layer callbacks, outside timed decode.
Both v2 backends completed every round; the original failure remains retained.

FI `layer_segment_timing_v2_flashinfer_fp16_20260909T202310Z_c66466e0`:
17.993354ms median, range17.991941–18.010292. PG
`layer_segment_timing_v2_page_gauge_20260909T202623Z_98bf76e8`:
15.516090ms median, range15.510329–15.538226. FI/PG **1.159658**.
Served KV bytes remain11542724608/7288520704 (36.85615% reduction).
`reduce_layer_timing.py` verified evidence and exact reduction, exit0.

This is the historical reference policy, not A0, and the first promising wider
integration point. Freeze it for independent fresh-process ABBA/BAAB replication
on disjoint development fixtures; do not call this a confidence-bound pass.
No speed claim includes prefill or capture, and no actual online serving result
is established by this pilot.
