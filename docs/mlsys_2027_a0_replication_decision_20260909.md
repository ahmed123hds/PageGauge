# A0 replication: evidence and next engineering decision

Development record, 9 September 2026. This is not a final selection manifest.

## Verified result

At RTX5090 B4/C20480/D1536, the declared A0 policy uses S4/A0/T768,
history160/exact32 and unchanged quantizer/kernel mathematics. Two separate
eight-fresh-process comparisons use ABBA/BAAB order, four adjacent pairs and
two exposed TRAIN fixture clusters. Each process has three timing repeats.

| Numerator / denominator | Neutral wall ratio | Hierarchical 95% interval |
|---|---:|---:|
| Reference PageGauge / A0 | 1.015064 | [1.013645, 1.015814] |
| FlashInfer / A0 | 1.175373 | [1.174148, 1.176165] |

Ratios are ratios of execution times, so values above one favor A0. A 1.175x
speedup means approximately 14.92% lower latency, not 17.54% lower latency.
All model decoding work, planner overhead and page finalization remain timed;
prefill, graph capture, restoration and cache scrubbing are outside the interval.
This is not complete-request latency, online serving or independent TEST.
The two-fixture interval does not describe uncertainty across arbitrary workloads.

Served KV bytes: FlashInfer11542724608, reference PageGauge7288520704,
A06214778880. A0 reduces reference PG storage by14.731958%, or46.1585%
relative to FP16. These are KV bytes, not total GPU memory reductions.

## Recovery disclosure

The FI/A0 launcher completed all eight workers but failed at final reduction:
whole-dictionary equality incorrectly required identical backend overhead.
FI uses one wrapper and PG two. Four fields consequently differ:
wrapper_plan_invocations, wrapper_last_page_len_device_fills,
flashinfer_full_plan_calls_all_wrappers and blocking_d2h_metadata_copies.
The separate recovery requires their exact backend-specific counts and equality
of every remaining timed-work field. It does not omit those overheads from timing.
Raw results, original launcher, frozen sources and original failure are preserved.
Original execution validators, source/input hashes and monitoring hashes were
checked again. Four recovery regression tests reject wrong counts, changed
output work and unexplained extra fields while allowing the expected overhead.

FI blocks5/6 returned quality-warning code2: minimum HF cosine0.9886747,
top1 agreement1.0. Their execution validations pass. These historical auxiliary
quality warnings remain in recovered assessments; they are not converted to
unconditional quality passes. No fresh quality claim follows from timing.

Authoritative artifacts under results/mlsys2027_ablation_v1:

- regional_reference_a0_20260909T035751Z_48404aa1/analysis.json
- regional_fi_a0_20260909T053631Z_bcb13816/recovered_analysis.json
  SHA256985B3ED3FE579178902D69E730A4C5139B1C75070B01E9C01EF55028E3F09621
- Recovery source experiments/mlsys2027/ablation_v1/recover_fi_a0.py
  SHA256792063E6AA951DFA9DD1CFF56CD938735391D6B6810ED1237D0A6BE248B6099D

## Decision: promising candidate, not automatic promotion

A0 passes the local speed target and has encouraging earlier two-model PG19
development fidelity. Keep it as a candidate. Do not multiply historical
speed ratios, retroactively replace workshop policies, or tune on final TEST.
The current60-cell reference-policy grid remains unchanged: it answers a
different, already declared generality question and should finish as declared.

## Optimization order after the live grid

1. Identify the actual slow cells and whether extra launches, host planning,
   padding, or cache traffic explain them. Use profiling only for diagnosis;
   measure uninstrumented fresh processes for performance claims.
2. Build an opt-in integration optimization with identical representation and
   workload first. Target repeated per-token dispatch/metadata operations and
   graph-compatible stable buffers. Do not skip page-boundary planning unless
   its complete state transition is correctly moved elsewhere and validated.
3. Validate heterogeneous requests, page close/aging, slot reuse and own greedy
   answers before a timing comparison. A serial loop is not a serving backend.
4. Remeasure full decoder and complete request costs separately. Include setup,
   prefill and all required sidecars in the latter; compare both fixed batch
   and equal memory budget, with documented native baseline constraints.
5. If runtime work does not produce a useful quality/latency/capacity operating
   point, revisit representation on development data. More compression is not
   automatically better, but latency alone does not prove serving superiority.

Paper integration must state both denominators, the repaired analysis, the
quality-warning distinction and the still-missing public-task/serving evidence.
