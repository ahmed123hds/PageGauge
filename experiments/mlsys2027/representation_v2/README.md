# Shared-channel representation prototype

Development experiment, 2026-09-08. Original production code, workshop PDF,
Step 02 source closure, and completed robustness_v1 evidence are unchanged.
This directory is a separate prototype, not the new production default.

## Change

The original scalar page/head format becomes

`Vhat_p = 1 cV^T + sV_p ZV_p diag(gV)`.

The gain vector is shared across all pages for one request/layer/KV head. Compute
prefill-centered RMS by channel, divide by its channel geometric mean, round the
log2 gain to the nearest integer, and clamp its exponent to [-8,8]. Store powers
of two in FP16. Fit once from the original initial prefill, never from future
decode tokens or attention queries. No hyperparameter sweep was performed.

This allocates the INT8 range differently by channel; it does not increase code
precision or extend the exact regions. Scalar page/head scales remain scalar.
The CPU experiment also evaluates K-only and joint K/V conditioning, but these
increased mean output error on this development cohort.

## Algebra and GPU realization

For the historical segment, its normalized centered output is
`o_old = (P_old sV ZV) diag(gV)`. The kernel computes the parenthesized quantity
using its unchanged code-valued FP16 contractions. The GPU prototype scales this
segment output channel-wise before merging it with the unchanged exact segment.
It then restores the common value center once. LSE does not depend on V, so the
same softmax-state merge weights apply. No key/query transformation is needed
for the value-only candidate.

This avoids scaling or re-rounding the exact FP16 cache. Exact regions, centers,
page partitions, code bit width and old-value scalar placement remain unchanged.
FP16 arithmetic still has rounding and range limitations; mathematical equivalence
does not imply bitwise equivalence. Channel conditioning itself is not claimed new.

The prototype adds a separate GPU multiply per historical attention output;
production would need to account for or fuse that operation. It also adds gain
fitting and division when encoding/finalizing pages. Performance is NOT measured.
Gain metadata at B4/L32/Hkv8/D128 is 262,144 bytes for V only (524,288 for K+V),
before implementation-specific expanded gain buffers and scratch space. No claim
of unchanged total memory or unchanged speed is made.

## Evidence and interpretation

CPU results: `results/mlsys2027_representation_v2/20260908T053301Z_2ad8ce32`.
All nine existing TRAIN captures, three layers and three decode snapshots,
72 KV-head cases / 288 query-head outputs, are reported.

| FP64 attention representation | Mean output L2 | Worst output cosine | Outputs improved / worsened vs baseline |
|---|---:|---:|---:|
| Original scalar format | 0.00494968 | 0.840102 | reference |
| K conditioned | 0.00532636 | 0.839909 | 129 / 159 |
| V conditioned | 0.00472930 | 0.997257 | 147 / 69 (72 unchanged) |
| K+V conditioned | 0.00507879 | 0.997241 | 170 / 118 |

For V conditioning, mean L2 falls 4.45%, p95 relative L2 falls from 0.28569 to
0.04783, but maximum L2 rises from 0.02338 to 0.02545. This is not uniformly
better. Cosine improvements in low-norm head outputs need downstream validation.
All CPU factorizations agree with explicit reconstruction to < 5e-15 max error.

The V-only candidate was advanced to GPU replay AFTER inspecting these results.
This is disclosed development selection, not held-out confirmation. It does not
override robustness_v1's prohibition on selecting a production policy from one
diagnostic window. Do not change the final configuration based solely on this run.

GPU replay completed for all nine captures on RTX 5090:
`results/mlsys2027_representation_v2/gpu_20260908T053519Z_cb1b89a1`.
Mean L2 against the original-input FP64 output is 0.00483634 (baseline GPU:
0.00505448, approximately 4.32% lower); minimum cosine is 0.996969 (baseline:
0.840116). Candidate GPU vs candidate explicit FP16 reconstruction minimum cosine
is 0.999971 and max absolute difference is 0.001953125. These are attention-head
outputs on fixed HF trajectories, not final logits, task scores, or PPL.

## Files and next decision

- `prototype.py`: fixed-rule encoder, K/V ablations, original-input controls,
  all-head CPU error reporting, and saved frozen prefill gains.
- `gpu_replay.py`: untimed value-conditioned execution through existing kernels,
  same-input explicit FP16 control, and original-input FP64 reference.
- `tests/test_pagegauge_representation_v2.py` (repository root): CPU checks for
  baseline parity, mixed-cache identity, fixed gains, and unchanged exact regions.

Next required evidence before production adoption: a fresh development window
with the same fixed rule, full-model cache-recurrence/PPL and generation quality,
then matched end-to-end timing including all added encoding/scaling work. Fuse
the channel multiply only if its measured cost warrants it; do not assume the
existing 1.132x speedup survives. Independent held-out quality comes after freezing.
