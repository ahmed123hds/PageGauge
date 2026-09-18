# Kitty SM120 integration: disclosed address-width correction

Native source: `dfd2c07b407d6b407179359207c612ab631f3ed1`.
Native Transformers fork: `37f8b0b53512e6aae0cfd15746c133c101783178` (4.53.2).
Environment uses Torch 2.12.1+cu130 and Triton 3.7.1 for SM120. Main environment
is unchanged. Kitty is installed editable as its README specifies; its regular
wheel omitted the native namespace package in this environment.

## Failure and diagnosis

Unmodified source failed the same-reconstructed-cache execution check:
`kitty_smoke_20260908T145137Z_ce4ac637`, first B1 step, relative L2 0.02229215.
The tolerance was 0.005 and remains unchanged. Original kernel files and runner
are preserved beside the failure. This is not an original-FP16 quality result.

`kitty_diagnostic_20260908T145542Z_a45feb01` isolates the difference to packed
QK. Sink/residual QK and SV using native probabilities agree with the reference.
One-hot queries in `kitty_diagnostic_20260908T145711Z_00e67670` recover the
native kernel's effective K: only the boosted channels differ. Nonboosted,
sink and residual K agree exactly. Compiled qk TTIR shows the high-bit address
product as `arith.muli ... tensor<128x1xi8>` before widening to i64.

The second diagnostic's auxiliary `qk` field is mislabeled (it contains the
probability comparison), and its `native_scores` tensor contains the last basis
probe. Use the first diagnostic for score comparisons and the explicitly named
`basis_native_k`/basis-region fields in the second. The diagnostic script is
corrected for future runs; the original artifacts are retained, not rewritten.

The uint8 dense-to-sparse index is multiplied by the 32-byte channel stride.
Indices 0..31 require offsets 0..992; uint8 arithmetic retains only eight
distinct offsets. Both pack and read paths therefore alias boosted channels,
including colliding writes. This diagnosis applies to the inspected released
source under the tested compiler, not an untested claim about all platforms.

## Correction and validation

Only two address calculations are changed: widen `dense_sparse_idx` in the
pack kernel and `boost_idx` in the QK kernel to `tl.int32` immediately after
loading. Stored uint8 indices, bit widths, channel selection, scale/minimum
metadata, residual policy and attention equations are unchanged. This is a
corrected native port, and must be labeled as such in later comparisons.

Corrected run: `kitty_smoke_20260908T150002Z_47d92e07`.
B1/B4, Hq32/Hkv8/D128, prefill511, decode257; 11 reference checkpoints per
batch, native append-attend-finalize, new packed K and V pages consumed.
Worst relative L2: 0.0000397537 (B1), 0.0000338776 (B4). Worst absolute error:
0.0000076294. Final logical length768. The independent packed-cache oracle is
unchanged. Three CPU bit-layout/address-width regression checks pass.

The eight-second worker finishes before the periodic ten-second observer sees
its PID. Its four explicit GPU-boundary samples see only its own PID; all
periodic samples have no unexpected PID or observation error. A separate
`assessment.json` combines these retained observations without rerunning GPU
work. This is sampled, not continuous, exclusivity evidence.

No pretrained Kitty quality, latency, or throughput result is established by
these checks. Those must use this disclosed corrected source and matched inputs.

Portable correction: `experiments/mlsys2027/baselines_v1/kitty_int32_addressing.patch`.
Apply it only to the pinned source above; a reverse dry-run check verifies the
current source already contains it. This patch does not change method defaults.
