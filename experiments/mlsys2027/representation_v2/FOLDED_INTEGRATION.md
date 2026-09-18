# Removing the runtime channel correction

The separate historical-output multiply costs 0.809 microseconds per layer on the
B4 RTX 5090 postprocessing microbenchmark (with equal reset copies in both arms).
The estimated 32-layer addition is 0.025888 ms, excluding page-finalization costs.
Evidence: `results/mlsys2027_representation_v2/cost_20260908T054055Z_82fec755`.
This is not a complete-decoder measurement.

## Alternative implemented

Let G be a positive diagonal matrix per KV head, shared across requests and tokens.
Using row-vector activations and PyTorch's weight orientation, replace:

- V projection weight rows: `Wv' = G^-1 Wv` (and V bias by `G^-1 bv` if present).
- Output projection weight columns: `Wo' = Wo G_HQ`, where G_HQ repeats the KV
  head's channel gains for its GQA query heads in the decoder's existing order.
- Prefill value cache: `V' = V G^-1`.

Then `P V' Wo'^T = P V Wo^T` in real arithmetic. Keys, queries, softmax weights,
attention segmentation, quantized code width, tensor shapes and decode operation
counts do not change. PageGauge's original scalar quantization and center
factorization operate on the reparameterized values. Generated values are already
scaled by the folded V projection, so runtime append/finalization needs no new
division, gain load, or launch. No new INT8 or merge kernel is needed.

Exact regions retain FP16 and the same token allocation, but their values now
use scaled coordinates. FP16 range and rounding can differ, including in folded
weights. This is not identical to keeping original-coordinate exact regions as
the earlier GPU prototype did. Measure full-model quality; do not transfer its
head-cosine result to this new realization without testing.

## Fitting and timing boundaries

To share one set of projection weights across B4, fit each layer/head's gain from
request zero's initial HF prefill, not separately per request. Reuse it for the
other requests. Rule: residual RMS / geometric mean, nearest power of two,
exponent clamp [-8,8]. No sweep and no future token/query fitting.

Only the destination prefill values are scaled. The HF oracle and its tensors
remain original. Fold packed projection weights only after all HF oracle calls
have completed, during production decoder construction and before graph capture.
Track gain hash, metadata bytes, prefill transform time, fold time and FP16 weight
round-trip changes. These setup costs are real and excluded from decode timing;
no full-serving/prefill-cost claim is made.

## Opt-in integration

`PAGEGAUGE_VALUE_CONDITIONING=folded_prefill_rms` enables the option for the main
decoder with the exclusive-cache builder. Default `none` retains the previous
path. Unsupported cache builders fail explicitly if the option is enabled. Do
not set the option on an unrelated experiment. The single-seed runner configures
it only in its own process environment; it does not persistently alter the shell.

Main integration files:

- `scripts/page_gauge_value_conditioning.py`
- `scripts/benchmark_page_gauge_transformer.py`
- `diagnostics/benchmark_backend_exclusive.py`
- `diagnostics/benchmark_sustained_dynamic_graphs.py`

Original versions of the three existing files are preserved under
`experiments/mlsys2027/representation_v2/pre_fold_sources`. Completed manifests
and results were not rewritten. Old manifests naturally describe the old source
versions, not this opted-in development revision.
After the single-seed pair completed, the CPU entry launcher's source list was
updated to freeze the new module in future experiments. Its pre-update version
is also preserved there. A fresh Step 01 attestation will be needed before using
the old 01/02 launcher on the updated source closure; old results are not reissued.

## One-seed validation

Runner: `experiments/mlsys2027/representation_v2/run.sh single_seed`.
One seed (20260861), same TRAIN B4/C20480/D1536 fixture as the earlier speed run;
two adjacent fresh processes: original PageGauge then folded PageGauge. Both use
three repeats in each cache mode, the same kernels and full decoder timing scope.
Execution, recurrence, source and sampled process-exclusivity checks remain active.
Cosine is descriptive (`--min-logits-cosine -1`), while top-1 >=0.99 is retained.

The practical cost criterion is <=0.5% mean cache-neutral latency increase,
with cache-hot results also reported. This is an operational tolerance, not proof
of zero added cost, not a statistically established equivalence margin, and not a
publication confirmation. One adjacent pair cannot distinguish small drift/order
effects from the transformation. No automatic default promotion occurs.

Completed evidence directory:
`results/mlsys2027_representation_v2/single_seed_20260908T054708Z_398a7e36`.

## Measured outcome

| Metric | Unchanged PG | Folded V/O PG |
|---|---:|---:|
| Cache-neutral wall ms/step | 15.21195583 | 15.21902529 |
| Cache-hot wall ms/step | 15.22117860 | 15.21995925 |
| Minimum model-logit cosine vs HF | 0.98179740 | 0.98691243 |
| Model top-1 agreement vs HF | 0.99967450 | 0.99967450 |
| Served-cache bytes | 7,288,520,704 | 7,288,586,240 |

Neutral latency increased 0.04647%; hot latency decreased 0.00801%. Both satisfy
the stated 0.5% practical tolerance, but one adjacent pair cannot establish zero
cost or statistical equivalence. Timed-operation counters are identical. Both
workers passed execution, graph/cache recurrence, top-1 and sampled-exclusivity
checks. Cosine remains descriptive, not an acceptance gate.

The folded option added 0.995614 seconds for prefill conditioning and 0.058336
seconds for weight folding, outside decode timing. Gain metadata adds 65,536
bytes (64 KiB). FP16 scaling changed 41,267 projection-weight elements under a
round-trip check, so the real-arithmetic reparameterization is NOT bitwise exact
in the stored weights. The measured one-seed fidelity includes those effects.
Checkpoint files on disk remain untouched; the reparameterization is in-memory.

Integration remains opt-in, not a default-policy or held-out quality promotion.
No claim that the prior 0.997 head-output cosine carries over to final logits,
and no new FlashInfer-relative speed claim from this PG-vs-PG pair.
