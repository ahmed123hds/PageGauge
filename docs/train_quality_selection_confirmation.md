# Frozen TRAIN selection/confirmation quality protocol

This protocol evaluates only the already-frozen PageGauge policy
`S3/T768/D1024` (`B4`, context 20,480). It does not alter the quantized
attention equations, implementation, quantizer, merge, or strict gates.

## Selection is not confirmation

The four WikiText-2 TRAIN windows beginning at token offsets 100,000,
121,984, 143,968, and 165,952 were used for diagnosis and policy selection.
They remain labeled **selection**. The selected artifact has 4,096 checked
rows, minimum logit cosine 0.9952273368835449, five top-1 mismatches
(agreement 0.998779296875), and zero rows below 0.995. Its worst row is request
3, step 907, model input position 21,387. The small 0.000227 margin and the
non-monotone response relative to T512 are why confirmation is strictly
one-shot.

The untouched confirmation cohort is frozen before execution as

```text
start_n = 300000 + n * 125000, n = 0,...,19
```

The 20 windows form five B4 shards. Each window contains 20,480 prefix tokens
and 1,024 labeled decode rows (21,504 corpus tokens total). The last interval
ends at 2,696,504, below the frozen 2,763,961-token TRAIN corpus length. All 24
selection-plus-confirmation intervals are unique and non-overlapping. The
corpus could fit 128 adjacent non-overlapping windows; 20 widely distributed
confirmation windows provide 20 whole-window bootstrap clusters in five
feasible worker runs.

## One-shot rule and gates

Initialization seals the policy, thresholds, selection artifact SHA-256,
source hashes, content coordinates, five commands, and confirmation windows in
`TRAIN_QUALITY_PREREGISTRATION.json`. All five shards run even if an earlier
one fails quality, which prevents optional stopping. A completed quality
failure is terminal: it is preserved, never rerun, and never causes a fallback
or a policy/threshold change.

Every B4 shard must contain exactly 4,096 rows and independently pass:

- minimum full-vocabulary logit cosine >= 0.995, with zero rows below the gate;
- top-1 agreement >= 0.99;
- eager/graph and restored-repeat equivalence;
- immutable adjacent-page and exact-prefix canaries;
- zero graph-bank misses and zero eager graph fallback;
- exact page-table partition and token coverage;
- consumption of exactly the 16 expected runtime-finalized INT8 old pages
  (logical pages 1,280 through 1,295);
- scheduler capacity without automatic split clamping.

The combined 20-window confirmation cohort must also pass cosine >= 0.995 and
top-1 >= 0.99. Results include the exact worst-row identity, cosine quantiles,
relative-L2 and maximum-absolute-error summaries, per-window summaries, and
fixed 50,000-draw percentile intervals that resample whole windows. Correlated
token rows inside a window are never treated as independent bootstrap units.

The sustained worker does not serialize complete probability distributions.
Therefore this protocol does not invent NLL, perplexity, KL, or JS from its
small top-vocabulary diagnostic subset. Those metrics are explicitly deferred
to the untouched held-out full-distribution protocol.

## Commands

Run these from the repository root in the frozen CUDA/Python environment. The
first command must complete before any confirmation worker is launched.

```bash
python diagnostics/run_train_quality_protocol.py initialize \
  --selection-artifact results/exact_prefix_s3_v1/train_structural/matched_d1024_t768.json \
  --output-dir results/exact_prefix_s3_v1/train_confirmation_s3_t768_d1024

python diagnostics/run_train_quality_protocol.py run \
  --output-dir results/exact_prefix_s3_v1/train_confirmation_s3_t768_d1024

python diagnostics/aggregate_train_quality_protocol.py \
  --manifest results/exact_prefix_s3_v1/train_confirmation_s3_t768_d1024/TRAIN_QUALITY_PREREGISTRATION.json \
  --status results/exact_prefix_s3_v1/train_confirmation_s3_t768_d1024/TRAIN_QUALITY_RUN_STATUS.json \
  --output results/exact_prefix_s3_v1/train_confirmation_s3_t768_d1024/TRAIN_QUALITY_AGGREGATE.json
```

Resume verifies the recorded result hash and revalidates every strict gate.
Untracked existing outputs, changed source files, changed selection bytes, and
changed status or manifest bytes fail closed rather than being adopted.
