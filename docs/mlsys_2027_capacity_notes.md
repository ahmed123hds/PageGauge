# E2 capacity: interpretation before boundary runs

The fixed-B4 CPU-backed timing pilot is complete:
`frontier_pilot_b4_20260908T154840Z_e4fffbdc/analysis.json`.
All five methods completed the full recurrence, but this single-workload
pilot is not a final comparative confidence interval or maximum capacity.

## Observed native KIVI INT4 allocation

Completed arm:
`results/mlsys2027_baselines_v1/frontier_kivi_int4_20260908T155311Z_18690b36`.
For each timed recurrence, initial GPU allocation is 18,431,025,664 bytes,
final allocation 18,684,578,304 bytes, and peak allocation 22,941,229,568 bytes.
Peak reserved memory reaches the fixed 28 GiB allocator budget. The original
FP16 reference cache is absent; post-preparation GPU allocation is
15,067,193,856 bytes. Native final served KV is 3,612,868,608 bytes.

Thus served-cache bytes alone substantially understate this implementation's
peak requirement. The measured difference is not evidence that the cache was
silently expanded permanently to FP16.

Source inspection of the pinned, unmodified `KIVI/models/mistral_kivi.py`
(commit 876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6) shows:

- Lines 152--156 and 211--215 repeat quantized GQA cache/metadata for query heads.
- Lines 172--180 allocate concatenated key codes/metadata at residual closure.
- Lines 220--229 quantize one oldest residual value token and concatenate its
  codes/metadata on every step once the residual window is full.
- The model returns newly assembled legacy cache tuples. Old input tuples can
  remain alive while the forward constructs the new layer-cache collection.

These mechanisms plausibly explain transient allocation and additional traffic;
this is a source-based explanation, not a measured per-operation attribution.
Do not attribute the entire latency or peak to one line without a profile.
Do not modify the native baseline during the current frozen cohort.

## Boundary-run policy

Use completed fixed-batch peak allocation as a **starting-point estimate** for
capacity candidates, not proof of capacity or a guaranteed linear model. Actual
full D1536 recurrence is required for each claimed feasible batch. Separate:

1. weights + persistent served-cache analytical lower bounds;
2. measured allocator/physical-memory peaks;
3. actual CUDA OOM, unsupported-kernel failure, and unrelated host/system failure.

Skip a larger batch without GPU execution only when an explicit conservative
lower bound already exceeds the fixed allocator budget; label it analytically
infeasible, not measured OOM. Reconfirm a successful boundary fresh and test its
next candidate. If higher candidates remain untested and not excluded by a
valid bound, report largest **verified** feasible batch, not a global maximum.
Avoid exhaustive prefills of obviously impossible shapes; if many boundary
probes become necessary, consider reusing source-frozen CPU compressed initial
states with preparation/storage costs separately disclosed.

This remains a common eager-engine contrast. It does not replace native-best
optimized serving or a prefill-inclusive admission/continuous-batching study.
