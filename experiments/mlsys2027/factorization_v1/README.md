# E1 matched page-scale placement control

The new control retains INT8 page storage, one scalar per page/head, shared
centering, FP16 MMA, exact-region merging and the existing code conversion. It
multiplies scales into converted K/V registers before MMA instead of correcting
score/probability fragments. It does not materialize an FP16 historical cache.

This intentionally strong control retains common-center cancellation/restoration
in both arms: it isolates page-scale placement, NOT every PageGauge identity.
Rounding differs: register reconstruction rounds scaled K/V to FP16, while
factorized QK applies scale after accumulation and factorized PV scales
probabilities. Compare both with their explicit reference and original values;
do not demand bitwise equality or silently attribute quantization error to a bug.

`control.py` mechanically generates a separately hashed header under
`build/mlsys_factorization_control/`; it never changes the live production header.
Its wrapper keeps identical argument shapes and a separate JIT module identity.
No GPU execution or speed evidence exists yet. First compile/validate an isolated
synthetic test after the active E0 worker exits, then captured real KV cases and
matched timing. Do not claim this is a validated/optimized baseline until then.
# Full-model contrast

`bash experiments/mlsys2027/factorization_v1/run.sh full_model` runs eight fresh
processes in ABBA/BAAB order on two frozen TRAIN fixtures, B4/C20480/D1536.
A is centered register reconstruction; B is original PageGauge. Both use the
same compressed representation, exact-region policy, split sizes, append/merge
path and unconditioned weights. Shared-center algebra remains in both arms.
This isolates **scale placement**, not the entire affine factorization.

The adapter changes only its own process's wrapper factory. Kernel identity and
scale placement are explicitly emitted; production sources are not modified.
Each block retains eager/graph, 96-page-close recurrence, timed dispatch, cache
accounting and sampled GPU-exclusivity checks. Ratios are register/factorized,
with fixture-then-pair bootstrap intervals (only two fixture clusters).
No positive result is required; ties and regressions must be retained. This is
not a FlashInfer-relative speed or independent held-out quality claim.
