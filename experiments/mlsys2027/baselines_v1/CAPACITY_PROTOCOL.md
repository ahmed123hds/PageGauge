# Prospective common-engine capacity grid

Development-only extension after the completed B4 CPU-backed pilot. Do not
confuse this grid with native-best serving or a proof of maximum capacity.

## Fixed comparison

- Pinned Mistral-7B-v0.3 FP16, common eager Transformers 4.36.2 body.
- Backends: FI FP16, PageGauge INT8 exact32, KIVI INT4, BitDecoding INT4,
  KIVI INT2, with the same cache policies as the completed B4 pilot.
- C20480, D1536, independent synchronized TRAIN requests, corpus offset472000,
  stride23600 and seed2026090815. No duplicate-request cache replication.
- Fixed grid B4, B8, B16. Reuse completed source-matched B4 evidence. At each
  larger shape use a fresh process, full recurrence warmup, then one full timed
  recurrence for initial feasibility. This repeat count is not a final timing CI.
- Same 28 GiB PyTorch allocator budget. Record allocator and physical-device
  boundaries separately; this is not a whole-device allocator cap.
- Serial coherent prefill, native packing, CPU snapshots and upload are outside
  decode timing and separately recorded. No retained GPU FP16 reference cache.
- Do not change native quantizers or kernels in response to a capacity result.

## Cheap exclusion and failure classification

An analytical exclusion requires the pinned model's actual FP16 parameter
storage plus a conservative lower bound on the native serving-cache storage to
exceed 28 GiB. Ignore activations, workspaces and temporary allocations in that
bound, making it conservative. Validate cache scaling from the actual native
tensor shapes; do not linearly extrapolate observed peak allocation as proof.

Otherwise run the complete trajectory. Classify a failure as CUDA OOM only from
the worker's actual CUDA-OOM exception. Unsupported kernel, host OOM, corpus
range errors and source drift are different outcomes, not capacity failures.
Retain all failures. Do not silently lower sequence length or decode horizon.

Higher grid points are not automatically excluded after a lower OOM. Either
run them, provide the conservative analytical exclusion, or label unmeasured.
Report the largest verified feasible point on this grid, never global maximum
batch. A selected feasible frontier point needs a separate fresh confirmation
and replicated timing before a final comparative throughput claim.

Attention-only graph FI/PG results form a separate dispatch ablation; do not
silently substitute them into this original common-eager five-method grid.
