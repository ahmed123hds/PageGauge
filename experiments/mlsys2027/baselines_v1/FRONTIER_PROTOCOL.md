# E2 development frontier, CPU-backed initialization

This is a common eager Mistral-engine comparison, not a replacement for the
optimized PageGauge/FlashInfer benchmark or each baseline's best serving engine.
The initial implementation covers FI FP16, PG INT8, KIVI INT2/INT4 and BitDecoding
INT4. Native NSN/Kitty performance requires their separate verified model stacks.

## Frozen pilot contract

- RTX5090, pinned FP16 Mistral-7B, full-context attention, no weight conditioning.
- Distinct WikiText TRAIN requests, offset 472000, stride 23600. All methods use
  the same token matrix for a given shape. No final TEST documents are accessed.
- PG S4/A128/T768, page16, historical split128, exact split32 from completed
  development launch selection. No quantizer changes. Native baseline policies
  remain those already tested for quality.
- Same **28 GiB PyTorch allocator budget** in each worker. This is explicitly
  not a physical-device or external-CUDA allocation cap. Record allocated and
  reserved peaks plus device-used memory at boundaries and process snapshots.
- Serial coherent FP16 prefill followed by native packing to CPU. Discard all
  GPU prefill/reference caches before restoring the batched native serving state.
  Count host snapshot memory. Prefill/pack/upload are outside decode timing and
  reported separately; no prefill-inclusive capacity/serving claim.
- Decode includes all model layers and LM head, append/finalization, planning,
  native attention/merging. No CUDA graphs or explicit cache eviction. One full
  recurrence warmup and three repeats, synchronized wall and CUDA-loop timing.
- First validate B4/C4096/D785: exact initial-state bit checks, serial-HF versus
  batched-output descriptive metrics, all layer lengths/recurrence. These timings
  include diagnostic transfers and must not become performance numbers.
- Then fixed B1/B4/C20480/D1536 pilots. Examine repeat spread and process
  exclusivity before scheduling fresh-process multi-block confidence intervals.
- Capacity screening proceeds B8, then B16/B32 where feasible, followed by
  integer bracketing of the boundary. A valid OOM is a resource result; an adapter
  or kernel failure is not a capacity result. Full D1536 recurrence is required
  for each claimed feasible batch. Report a lower bound if the search ceiling is
  reached, not an invented maximum. Reconfirm boundary and next batch fresh.

An allocator-limited common-engine throughput frontier will be labeled with this
scope, alongside (not substituted for) optimized matched-system performance.
Quality-memory rows remain separate until actual matching latency evidence exists.
