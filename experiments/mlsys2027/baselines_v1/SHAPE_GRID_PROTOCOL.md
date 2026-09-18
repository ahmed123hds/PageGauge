# Next fixed shape grid (development; not yet launched or final-frozen)

Run after regional cost assessment and the synthetic own-generation smoke.
This addresses context/batch breadth, not the original optimized speed claim.

- Model: the retained full-context Mistral checkpoint, native tokenizer and
  existing common eager model body. No sliding-window alteration.
- Contexts: 8192, 20480, 30720; decode: 1536 real teacher-forced positions.
  The last value deliberately replaces 32768: 32768+1536 exceeds this model's
  declared 32768-position context. 30720+1536=32256 remains within it.
- Batches: 1, 4, 8 independent requests. Never replicate a request or shard a
  nominal batch to make an infeasible point appear feasible.
- Five mechanisms: FlashInfer FP16, PageGauge, native KIVI INT4, KIVI INT2,
  and BitDecoding INT4 in the same audited eager model body.
- Pin the PageGauge regional policy before launching the grid. Keep a separate
  identifier for the historical S4/A128/T768 reference and any selected A0
  development candidate; never silently replace an old result.
- One fixed TRAIN starting offset; request stride 35000 tokens so all longest
  request windows, including the next-token label, are disjoint. Freeze actual
  token IDs/hashes and use the identical tensor for all five backends per shape.
- Budget: 28 GiB PyTorch allocator, expandable segments for every backend;
  separately report non-PyTorch/device memory where observable. Analytical
  weights+minimum-code lower bounds may rule out a cell but are not measured OOMs.
- FI/PG use explicit non-graph planning in the common eager body. No graph
  replays are implied. KIVI/BitDecoding retain their audited native kernels.
- Before timing a new context, validate B1 full recurrence against coherent HF
  and selected reconstructed-cache execution probes for FI/PG. Predictive
  PPL/top1 are descriptive; computational correctness checks remain required.
- Capacity pilot: one fresh process per cell, one full warmup and three full
  1536-step repeats, CPU-backed restore and bitwise initial-state checks.
  Report medians/ranges, persistent served bytes and peak allocated memory.
  Preparation/upload are separate, not zero-cost prefill. No hierarchical CI
  from three repeats in one process.
- Retain genuine CUDA OOMs and unsupported cells. Stop on other implementation
  or monitoring failures, diagnose and version the repair; do not label them OOM.
- Execute one GPU worker at a time. Use the validated WSL context/host monitor
  while Linux NVML process enumeration is unavailable. Pause only our verified
  downloader during clean timing, with an unconditional resume trap.
- No public task TEST or final PG19 TEST is involved. No cross-GPU or online
  serving claim follows from this grid alone.

Implementation note: frontier_run's CLI currently restricts contexts to
4096/20480 and fixes stride 23600. Do not merely pass a longer context through
that interface. A new source-frozen launcher must explicitly construct these
fixtures and invoke the audited worker. Preserve historical launcher files.
