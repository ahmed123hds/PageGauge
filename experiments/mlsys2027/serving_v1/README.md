# Serving integration: CPU components implemented, GPU/backend still pending

`pagegauge_vllm/sidecars.py` now allocates the actual reserved exact K/V and
center tensors. CPU tests verify contiguous FP16 storage, exact byte counts,
and budget rejection before allocation. The combined command below now passes
31 tests (0.110 s test execution). No engine allocation hook or GPU allocation
has been validated. The cache-only budget must already exclude weights,
workspace, graphs and allocator overhead; this helper does not profile them.

## Scheduler bridge update (2026-09-11)

`pagegauge_vllm/scheduler_bridge.py` now accepts the pinned vLLM
`SchedulerOutput` API and connects it to request leases, pending prefill,
full decode block tables, and explicit finish/abort events. Entire scheduler
updates are validated on copied CPU bookkeeping before publication; rejected
updates do not partially admit requests, release leases, or advance tokens.
Mixed prefill/decode, actual row reordering, temporary unscheduling, and
finish/resubmit with a new generation are covered. Generic block copy,
preemption/resume, prefix reuse, chunked prefill, and zeroing live pages reject.

The engine must invoke this bridge before generic block zero/copy, pass its
actual row order, run plane-aware zero/packing/decode, and record completion
events covering every participating stream. Only one step may be in flight;
async scheduling is not supported by this initial bridge. `complete` waits
for all prefill layers and the completion event; `abort` only reclaims after
a last-use event and requires the engine to abort/free the corresponding
requests too. It does not convert failed computations into successful results.

Eight new tests use actual pinned scheduler dataclasses and fake completion
events. Together with existing lifecycle, decode, partition/accounting, and
packed cache tests: **29 tests passed**, 0.092 s of test execution, CPU only.
This verifies bookkeeping, not actual engine hooks, CUDA event coverage,
packing/scatter kernels, generated outputs, or serving performance. The bridge
copies CPU state and has not been optimized or timed as a serving hot path.

From the repository root in WSL, reproduce with:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
  /home/anonymous/pagegauge_baselines/vllm-7a100bb617471801ee1d5525bfbb8fb238a345ea/.venv/bin/python \
  -m unittest \
  experiments.mlsys2027.serving_v1.test_pinned_scheduler_bridge \
  tests.test_serving_prefill_lifecycle tests.test_serving_decode_metadata \
  tests.test_serving_cpu_contract \
  experiments.mlsys2027.serving_v1.test_pinned_cache_spec
```

Verified source SHA256:

- Bridge: `E39A275C2DEF77FF89864AEE89D19A1EEF1F684A448C40B4632EE482AA79DDF9`.
- New tests: `02F065761F96737679A24BB746281CBA8A8E2586E8F709C19E4C4D0FEDA5F3B2`.

Next: connect raw layer allocations and exact/center sidecar reservation to the
engine; implement native prefill packing and heterogeneous GPU scatter/decode
before installing the model-runner hook. Do not register a placeholder backend.

## Prior component evidence

Source preparation (2026-09-10): pinned archive extracted separately to
`/home/anonymous/pagegauge_baselines/vllm-7a100bb617471801ee1d5525bfbb8fb238a345ea`.
Archive SHA256 is
`71ea8673e8597d2795775939a9798881ffcc7856b2a068b1d183d3de87ac6d7f`.
The exact-commit wheel is installed in a separate source-local `.venv`;
`vllm_environment_20260910.txt` records all196installed distributions.
`uv pip check` passed. CPU-only import session28751 exited0: vLLM, Torch,
AttentionBackend, AttentionImpl and register_backend imported successfully;
`torch.cuda.is_initialized()` stayed false before and after backend imports.
This is not GPU compatibility or serving validation. Local `requirements/cuda.txt`
pins Torch2.13.0 and FlashInfer0.6.17, so the live evaluation environment must
not be reused or upgraded. The source's AGENTS.md has been read.
Confirmed source entry points: backend.py `customize_spec` at136 and
`AttentionImpl.forward` at896; registry.py `register_backend` at243;
gpu_model_runner.py `_update_states` at1191 handles explicit
`scheduler_output.finished_req_ids`. These are integration locations, not
evidence that request lifecycle handling or the custom backend is implemented.

This directory is not yet a runnable vLLM backend or a serving result. Do not
register an empty backend, replace vLLM with a serial loop, or call the native
request-cost pilots a continuous-batching evaluation.

The retained target is vLLM commit
`7a100bb617471801ee1d5525bfbb8fb238a345ea`. GitHub's commit API was checked on
2026-09-09: commit date2026-08-30T21:59:36Z. The pinned source exposes
`AttentionBackendEnum.CUSTOM`/`register_backend`, `AttentionBackend.customize_spec`,
and the `FullAttentionSpec` hierarchy. These observations are not an installed
environment or a GPU compatibility attestation.

Pinned vLLM allocation preparation:

- `cache_spec.py` constructs a packed full-attention specification using the
  pinned interface's `tokens_per_state=16` and byte-sized state content. A
  CPU-only check in the isolated vLLM environment verified 32800 bytes per
  8-head page (INT8 K/V plus FP16 scales), versus 65536 bytes for FP16, without
  changing the input spec or initializing CUDA. This is not a registered
  backend: raw-view allocation, kernel strides and sidecar reservation still
  need wiring and validation.
  Production `prefill.cuh` indexes scales as `page_id * num_kv_heads + head`:
  inline per-page code/scale views would NOT satisfy this dense metadata ABI.
  The specification currently establishes capacity only. Allocate compatible
  dense scale planes within the reserved storage, or explicitly develop and
  validate a stride-aware serving kernel before use; no such kernel change
  has been made to the frozen evaluation.
  `test_pinned_cache_spec.py` adds three passing CPU-only tests in the pinned
  environment: accounting for 1/4/8 KV heads, rejection of unsupported or already
  quantized/customized specs, and no CUDA initialization. These are separate
  from the 16 dependency-free serving tests.
- `cache_views.py` now creates zero-copy whole-pool K/V and dense scale planes
  from an exact-size raw layer allocation. Five pinned-environment tests pass
  in total, including storage aliasing, disjoint writes, scale/code strides and
  malformed storage rejection. Generic engine block zero/copy/transfer uses
  different offsets and must be disabled or adapted before wiring these views.
  No GPU view/kernel invocation or engine allocator integration is validated yet.
  Compact engine-layer handoff preserves storage offsets and rejects interleaved
  layouts without copying. `zero_physical_pages` clears each code/scale plane
  rather than generic raw page slices; invalid page IDs are rejected before
  writes. Seven pinned-environment tests now pass. Actual engine hooks, ownership
  checks and device ordering are still required; zeroing allocates an index
  tensor and is not a capture-safe optimized implementation.

Implemented without CUDA/vLLM imports:

- `prefill_lifecycle.py`: reserves pending-prefill leases separately from
  decode readiness, requires complete layer coverage and a queryable completion
  event before decoding, and delays finish/abort reclamation until a last-use
  event completes. Six CPU tests cover pending packing, stale generations,
  same-ID reuse, unscheduled rows and unsupported admission. Fake events test
  bookkeeping only: the engine must actually record events covering all streams;
  this module cannot validate that an event was recorded or covers all uses.
  Decode preparation/commit now delegates to the existing metadata reference,
  gates advancement on a caller-supplied completion event and rejects retiring
  batches before any token advancement. Engine wiring, event-to-batch association
  and real CUDA error handling remain unimplemented.

- `partition.py`: page-level prefix/static-prefill-suffix/tail partition and
  exact-slot mapping, including deduplication when static suffix overlaps tail.
  Non-aligned prefills define the static suffix over completed prefill pages;
  the partial last page stays exact. Short unsupported prompts require an
  explicitly reported FP16 fallback, not silently smaller exact regions.
- `request_slots.py`: immutable request leases and snapshots; explicit admit,
  append and finish events; stable center/exact-slot identity across batch
  reordering and unscheduled intervals. Reused slots increment generations.
  Duplicate/shared physical blocks and unsupported remapping are rejected.
- `cache_layout.py`: all code/scales plus fully reserved exact/center/output
  sidecars, including empty admission slots. The engine must subtract sidecar
  allocation before admitting requests or assigning its code-page pool.

Four CPU tests pass. They compare partitions with functions extracted from the
unchanged production source, replay800decode positions with reordering and
real page aging, reject stale leases/cross-request page aliases, and reproduce
the independently measured B4 storage totals7288520704B(reference) and
6214778880B(A0). These tests do not validate CUDA scatter, CUDA stream safety,
engine hooks, native prefill, backend registration, graph replay or serving speed.

Important precise semantics: a T768 policy is a48-page ring. With a partially
filled last page, that recent region contains753--768tokens; at the complete
page endpoints used for final accounting, it contains768. This matches the
current production implementation. Do not silently change the ring or describe
it as guaranteeing that every one of the last768tokens is always exact.

Next integration obligations:

`decode_metadata.py` now defines a CPU reference for heterogeneous single-token
decode: per-row request lease/generation, position, physical token address,
exact-ring address and outgoing historical-page source/destination metadata.
Preparation does not mutate state. Commit revalidates the complete batch before
bookkeeping changes, rejecting stale state, duplicate rows and page conflicts.
Four additional tests pass (ten serving CPU tests in total). The aging test
explicitly demonstrates that the outgoing exact page and incoming token can
share a ring slot: quantization must finish before overwrite. Commit is NOT a
GPU fence; device generation checks and stream ordering remain unimplemented.
This reference uses Python tuples and is not a measured optimized hot path.

1. Pin a separate vLLM environment/source closure; do not alter audited HF/FI
   environments while benchmarks run.
2. Pass true request IDs and explicit completion events from the engine into
   this manager. Attention row indices or missing scheduled rows are not
   lifecycle events. Connect request-slot generation checks to device metadata.
3. Define a cache spec and raw code/scale views whose physical allocation matches
   the measured page bytes; reserve exact/center sidecars before admission.
4. Implement the necessary per-token scatter ABI (physical slot, per-request
   position and stable request lease) and one-shot native FP16 prefill followed
   by fixed-center packing. No full FP16 historical cache may remain resident
   in the measured decoder. Kernel/metadata stream ordering must be validated
   before releasing or reusing a slot; CPU leases alone do not prove it.
5. Validate heterogeneous B1/B4 decoding and own generated answers, then dynamic
   add/remove/reorder before measuring a mixed-length arrival workload.

Initial envelope remains TP1, full-context FP16 model/query/output, native
16-token pages, no prefix caching/chunked prefill/speculation/preemption/DCP.
Unsupported engine features must fail clearly rather than use an unreported
fallback. Regional-policy selection remains separate from this CPU work.

Sources:
- [Pinned backend registry](https://github.com/vllm-project/vllm/blob/7a100bb617471801ee1d5525bfbb8fb238a345ea/vllm/v1/attention/backends/registry.py)
- [Pinned attention interface](https://github.com/vllm-project/vllm/blob/7a100bb617471801ee1d5525bfbb8fb238a345ea/vllm/v1/attention/backend.py)
- [Pinned cache specification](https://github.com/vllm-project/vllm/blob/7a100bb617471801ee1d5525bfbb8fb238a345ea/vllm/v1/kv_cache_interface.py)
