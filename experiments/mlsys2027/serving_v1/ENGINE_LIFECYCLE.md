# Pinned engine lifecycle integration contract

Inspected against vLLM commit
`7a100bb617471801ee1d5525bfbb8fb238a345ea`, 2026-09-10.
This is an implementation contract, not a completed engine adapter.

2026-09-11: the CPU scheduler-to-lifecycle bridge is implemented in
`pagegauge_vllm/scheduler_bridge.py` and tested with the actual pinned scheduler
dataclasses. The lifecycle sequences below now have CPU coverage; the hook
and GPU obligations remain pending. Call `apply` before the engine's generic
zero/copy operations, using the final request-row ordering. Its `zero_pages`
must be handled with plane-aware zeroing, not raw generic page slices.
Supply last-use events keyed by the old Lease (not a reused request string).
Only publish `complete` after every prefill layer and decode operation has
finished successfully. Engine-level error/abort propagation and event recording
remain mandatory. This initial bridge requires synchronous step completion;
do not enable asynchronous scheduling around it.

`GPUModelRunner._update_states` processes `finished_req_ids` before new
requests. A finished ID can be resubmitted in the same scheduler output;
the replacement must get a new lease generation, even if its string ID and
physical blocks match. Removing an unscheduled row from `input_batch` does
not finish that request. Retain its center, exact slots and ownership.

`NewRequestData.block_ids` is a tuple of block lists (cache groups).
For the initial single-group envelope, explicitly reject multiple groups;
do not flatten group indices into physical page IDs.
Admission receives a prompt whose KV has not yet been computed:
`num_computed_tokens` must be zero with prefix reuse/chunked prefill disabled.
The existing `RequestSlots.admit` represents an already packed prompt, so
it cannot be called as if scheduler admission had completed native prefill.
Introduce a pending-prefill state and publish the ready lease only after
all layers have packed their prompt and device work is ordered.

`CachedRequestData.new_block_ids` contains increments for ordinary requests,
but replacement tables for `resumed_req_ids`. Reject resumed requests in
the initial no-preemption envelope. Preserve complete tables independently
of the current attention row ordering. Reject block-copy events while
prefix sharing/remapping is unsupported.

Before hooking the engine, test these event sequences:

1. Admit, complete prefill, decode; compare position and complete block table.
2. Temporarily unschedule, reorder other requests, resume without preemption;
   retain the same lease and centers.
3. Finish and resubmit the same ID in one scheduler output; old device work
   must complete before sidecar/page reuse and the generation must change.
4. Reject chunked/prefix-reused prefill, resumed/remapped requests and multiple
   cache groups before mutating request state.
5. Reject failed/partial prefill as decode-ready; release resources only after
   the corresponding device work has safely completed.

CPU bookkeeping tests cannot establish device lifetime safety. Engine hooks,
CUDA events/generation metadata, packing and heterogeneous decode validation
remain required before any serving throughput measurement.

## Allocation handoff discovered in pinned source

`vllm/v1/worker/utils.py:379` allocates one shared INT8 backing buffer, then
`create_kv_cache_views` produces per-layer `[B,H,N,C]` tensors. Layer offsets
and block strides come from `KVCacheTensor`; a layer is not automatically an
independent flat allocation. The engine adapter must require a layer-compact
layout (for example `KVCacheLayout.LBHNC`) and validate the actual layer slice
before passing a zero-copy byte view to `dense_pool_views`. Never call
`contiguous()` to hide a strided layer: that would duplicate the cache and
invalidate memory accounting. Non-layer-compact layouts need a separate design.

Even with a compact layer, PageGauge's whole-pool planes intentionally differ
from generic page-sized slices. The generic `_zero_block_ids` and
`copy_kv_cache_blocks_inplace` paths cannot operate on those slices after
reinterpretation. Handle zeroing per plane; reject block copying/transfers
within the initial no-sharing/no-preemption envelope. These hooks remain pending.
