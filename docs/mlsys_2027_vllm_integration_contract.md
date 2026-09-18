# PageGauge vLLM serving-integration contract

The lowest-risk MLSys serving target is an out-of-tree vLLM custom attention
backend pinned to vLLM commit
`7a100bb617471801ee1d5525bfbb8fb238a345ea`. That revision still expects
FlashInfer 0.6.17, matching the audited PageGauge source, and its native cache
block size is the required 16 tokens. SGLang is not the first target because its
current FlashInfer path plans token-level (`page_size=1`) indices and does not
guarantee a physically contiguous PageGauge tile.

## Initial support envelope

- final S4/A128/T768 representation; the attention mathematics and quantizer
  are unchanged;
- 32 query heads, 8 KV heads, head dimension 128, FP16 model/query/output;
- tensor parallelism 1 and native FlashInfer FA2;
- one-shot prefill and variable-length decode;
- no prefix caching, chunked prefill, speculative decoding, preemption,
  sliding-window attention, cross-attention, or DCP in phase 1;
- exact-region sidecar memory reserved before capacity admission.

Prefix caching is initially disabled because PageGauge centers are
request-specific and are finalized from the complete initial prompt. Ordinary
partial-prefix reuse is therefore not center-compatible.

## Package boundary

```text
pagegauge_vllm/
  pyproject.toml
  pagegauge_vllm/plugin.py
  pagegauge_vllm/config.py
  pagegauge_vllm/cache_spec.py
  pagegauge_vllm/cache_layout.py
  pagegauge_vllm/partition.py
  pagegauge_vllm/request_slots.py
  pagegauge_vllm/metadata.py
  pagegauge_vllm/flashinfer_factory.py
  pagegauge_vllm/backend.py
  pagegauge_vllm/provenance.py
  tests/cpu/
```

Use the `vllm.general_plugins` entry point, a custom `AttentionBackend`, and a
custom `FullAttentionSpec`. Stable request-slot identities must flow from
vLLM's request IDs; transient batch-row numbers must never own centers or exact
pages after row reordering.

## Required append ABI

Continuous batching cannot use the existing shared-position append call. The
serving path needs a scatter operation driven by `slot_mapping`, per-token
positions, token-to-request identities, and stable center slots. It writes
centered exact K/V for each token and finalizes INT8 codes/scales when token 15
closes a page.

## CPU-first implementation order

1. Pin vLLM, FlashInfer, and patched-header hashes.
2. Implement pure S/A/T partitioning, overlap deduplication, cache layout, and
   deterministic byte accounting without CUDA imports.
3. Replay synthetic scheduler traces covering add, decode, reorder, remove,
   page close, row reuse, and exact-slot reclamation.
4. Test plugin/spec registration and fail closed on ABI or source drift.
5. Only after the unrelated GPU workload finishes, run eager B1 correctness,
   variable-length B4, continuous batching, CUDA graphs, and SLO goodput in
   that order.

The future runner must refuse to start when the selected GPU is occupied. It
must never stop, kill, or attach to an unrelated process.
