"""Packed page specification for the pinned vLLM interface, not a backend.

One stored state accounts for K/V INT8 codes and two FP16 scales per head
per 16-token page. This specifies capacity, not compatible kernel strides.
The existing kernel requires dense page/head scale arrays: do not pass scales
interleaved with code bytes to it. Allocation/view wiring remains outstanding.
Request-specific exact/center sidecars are reserved separately.
"""
from dataclasses import replace


def packed_page_spec(spec):
    import torch
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVQuantMode

    if type(spec) is not FullAttentionSpec:
        raise ValueError('Only plain full attention is supported initially')
    if (spec.block_size != 16 or spec.dtype != torch.float16
            or spec.kv_quant_mode != KVQuantMode.NONE
            or spec.head_size != 128 or spec.head_size_v != 128
            or spec.sliding_window is not None
            or spec.attention_chunk_size is not None or spec.non_causal
            or spec.num_head_slots is not None
            or spec.state_content_bytes is not None
            or spec.tokens_per_state != 1 or spec.page_size_padded is not None):
        raise ValueError('Unsupported or already customized attention specification')
    return replace(spec, dtype=torch.uint8, tokens_per_state=16,
                   state_content_bytes=2*16*spec.head_size+4)
