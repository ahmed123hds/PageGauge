"""Zero-copy layer-pool views; not compatible with generic block-copy operations.

The raw layer allocation is split into whole-pool K, V, K-scale and V-scale
planes. Physical page IDs index each plane separately. Engine block copying,
zeroing and transfer must use these planes rather than raw page-sized slices.
"""


def dense_pool_views(raw, physical_pages, kv_heads, head_dim=128):
    import torch

    if any(type(x) is not int or x < 1 for x in (physical_pages, kv_heads, head_dim)):
        raise ValueError('Positive integer pool geometry required')
    codes = physical_pages*16*kv_heads*head_dim
    scales = physical_pages*kv_heads*2
    if (raw.dtype != torch.uint8 or raw.ndim != 1 or not raw.is_contiguous()
            or raw.numel() != 2*codes+2*scales or raw.data_ptr() % 16):
        raise ValueError('Aligned contiguous uint8 layer allocation of exact size required')
    return {
        'key_codes': raw[:codes].view(torch.int8).view(physical_pages, 16, kv_heads, head_dim),
        'value_codes': raw[codes:2*codes].view(torch.int8).view(physical_pages, 16, kv_heads, head_dim),
        'key_scales': raw[2*codes:2*codes+scales].view(torch.float16).view(physical_pages, kv_heads),
        'value_scales': raw[2*codes+scales:].view(torch.float16).view(physical_pages, kv_heads),
    }


def from_engine_layer(layer, physical_pages, kv_heads):
    """Reinterpret a validated compact [B,H,1,C] packed engine layer, no copy.

    This changes the meaning of bytes, not the generic engine tensor's strides.
    All subsequent cache access must use PageGauge views, including zeroing.
    """
    import torch

    expected = (physical_pages, kv_heads, 1, 2*16*128+4)
    if (layer.dtype != torch.uint8 or tuple(layer.shape) != expected
            or not layer.is_contiguous()):
        raise ValueError('Compact packed engine layer required; copying is forbidden')
    return dense_pool_views(layer.view(-1), physical_pages, kv_heads)


def zero_physical_pages(views, page_ids):
    """Zero all four planes for engine-owned fresh pages on the current stream.

    Caller must verify ownership and order this before use; this function does
    not reclaim leases, synchronize CUDA, or support capture-time allocation.
    """
    import torch

    names = ('key_codes', 'value_codes', 'key_scales', 'value_scales')
    tensors = tuple(views[name] for name in names)
    first = tensors[0]
    pages = tuple(page_ids)
    if (any(t.device != first.device or t.shape[0] != first.shape[0] for t in tensors)
            or any(type(p) is not int or not 0 <= p < first.shape[0] for p in pages)
            or len(set(pages)) != len(pages)):
        raise ValueError('Invalid physical pages or inconsistent pool views')
    if not pages:
        return
    indices = torch.tensor(pages, dtype=torch.int64, device=first.device)
    for tensor in tensors:
        tensor.index_fill_(0, indices, 0)
