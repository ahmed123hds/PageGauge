"""Independent PyTorch decoding of Kitty's released packed storage layout.

This is an execution oracle, never a timed fallback or alternative quantizer.
"""
import torch


def unpack_keys(packed, metadata, heads, dim=128, page=128, boosted=32):
    """[physical pages, bytes] -> [physical pages, heads, page, dim]."""
    per_channel = page//4
    per_head = (dim+boosted)*per_channel+dim
    raw = packed.reshape(-1, heads, per_head).long()
    low = raw[..., :dim*per_channel].reshape(-1, heads, dim, per_channel)
    high = raw[..., dim*per_channel:(dim+boosted)*per_channel].reshape(-1, heads, boosted, per_channel)
    index = raw[..., (dim+boosted)*per_channel:]
    if bool(((index < 0) | (index >= dim)).any()):
        raise ValueError('Invalid dense-to-sparse channel index')
    time = torch.arange(page, device=packed.device)
    shifts = (time % 4)*2
    codes = (low[..., time//4] >> shifts) & 3
    selected_high = high.gather(2, index.clamp_max(boosted-1)[..., None].expand(-1, -1, -1, per_channel))
    high_codes = (selected_high[..., time//4] >> shifts) & 3
    codes = codes | torch.where(index[..., None] < boosted, high_codes << 2, 0)
    values = codes.transpose(-1, -2).float()*metadata[..., 0].float().unsqueeze(-2)+metadata[..., 1].float().unsqueeze(-2)
    return values.half()


def unpack_values(packed, metadata, heads, dim=128, page=128):
    raw = packed.reshape(-1, heads, page, dim//4).long()
    channels = torch.arange(dim, device=packed.device)
    codes = (raw[..., channels//4] >> ((channels % 4)*2)) & 3
    return (codes.float()*metadata[..., 0].float().unsqueeze(-1)+metadata[..., 1].float().unsqueeze(-1)).half()


def reconstruct(cache):
    """Chronological K/V, including differently sized native residual regions."""
    keys, values = [], []
    for request in range(cache.MAX_BS):
        k_ids = cache.PageTable_K[request, :cache.PageCount_K].long()
        v_ids = cache.PageTable_V[request, :cache.PageCount_V].long()
        k = unpack_keys(cache.KeyCache[k_ids], cache.KeyCache_metadata[k_ids], cache.H_KV,
            cache.D, cache.PAGE_SIZE, cache.D_BOOSTED).permute(1, 0, 2, 3).flatten(1, 2)
        v = unpack_values(cache.ValueCache[v_ids], cache.ValueCache_metadata[v_ids], cache.H_KV,
            cache.D, cache.PAGE_SIZE).permute(1, 0, 2, 3).flatten(1, 2)
        local_ids = (torch.arange(cache.Local_Count_V, device=v.device)+cache.Write_Offset_Local_V) % cache.PAGE_SIZE
        keys.append(torch.cat((cache.Sink_Buffer_K[request, :, :cache.Sink_Count], k,
            cache.Q_Buffer_K[request, :, :cache.Q_Buffer_Count_K]), dim=1))
        values.append(torch.cat((cache.Sink_Buffer_V[request, :, :cache.Sink_Count], v,
            cache.Q_Buffer_V[request, :, :cache.Q_Buffer_Count_V],
            cache.Local_Buffer_V[request].index_select(1, local_ids)), dim=1))
    return torch.stack(keys), torch.stack(values)


def attention(query, cache):
    key, value = reconstruct(cache)
    repeats = query.shape[1]//key.shape[1]
    key = key.repeat_interleave(repeats, dim=1)
    value = value.repeat_interleave(repeats, dim=1)
    # Native QK stores FP16 scores; native softmax is FP32 then casts to FP16.
    scores = (query.float() @ key.float().transpose(-1, -2) / query.shape[-1]**.5).half()
    probabilities = scores.float().softmax(-1).half()
    output = (probabilities.float() @ value.float()).half().transpose(1, 2).contiguous()
    return output
