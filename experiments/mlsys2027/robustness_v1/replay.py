"""Replay fixed HF fixtures through the unchanged production PageGauge attention."""
import sys
from pathlib import Path


def run_one(path, plan, root):
    import numpy as np
    import torch
    import flashinfer
    sys.path.insert(0, str(root / "scripts"))
    import benchmark_page_gauge_transformer as pg
    with np.load(path, allow_pickle=False) as data:
        q = torch.from_numpy(data["query"].copy()).cuda().unsqueeze(0)
        original = {name: torch.from_numpy(data[name].copy()).cuda() for name in ("key", "value")}
        centers = {name: torch.from_numpy(data[name + "_center"].copy()).cuda() for name in ("key", "value")}
    n = original["key"].shape[0]
    page, dim, heads = 16, 128, 8
    pages = (n + page - 1) // page
    initial = plan["initial_context_tokens"] // page
    policy = plan["policy"]
    tail = policy["exact_tail_tokens"] // page
    storage = policy["exact_prefix_pages"] + policy["exact_static_suffix_pages"] + tail
    last = (n - 1) % page + 1
    partition = pg.page_gauge_logical_partition(pages, tail, policy["exact_prefix_pages"], last,
                                               policy["exact_static_suffix_pages"], initial)
    ring = pg.request_major_exact_page_table(1, pages, tail, policy["exact_prefix_pages"],
                                             policy["exact_static_suffix_pages"], initial)
    logical = torch.tensor(partition["exact_logical_pages"], dtype=torch.long, device="cuda")
    physical = ring.index_select(1, logical).flatten().long()
    if physical.unique().numel() != physical.numel():
        raise RuntimeError("Exact physical pages alias")
    codes, scales, exact, reconstructed = {}, {}, {}, {}
    for name, values in original.items():
        padded = torch.zeros((pages * page, heads, dim), dtype=torch.float16, device="cuda")
        padded[:n] = values
        residual = (padded.float() - centers[name].float()).reshape(pages, page, heads, dim)
        scale = (residual.abs().amax(dim=(1, 3)) / 127.0).clamp_min(2.0**-20).half()
        quotient = residual / scale.float()[:, None, :, None]
        code = (quotient.sign() * (quotient.abs() + 0.5).floor()).clamp(-127, 127).to(torch.int8)
        codes[name], scales[name] = code.contiguous(), scale.contiguous()
        exact[name] = torch.zeros((storage, page, heads, dim), dtype=torch.float16, device="cuda")
        exact[name].index_copy_(0, physical, residual.half().index_select(0, logical))
        reconstructed[name] = (code.float() * scale.float()[:, None, :, None]).half()
        reconstructed[name][logical] = residual.half()[logical]
    cache = pg.GaugeCache(exact["key"][None], exact["value"][None], codes["key"][None], codes["value"][None],
                          scales["key"][None], scales["value"][None], centers["key"][None, None], centers["value"][None, None],
                          exact_tail_pages=tail, exact_sink_pages=policy["exact_prefix_pages"],
                          exact_static_suffix_pages=policy["exact_static_suffix_pages"], initial_context_pages=initial)
    decoder = object.__new__(pg.TransformerDecoder)
    decoder.backend, decoder.tail_attention, decoder.center_restore = "page_gauge", "flashinfer_merge", "attention_add"
    decoder.cache, decoder.rotated_query = cache, q
    decoder.attention_output = torch.empty((1, 1, 32, 128), dtype=torch.float16, device="cuda")
    decoder.exact_output = torch.empty_like(decoder.attention_output)
    decoder.old_lse = torch.empty((1, 1, 32), dtype=torch.float32, device="cuda")
    decoder.exact_lse = torch.empty_like(decoder.old_lse)
    decoder.old_wrapper = pg.GraphDecodeWrapper(flashinfer, pages, torch.int8, True)
    decoder.exact_wrapper = pg.GraphDecodeWrapper(flashinfer, storage, torch.float16, False)
    reference = pg.GraphDecodeWrapper(flashinfer, pages, torch.float16, False)
    table = pg.request_major_page_table(1, pages)
    old_table, exact_table = torch.empty_like(table), torch.empty((1, storage), dtype=torch.int32, device="cuda")
    old = pg.write_page_gauge_old_page_table(old_table, table, partition["old_logical_pages"])
    exact_indices = pg.write_exact_prefix_tail_page_table(exact_table, ring, partition["tail_logical_begin"], pages,
                policy["exact_prefix_pages"], policy["exact_static_suffix_pages"], initial)
    decoder.old_wrapper.plan(old, partition["old_token_count"], page, 128)
    decoder.exact_wrapper.plan(exact_indices, partition["exact_token_count"], last, 128)
    reference.plan(table, n, last, 256)
    # Untimed snapshot attention, not graph/recurrence/end-to-end validation.
    pg.TransformerDecoder.eager_attention(decoder, 0)
    factorized = decoder.attention_output[0, 0].cpu().numpy()
    explicit = reference.wrapper.run(q, (reconstructed["key"], reconstructed["value"]))
    explicit.add_(cache.output_center[0])
    torch.cuda.synchronize()
    return factorized, explicit[0].cpu().numpy()
