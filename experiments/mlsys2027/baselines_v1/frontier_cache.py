"""CPU-backed initial cache snapshots for deployment-capacity experiments.

Prefill/packing are untimed preparation. Only native serving state is uploaded
for decode; the original FP16 reference cache is not retained on the GPU.
"""
import math


def cpu_state(legacy, backend, maximum):
    import torch
    if legacy[0][0].shape[0] != 1:
        raise ValueError('Prepare one independent request at a time')
    if backend.startswith('kivi_'):
        from kivi_adapter import pack_initial_cache
        bits = 2 if backend == 'kivi_int2' else 4
        packed = pack_initial_cache(legacy, bits)
        return {'kind': 'legacy', 'layers': tuple(tuple(v.cpu() if isinstance(v, torch.Tensor) else v for v in layer) for layer in packed)}
    if backend == 'bitdecode_int4':
        from bitdecode_adapter import pack_initial_cache
        packed = pack_initial_cache(legacy, 4)
        return {'kind': 'bitdecode', 'layers': [{name: v.cpu() if isinstance(v, torch.Tensor) else v
            for name, v in vars(layer[0]).items()} for layer in packed]}
    if backend not in ('flashinfer_fp16', 'page_gauge'):
        raise ValueError('Unsupported backend')
    import benchmark_pg19_external_quality as quality
    pg = quality.PG
    layers = len(legacy)
    _, heads, context, dim = legacy[0][0].shape
    if dim != 128 or context % 16:
        raise ValueError('Unsupported cache shape')
    pages = math.ceil(maximum/16)
    cache = quality.allocate_baseline_cache(layers, pages, 1, heads)
    for layer, (k, v) in enumerate(legacy):
        cache.key[layer, :context//16].copy_(k[0].transpose(0, 1).reshape(-1, 16, heads, dim))
        cache.value[layer, :context//16].copy_(v[0].transpose(0, 1).reshape(-1, 16, heads, dim))
    if backend == 'page_gauge':
        cache = quality.build_gauge_cache_from_baseline(cache, layers, pages, context//16, 48, 1, heads, 4, 128)
        names = ('exact_key', 'exact_value', 'key_codes', 'value_codes', 'key_scales',
                 'value_scales', 'key_center', 'value_center')
    else:
        names = ('key', 'value')
    return {'kind': backend, 'tensors': {name: getattr(cache, name).cpu() for name in names},
            'context': context, 'maximum': maximum}


def merge_field(values, dim, device):
    import torch
    first = values[0]
    if isinstance(first, torch.Tensor):
        if not all(isinstance(v, torch.Tensor) and v.dtype == first.dtype and v.device.type == 'cpu' for v in values):
            raise ValueError('Initial serving snapshots must be matching CPU tensors')
        return torch.cat(values, dim=dim).to(device)
    if not all(v == first for v in values):
        raise ValueError('Native scalar state differs across synchronized requests')
    return first


def restore(model, snapshots, backend, maximum, exact_split=32):
    """Upload only serving state; no retained GPU FP16 fixture or second cache."""
    import torch
    batch = len(snapshots)
    if not batch or len({s['kind'] for s in snapshots}) != 1:
        raise ValueError('Mixed/empty snapshots')
    if backend.startswith('kivi_'):
        from kivi_adapter import configure_decode
        configure_decode(model, 2 if backend == 'kivi_int2' else 4)
        return tuple(tuple(merge_field([s['layers'][i][j] for s in snapshots], 0, 'cuda')
            for j in range(len(snapshots[0]['layers'][i]))) for i in range(len(snapshots[0]['layers'])))
    if backend == 'bitdecode_int4':
        from bitdecode_adapter import configure_decode
        from bitdecode_cache import BitDecodeCache
        configure_decode(model, 4)
        result = []
        for i in range(len(snapshots[0]['layers'])):
            cache = object.__new__(BitDecodeCache)
            for name in snapshots[0]['layers'][i]:
                setattr(cache, name, merge_field([s['layers'][i][name] for s in snapshots], 0, 'cuda'))
            result.append((cache, cache.length))
        return tuple(result)
    import flashinfer
    import benchmark_pg19_external_quality as quality
    from kivi_adapter import configure_decode
    from paged_mistral_adapter import PagedMistralAttention
    pg = quality.PG
    if backend not in ('flashinfer_fp16', 'page_gauge'):
        raise ValueError('Unsupported backend')
    if any(s['maximum'] != maximum or s['context'] != snapshots[0]['context'] for s in snapshots):
        raise ValueError('Mismatched capacity')
    context = snapshots[0]['context']
    tensors = {name: merge_field([s['tensors'][name] for s in snapshots], 1, 'cuda')
               for name in snapshots[0]['tensors']}
    if backend == 'flashinfer_fp16':
        cache = pg.BaselineCache(**tensors)
    else:
        cache = pg.GaugeCache(**tensors, exact_tail_pages=48, exact_sink_pages=4,
            exact_static_suffix_pages=128, initial_context_pages=context//16)
    dummy = torch.empty(1, 8, 1, 128, device='cuda', dtype=torch.float16)
    cos, sin = model.model.layers[0].self_attn.rotary_emb(dummy, seq_len=maximum)
    driver = pg.TransformerDecoder(model, flashinfer, pg.RUNTIME.load_append_extension(), backend,
        cache, maximum, 768, 256, 128, cos.half().contiguous(), sin.half().contiguous(),
        'attention_add', tail_attention='flashinfer_merge', batch_size=batch,
        exact_sink_pages=4 if backend == 'page_gauge' else 0,
        exact_static_suffix_pages=128 if backend == 'page_gauge' else 0,
        initial_context_pages=context//16 if backend == 'page_gauge' else None)
    driver.logical_lengths = [context]*len(model.model.layers)
    driver.exact_split_override = exact_split if backend == 'page_gauge' else None
    if backend == 'page_gauge':
        original = driver.exact_wrapper.plan
        def exact_plan(indices, length, last, split, page_table_epoch=None):
            return original(indices, length, last, exact_split, page_table_epoch)
        driver.exact_wrapper.plan = exact_plan
    configure_decode(model, 4)
    for i, layer in enumerate(model.model.layers):
        layer.self_attn.__class__ = PagedMistralAttention
        layer.self_attn._pagegauge_common_layer_index = i
    return tuple((driver, context) for _ in model.model.layers)


def cpu_snapshot_bytes(snapshots):
    import torch
    stores = {}
    def visit(value):
        if isinstance(value, torch.Tensor):
            if value.device.type != 'cpu':
                raise ValueError('Reference snapshot accidentally retained on GPU')
            stores[value.untyped_storage().data_ptr()] = value.untyped_storage().nbytes()
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)
    visit(snapshots)
    return sum(stores.values())


def verify_restored(past, snapshots, backend):
    """Untimed bitwise check of upload and request-concatenation semantics."""
    import torch
    count = 0
    def check(actual, values, dim):
        nonlocal count
        expected = merge_field(values, dim, 'cpu')
        if isinstance(expected, torch.Tensor):
            # Compare bits, including any uninitialized capacity slots. NaN
            # payloads in unused capacity must not cause a false mismatch.
            observed = actual.detach().cpu().contiguous()
            expected = expected.contiguous()
            if observed.shape != expected.shape or observed.dtype != expected.dtype or not torch.equal(
                    observed.view(torch.uint8), expected.view(torch.uint8)):
                raise ValueError('Initial serving state differs from CPU snapshot')
        elif actual != expected:
            raise ValueError('Initial native scalar state changed')
        count += 1
    if backend in ('flashinfer_fp16', 'page_gauge'):
        for name in snapshots[0]['tensors']:
            check(getattr(past[0][0].cache, name), [s['tensors'][name] for s in snapshots], 1)
    elif backend == 'bitdecode_int4':
        for i, layer in enumerate(past):
            for name in snapshots[0]['layers'][i]:
                check(getattr(layer[0], name), [s['layers'][i][name] for s in snapshots], 0)
    elif backend.startswith('kivi_'):
        for i, layer in enumerate(past):
            for j, value in enumerate(layer):
                check(value, [s['layers'][i][j] for s in snapshots], 0)
    else:
        raise ValueError('Unsupported native state')
    return {'bitwise_initial_state_match': True, 'fields_checked': count}
