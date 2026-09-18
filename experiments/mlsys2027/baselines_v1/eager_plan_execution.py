"""Non-graph planning and VRAM-sharded reconstructed-cache execution checks."""
import math


def reference_head(driver, layer, request, head):
    """One KV head/request at a time; never reconstruct the full B-way cache."""
    import torch
    cache = driver.cache
    def indices(wrapper):
        first, last = (int(wrapper.indptr[j].item()) for j in (request, request+1))
        ids = wrapper.indices[first:last].long()
        length = (len(ids)-1)*16+int(wrapper.last_len[request].item())
        return ids, length
    if driver.backend == 'flashinfer_fp16':
        ids, count = indices(driver.baseline_wrapper)
        keys = cache.key[layer, ids, :, head, :].reshape(-1, 128)[:count].float()
        values = cache.value[layer, ids, :, head, :].reshape(-1, 128)[:count].float()
        center = 0
    else:
        ids, count = indices(driver.old_wrapper)
        keys = (cache.key_codes[layer, ids, :, head, :].float()*
                cache.key_scales[layer, ids, head].float().reshape(-1, 1, 1)).reshape(-1, 128)[:count]
        values = (cache.value_codes[layer, ids, :, head, :].float()*
                  cache.value_scales[layer, ids, head].float().reshape(-1, 1, 1)).reshape(-1, 128)[:count]
        exact_ids, exact_count = indices(driver.exact_wrapper)
        exact_keys = cache.exact_key[layer, exact_ids, :, head, :].reshape(-1, 128)[:exact_count].float()
        exact_values = cache.exact_value[layer, exact_ids, :, head, :].reshape(-1, 128)[:exact_count].float()
        keys = torch.cat((keys, exact_keys))
        values = torch.cat((values, exact_values))
        # output_center is expanded to QUERY heads, unlike value_center.
        center = cache.output_center[layer, request, head*4:(head+1)*4].float()
    if keys.shape[0] != driver.logical_lengths[layer]+1 or keys.shape != values.shape:
        raise ValueError('Oracle page tables do not cover the complete served cache')
    query = driver.rotated_query[request, head*4:(head+1)*4].float()
    return (query@keys.T/math.sqrt(128)).softmax(-1)@values+center


def install_oracle(driver, context, steps, stats):
    original = driver.eager_attention
    selected_steps, layers = {0, 768, steps-1}, {0, 15, 31}
    requests, heads = sorted({0, driver.batch_size-1}), (0, 7)
    stats.update(oracle_steps=sorted(selected_steps), oracle_layers=sorted(layers),
        oracle_requests=requests, oracle_kv_heads=list(heads), oracle_calls=0,
        maximum_relative_l2=0.0, maximum_absolute_error=0.0,
        tolerances={'relative_l2': .005, 'absolute': .02},
        oracle_scope='Selected same-cache FP32 explicit reconstruction, sharded one KV head/request for VRAM. Not original-FP16 predictive quality or an all-head/all-step execution claim.')
    def attention(layer):
        result = original(layer)
        if layer in layers and driver.logical_lengths[layer]-context in selected_steps:
            for request in requests:
                for head in heads:
                    expected = reference_head(driver, layer, request, head)
                    actual = result[request, head*4:(head+1)*4].float()
                    delta = actual-expected
                    absolute = float(delta.abs().max().item())
                    relative = float((delta.norm()/expected.norm().clamp_min(1e-30)).item())
                    if not bool(actual.isfinite().all()) or not bool(expected.isfinite().all()):
                        raise ValueError('Nonfinite reconstructed-cache execution output')
                    stats['oracle_calls'] += 1
                    stats['maximum_relative_l2'] = max(stats['maximum_relative_l2'], relative)
                    stats['maximum_absolute_error'] = max(stats['maximum_absolute_error'], absolute)
                    stats['last_oracle'] = {'step': driver.logical_lengths[layer]-context,
                        'layer': layer, 'request': request, 'kv_head': head,
                        'relative_l2': relative, 'absolute': absolute}
                    if relative > .005 or absolute > .02:
                        raise ValueError('Reconstructed-cache execution tolerance exceeded: '+str(stats['last_oracle']))
        # Continue with actual production output, never the reconstructed oracle.
        return result
    driver.eager_attention = attention


def verify_oracle(stats, validation):
    if not stats['wrapper_count'] or stats['graph_enabled_count']:
        raise ValueError('Missing wrappers or graph planner accidentally enabled')
    if validation:
        expected = math.prod(len(stats[name]) for name in
            ('oracle_steps', 'oracle_layers', 'oracle_requests', 'oracle_kv_heads'))
        if stats['oracle_calls'] != expected:
            raise ValueError('Incomplete selected execution-oracle trajectory')
