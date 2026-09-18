"""Opt-in attention-only CUDA-graph dispatch for the common eager FI/PG body.

Uses the existing production attention capture and structural/pointer guards.
No new attention kernel, cache representation, projection packing or model math.
Every served topology is prepared outside timing; missing banks fail closed.
"""
import hashlib
import time


def oracle_steps(steps):
    return sorted({step for step in (0, 15, 16, 31, 32, 767, 768, 769, 783, 784, steps-1) if 0 <= step < steps})


def prepare(driver, context, steps, validate=False, maximum_banks=64):
    import torch
    if driver.backend not in ('flashinfer_fp16', 'page_gauge') or context % 16 or steps < 785:
        raise ValueError('Audited FI/PG recurrence required')
    original_plan, original_eager = driver.plan, driver.eager_attention
    torch.cuda.synchronize()
    setup_start = time.perf_counter()
    free_before, _ = torch.cuda.mem_get_info()
    allocated_before = torch.cuda.memory_allocated()
    representatives = {}
    for position in range(context, context+steps):
        original_plan(position+1)
        signature = driver.structural_plan_signature()
        representatives.setdefault(signature, position+1)
        if len(representatives) > maximum_banks:
            raise ValueError('Too many attention graph topologies; no partial eager fallback')
    torch.cuda.synchronize()
    preflight_seconds = time.perf_counter()-setup_start
    capture_start = time.perf_counter()
    stats = {'mode': 'attention_only_graphs', 'backend': driver.backend,
        'bank_count': len(representatives), 'graphs_per_bank': driver.layers,
        'representative_contexts': list(representatives.values()),
        'signature_sha256': [hashlib.sha256(repr(key).encode()).hexdigest() for key in representatives],
        'served_plan_calls': 0, 'graph_replays': 0, 'missing_bank_count': 0,
        'same_cache_oracle_steps': oracle_steps(steps) if validate else [],
        'same_cache_oracle_calls': 0, 'max_same_cache_relative_l2': 0.0, 'max_same_cache_abs': 0.0,
        'same_cache_tolerances': {'relative_l2': .005, 'absolute': .02},
        'scope': 'Only attention-body host dispatch changes. Original append/planning/dense body and FI/PG kernels remain in scope. Same-cache oracle is execution-only, not a predictive quality threshold.'}
    banks = {}
    for signature, representative in representatives.items():
        driver.capture_attention_graphs(representative)
        if driver.structural_plan_signature() != signature:
            raise ValueError('Topology changed between preflight and capture')
        if len(driver.attention_graphs) != driver.layers:
            raise ValueError('Incomplete layer capture')
        banks[signature] = driver.attention_graphs
    torch.cuda.synchronize()
    stats['preflight_seconds'] = preflight_seconds
    stats['capture_seconds'] = time.perf_counter()-capture_start
    # Capture is read-only on served KV; the validation caller verifies every
    # restored cache bit against the original CPU snapshots after this returns.
    original_plan(context)
    driver.reset_runtime_operation_counts()
    driver.reset_attention_dispatch_counts()
    active = {'bank': None, 'position': None}
    selected_checks = set(stats['same_cache_oracle_steps'])

    def plan(served_context):
        if not context < served_context <= context+steps:
            raise ValueError('Served position outside prepared graph trajectory')
        original_plan(served_context)
        signature = driver.structural_plan_signature()
        if signature not in banks:
            stats['missing_bank_count'] += 1
            raise ValueError('Uncaptured attention topology/pointer change')
        active['bank'] = banks[signature]
        active['position'] = served_context-1
        stats['served_plan_calls'] += 1

    def replay(layer):
        if active['bank'] is None or not 0 <= layer < driver.layers:
            raise ValueError('Attention replay before valid planning')
        active['bank'][layer].replay()
        stats['graph_replays'] += 1
        result = driver.attention_output[layer]
        if validate and active['position']-context in selected_checks:
            saved = result.clone()
            expected = original_eager(layer)
            delta = saved.float()-expected.float()
            absolute = float(delta.abs().max().item())
            relative = float((delta.norm()/expected.float().norm().clamp_min(1e-30)).item())
            if not bool(saved.isfinite().all()) or not bool(expected.isfinite().all()):
                raise ValueError('Nonfinite same-cache execution output')
            stats['same_cache_oracle_calls'] += 1
            stats['max_same_cache_relative_l2'] = max(stats['max_same_cache_relative_l2'], relative)
            stats['max_same_cache_abs'] = max(stats['max_same_cache_abs'], absolute)
            if relative > .005 or absolute > .02:
                raise ValueError('Attention graph execution differs from original same-cache kernel')
            # Continue the model with the graph output, NOT the oracle output.
            return saved
        return result

    # The common adapter's call site is named eager_attention. This explicit
    # per-instance replacement changes dispatch only; graph capture above used
    # the untouched original implementation, which is retained for the oracle.
    driver.plan = plan
    driver.eager_attention = replay
    torch.cuda.synchronize()
    free_after, _ = torch.cuda.mem_get_info()
    stats.update(total_graph_setup_seconds=time.perf_counter()-setup_start,
        graph_setup_allocated_delta_bytes=torch.cuda.memory_allocated()-allocated_before,
        graph_setup_device_used_delta_bytes=free_before-free_after)
    return stats


def verify_stats(stats, steps, layers, validate):
    expected = layers*len(oracle_steps(steps)) if validate else 0
    if (stats['served_plan_calls'] != steps or stats['graph_replays'] != layers*steps
            or stats['missing_bank_count'] or stats['same_cache_oracle_calls'] != expected):
        raise ValueError('Incomplete graph dispatch/oracle trajectory')
