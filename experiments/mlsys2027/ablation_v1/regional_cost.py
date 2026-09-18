"""Four fixed regional policies, clean fresh-process B4 decoder-cost pilots.

No production edits, policy search, quality-loop timing, or final TEST use.
"""
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import sys
import uuid

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'diagnostics'))
sys.path.insert(0, str(ROOT/'experiments/mlsys2027'))
sys.path.insert(0, str(ROOT/'experiments/mlsys2027/baselines_v1'))
import mlsys_rtx5090_entry as base
import mlsys_rtx5090_step02 as previous
from regional_quality import POLICIES
import wsl_gpu_monitor

OLD_ADAPTER = 'experiments/mlsys2027/baselines_v1/optimized_split_worker.py'
ADAPTER = 'experiments/mlsys2027/ablation_v1/regional_split_worker.py'


def configuration(block, model):
    _, prefix, suffix, tail = next(p for p in POLICIES if p[0] == block['policy'])
    return {**previous.worker_config(block, model), 'exact_sink_pages': prefix,
        'exact_prefix_pages': prefix, 'exact_static_suffix_pages': suffix,
        'exact_tail_tokens': tail, 'min_logits_cosine': -1.0,
        'candidate_split_pages': block.get('history_split_pages', 128)}


def validate_launch(p, block, manifest):
    require = previous.require
    history = block.get('history_split_pages', 128)
    adapter = block.get('adapter', OLD_ADAPTER)
    r = p['exact_split_experiment']; observed = r['observed_planning']
    require(r['adapter_sha256'] == manifest['source_sha256'][adapter], 'Wrong launch adapter')
    require(r['exact_split_pages'] == block['exact_split_pages'], 'Wrong exact split')
    require(not r['production_sources_modified'] and not r['quantization_or_kernel_changed'], 'Changed kernel/representation')
    require(observed['decoder_instances'] > 0 and observed['exact_plan_calls'] >= 1536 and
        observed['exact_plan_calls'] == observed['old_plan_calls'], 'Missing observed planning')
    require(observed['requested_exact_splits'] == [history] and observed['effective_exact_splits'] == [block['exact_split_pages']]
        and observed['old_splits'] == [history], 'Launch shape drift')
    capacity = p['scheduler_capacity']['wrappers']
    require(capacity['exact_fp16']['fixed_split_pages'] == block['exact_split_pages'] and
        capacity['old_int8']['fixed_split_pages'] == history, 'Wrong actual capacity report')


def accounting(config):
    """Independent fixed-shape calculation; only exact FP16 storage changes."""
    expected = previous.expected_cache('page_gauge')
    pages = config['exact_sink_pages']+config['exact_static_suffix_pages']+config['exact_tail_tokens']//16
    for name in ('exact_key', 'exact_value'):
        shape = [32, 4*pages, 16, 8, 128]
        expected['tensors'][name] = {'shape': shape, 'dtype': 'torch.float16', 'bytes': math.prod(shape)*2}
    expected['total_bytes'] = sum(t['bytes'] for t in expected['tensors'].values())
    expected['served_bytes'] = expected['total_bytes']-expected['canary_bytes']
    return expected


def validate_runtime(gate, config):
    require = previous.require
    require(gate['passed'], 'Runtime count gate failed')
    dispatch = gate['observed_dispatch']
    require(all(dispatch[k] == v for k, v in {'graph_replays': 49152, 'eager_calls': 0,
        'graph_bank_misses': 0}.items()), 'Graph dispatch changed')
    require(gate['observed_nested_attention_dispatch']['total_calls'] == 0, 'Nested fallback')
    operations = gate['observed_operations']
    for k, v in {'decoder_plan_calls': 1536, 'device_position_fills': 1536,
                 'heterogeneous_page_table_updates': 0, 'exact_page_table_updates': 96}.items():
        require(operations[k] == v, 'Wrong operation count '+k)
    rebuilds = {'old_int8': 96-min(config['exact_static_suffix_pages'], config['exact_tail_tokens']//16),
                'exact_fp16': 96}
    require(set(operations['wrappers']) == set(rebuilds), 'Wrong wrappers')
    for name, n in rebuilds.items():
        require(operations['wrappers'][name] == {'plan_invocations': 1536, 'plan_rebuilds': n,
            'last_page_len_device_fills': 1536}, 'Wrong wrapper counts '+name)


def assess(p, block, manifest, completion):
    """Retain execution checks; expected changes are explicit policy functions."""
    require = previous.require
    require(completion['return_code'] in (0, 2) and completion['sampled_exclusivity_passed'],
            'Worker failure or GPU interference')
    require(p['schema_version'] == 3 and p['backend'] == 'page_gauge', 'Wrong worker')
    validate_launch(p, block, manifest)
    config = configuration(block, manifest['inputs']['model_path'])
    for key, value in config.items():
        require(p['configuration'].get(key) == value, 'Wrong configuration '+key)
    require(p['configuration']['value_conditioning_mode'] == 'none', 'Conditioning changed')
    require('RTX 5090' in p['environment']['gpu'] and p['environment']['compute_capability'] == [12, 0], 'Wrong GPU')
    for key, package in (('torch', 'torch'), ('flashinfer', 'flashinfer-python'), ('transformers', 'transformers')):
        require(p['environment'][key] == manifest['environment'][package], 'Package drift')
    for name, digest in p['source_sha256'].items():
        require(manifest['source_sha256'].get(name) == digest, 'Unfrozen worker '+name)
    require(p['attention_implementation']['custom_module_source_hashes']['header_sha256'] ==
            manifest['kernel_header_sha256'], 'Changed kernel header')
    same = p['correctness']['same_backend_eager_vs_graph']
    require(same['passed'] and same['eager_gate_passed'], 'Eager/graph mismatch')
    for comparison in (same, same['restored_graph_repeat']):
        require(comparison['passed'], 'Restored graph mismatch')
        for name in ('full_mutated_cache_range', 'every_page_close_cache_digest', 'final_serving_metadata'):
            require(comparison[name]['passed'], 'Cache recurrence mismatch '+name)
        require(comparison['every_page_close_cache_digest']['checked_page_closes'] == 96, 'Missing page closes')
    recurrence = p['correctness']['runtime_page_finalization_and_consumption']
    require(recurrence['passed'] and recurrence['runtime_finalized_pages_consumed_as_int8'] ==
            list(range(1280, 1376-config['exact_tail_tokens']//16)), 'Missing new history consumption')
    for name in ('logical_token_coverage_exactly_once', 'logical_page_sets_disjoint', 'prefix_exclusion_gate_passed',
                 'static_suffix_exclusion_gate_passed', 'old_attention_page_table_gate_passed',
                 'exact_attention_page_table_gate_passed', 'final_attention_page_table_gate_passed'):
        require(recurrence[name], 'Partition gate '+name)
    graph = p['cuda_graph_provenance']
    require(graph['structure_gate_passed'] and graph['strict_missing_bucket_failure'] and
        graph['graph_bank_misses'] == 0 and graph['preflight_position_count'] == 1536 and
        graph['graphs_per_bank'] == 32 and not graph['nested_attention_graphs'], 'Graph structure/fallback')
    require(p['scheduler_capacity']['all_wrappers_analytically_within_capacity'], 'Scheduler capacity')
    for mode in previous.MODES:
        samples = p['timing_modes'][mode]['raw_samples']
        require(len(samples) == 3 and [r['sample_index'] for r in samples] == [0, 1, 2], 'Missing repeats')
        for sample in samples:
            validate_runtime(sample['runtime_gate'], config)
            require(sample['exact_prefix_canary']['passed'], 'Canary mismatch')
            for metric in previous.METRICS:
                require(math.isfinite(sample[metric]) and sample[metric] > 0, 'Invalid timing')
    expected = accounting(config)
    cache = p['cache_build']
    require(cache['selected_backend_cache_served_bytes_excluding_following_canary'] == expected['served_bytes'],
            'Wrong served cache bytes')
    tensors = cache['selected_backend_cache']['tensors']
    require(set(tensors) == set(expected['tensors']), 'Unexpected tensors')
    for name, fields in expected['tensors'].items():
        require(all(tensors[name][k] == v for k, v in fields.items()), 'Wrong cache layout '+name)
    require(not cache['opposite_backend_full_gpu_cache_allocated'] and
        not p['exclusivity']['opposite_backend_full_gpu_cache_allocated'] and
        p['exclusivity']['hf_dynamic_caches_released_before_decoder_construction'], 'Opposite cache residency')
    hf = p['correctness']['backend_vs_hf_sdpa_fp16']
    require(completion['return_code'] == (0 if hf['passed'] else 2), 'Wrong quality exit status')
    return {'execution_passed': True, 'hf_diagnostic': hf, 'served_bytes': expected['served_bytes']}


def reduce_results(payloads, blocks):
    if len(payloads) != 4 or [b['policy'] for b in blocks] != [p[0] for p in POLICIES]:
        raise ValueError('All four fixed policies required')
    for p in payloads[1:]:
        for key in previous.MATCHED_FIELDS:
            previous.require(p['pairing']['configuration'][key] == payloads[0]['pairing']['configuration'][key],
                             'Unmatched fixture '+key)
        previous.require(p['timed_work'] == payloads[0]['timed_work'], 'Different timed-work scope')
    rows = {}
    for p, block in zip(payloads, blocks):
        row = {'served_bytes': p['cache_build']['selected_backend_cache_served_bytes_excluding_following_canary']}
        for mode in previous.MODES:
            row[mode] = {}
            for metric in previous.METRICS:
                values = [s[metric]/1536 for s in p['timing_modes'][mode]['raw_samples']]
                row[mode][metric] = {'median_ms_per_step': statistics.median(values),
                    'range_ms_per_step': [min(values), max(values)], 'raw_ms_per_step': values}
        rows[block['policy']] = row
    return {'status': 'complete', 'execution_passed': True, 'rows': rows, 'fresh_process_blocks': 4,
        'fixture_clusters': 1, 'repeats_per_mode_per_process': 3,
        'scope': 'Fixed four-policy B4/C20480/D1536 optimized full-decoder pilots. One process per policy, fixed order, no comparative CI or policy promotion. TRAIN teacher forcing, not online serving. Excludes prefill, load, capture, restore and scrub; includes full decoder, append/finalize, planner, LM head and argmax.'}


def main():
    import fcntl
    old = ROOT/'results/mlsys2027_rtx5090/02_fp16_performance/20260908T041311Z_076aaa6c'
    original = json.loads((old/'manifest.json').read_text())
    blocks = [dict(previous.schedule()[0], index=i, policy=policy[0], backend='page_gauge', exact_split_pages=32,
                   history_split_pages=160, adapter=ADAPTER)
              for i, policy in enumerate(POLICIES)]
    names = set(original['source_sha256']) | {'scripts/page_gauge_value_conditioning.py', ADAPTER, OLD_ADAPTER,
        'experiments/mlsys2027/wsl_gpu_monitor.py',
        'experiments/mlsys2027/ablation_v1/regional_cost.py', 'experiments/mlsys2027/ablation_v1/regional_quality.py',
        'experiments/mlsys2027/ablation_v1/run_clean_cost.sh',
        'experiments/mlsys2027/baselines_v1/optimized_split.py',
        'experiments/mlsys2027/factorization_v1/full_model.py', 'experiments/mlsys2027/factorization_v1/control.py',
        'experiments/mlsys2027/representation_v2/run.sh'}
    # Import closures used by the policy definition, although no quality loop is run.
    names.update(str(p.relative_to(ROOT)) for p in (ROOT/'experiments/mlsys2027/generalization_v1').glob('*.py'))
    sys.path.insert(0, str(ROOT/'experiments/mlsys2027/factorization_v1'))
    from control import EXPECTED
    idle = base.idle_preflight(0); gpu = idle[-1]['uuid']
    with (Path('/tmp')/f'pagegauge-mlsys-{gpu}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        m = {**original, 'experiment': 'E3_regional_policy_clean_cost_pilots', 'schedule': blocks,
            'source_sha256': {n: base.sha256_file(ROOT/n) for n in sorted(names)},
            'kernel_header_sha256': EXPECTED, 'created_utc': datetime.now(timezone.utc).isoformat(), 'idle': idle,
            'config': {**original['config'], 'min_logits_cosine': -1.0, 'candidate_split_pages': 160},
            'launch_capacity_repair': 'All four policies use exact32/history160. A0 requires ceil(1324/10)=133 pages per history chunk;160 is the next multiple of32. Not timing-selected; original history128 records retained.',
            'statistics': {'unit': 'one fresh process per policy', 'primary': 'cache_neutral.wall_ms',
                'interpretation': 'Pilot medians/ranges only; no CI, threshold or outcome-based stopping'},
            'quality_policy': 'HF diagnostics descriptive, including failures; no independent TEST claim',
            'process_monitor': {'contract': wsl_gpu_monitor.CONTRACT,
                'validation': 'results/mlsys2027_monitor_v1/20260909T022502Z_0cdbbf6a/analysis.json',
                'scope': 'Sampled Linux all-user /dev/dxg ownership plus Windows NVML. Distinct from historical Linux NVML-only contract.'},
            'scope': 'All four policies regardless direction; no kernel/quantizer edits or production default change'}
        m.pop('manifest_sha256', None); m['manifest_sha256'] = base.canonical_hash(m)
        previous.verify_frozen(m, full_inputs=True)
        out = ROOT/'results/mlsys2027_ablation_v1'/('regional_cost_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
        out.mkdir(parents=True); base.atomic_json(out/'manifest.json', m)
        base.atomic_json(out/'orchestrator.json', {'pid': os.getpid(), 'gpu': gpu})
        print('Regional clean cost: '+str(out), flush=True)
        os.environ['PAGEGAUGE_VALUE_CONDITIONING'] = 'none'
        os.environ['PAGEGAUGE_EXPERIMENT_EXACT_SPLIT'] = '32'
        payloads = []
        for b in blocks:
            previous.verify_frozen(m); base.idle_preflight(0)
            target = out/f"block_{b['index']}.json"
            command = previous.worker_command(b, m['inputs']['model_path'], target)
            command[2] = str(ROOT/ADAPTER)
            config = configuration(b, m['inputs']['model_path'])
            for flag, key in (('--exact-sink-pages', 'exact_sink_pages'), ('--exact-static-suffix-pages', 'exact_static_suffix_pages'),
                              ('--exact-tail', 'exact_tail_tokens'), ('--min-logits-cosine', 'min_logits_cosine'),
                              ('--candidate-split-pages', 'candidate_split_pages')):
                command[command.index(flag)+1] = str(config[key])
            base.atomic_json(out/f"block_{b['index']}_invocation.json", {'command': command, 'policy': b['policy']})
            print('Regional clean cost policy '+b['policy'], flush=True)
            completion = wsl_gpu_monitor.run_process(previous, command, out, b, gpu)
            base.atomic_json(out/f"block_{b['index']}_completion.json", completion)
            if completion['return_code'] not in (0, 2) or not target.is_file():
                base.atomic_json(out/'failure.json', {'block': b, 'completion': completion,
                    'scope': 'Worker error before a valid result; inspect retained log, never label a capacity or numerical outcome without diagnosis'})
                raise RuntimeError('Regional worker failed; original log and completion retained')
            p = json.loads(target.read_text()); checks = assess(p, b, m, completion)
            previous.verify_frozen(m)
            base.atomic_json(out/f"block_{b['index']}_assessment.json", {**checks, 'result_sha256': base.sha256_file(target)})
            payloads.append(p)
        previous.verify_frozen(m, full_inputs=True)
        result = reduce_results(payloads, blocks); result['manifest_sha256'] = m['manifest_sha256']
        base.atomic_json(out/'analysis.json', result)
        print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
