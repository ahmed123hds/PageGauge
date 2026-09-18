"""Eight fresh-process E1 blocks; same compressed cache, different scale placement."""
import json
import math
import os
from pathlib import Path
import statistics
import sys
from datetime import datetime, timezone
import uuid

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'diagnostics'))
import mlsys_rtx5090_entry as base
import mlsys_rtx5090_step02 as previous
from control import prepare, EXPECTED


def schedule():
    return [dict(row, arm=('register_control' if row['backend'] == 'flashinfer_fp16' else 'factorized'),
                 backend='page_gauge') for row in previous.schedule()]


def assess(p, block, manifest, completion, experiment_validator=None):
    require = previous.require
    require(completion['return_code'] in (0, 2) and completion['sampled_exclusivity_passed'],
            'Worker failure or device interference')
    require(p.get('schema_version') == 3 and p['backend'] == 'page_gauge', 'Wrong worker/schema')
    if experiment_validator is None:
        require(p['factorization_experiment']['arm'] == block['arm'], 'Wrong arm')
        require(p['factorization_experiment']['adapter_sha256'] == manifest['source_sha256'][
            'experiments/mlsys2027/factorization_v1/full_model_worker.py'], 'Wrong adapter')
    else:
        experiment_validator(p, block, manifest)
    expected = previous.worker_config(block, manifest['inputs']['model_path'])
    expected['min_logits_cosine'] = -1.0
    for name, value in expected.items():
        require(p['configuration'].get(name) == value, 'Configuration mismatch: '+name)
    require(p['configuration']['value_conditioning_mode'] == 'none', 'Unexpected conditioning')
    require('RTX 5090' in p['environment']['gpu'] and p['environment']['compute_capability'] == [12, 0], 'Wrong device')
    for key, package in (('torch','torch'), ('flashinfer','flashinfer-python'), ('transformers','transformers')):
        require(p['environment'][key] == manifest['environment'][package], 'Package drift')
    for name, sha in p['source_sha256'].items():
        require(manifest['source_sha256'].get(name) == sha, 'Unfrozen worker source '+name)
    expected_header = EXPECTED if experiment_validator is not None or block['arm'] == 'factorized' else manifest['control_header_sha256']
    require(p['attention_implementation']['custom_module_source_hashes']['header_sha256'] == expected_header,
            'Wrong actual kernel header')
    same = p['correctness']['same_backend_eager_vs_graph']
    require(same['passed'] and same['eager_gate_passed'], 'Eager/graph mismatch')
    for comparison in (same, same['restored_graph_repeat']):
        require(comparison['passed'], 'Restored repeat mismatch')
        for field in ('full_mutated_cache_range', 'every_page_close_cache_digest', 'final_serving_metadata'):
            require(comparison[field]['passed'], 'Recurrence mismatch '+field)
        require(comparison['every_page_close_cache_digest']['checked_page_closes'] == 96, 'Missing page closes')
    recurrence = p['correctness']['runtime_page_finalization_and_consumption']
    require(recurrence['passed'] and recurrence['runtime_finalized_pages_consumed_as_int8'] == list(range(1280,1328)),
            'Missing quantized recurrence')
    for field in ('logical_token_coverage_exactly_once', 'logical_page_sets_disjoint', 'prefix_exclusion_gate_passed',
                  'static_suffix_exclusion_gate_passed', 'old_attention_page_table_gate_passed',
                  'exact_attention_page_table_gate_passed', 'final_attention_page_table_gate_passed'):
        require(recurrence[field], 'Partition gate '+field)
    graph = p['cuda_graph_provenance']
    require(graph['structure_gate_passed'] and graph['strict_missing_bucket_failure']
            and graph['graph_bank_misses'] == 0 and graph['preflight_position_count'] == 1536
            and graph['graphs_per_bank'] == 32 and not graph['nested_attention_graphs'], 'Graph fallback/structure')
    require(p['scheduler_capacity']['all_wrappers_analytically_within_capacity'], 'Scheduler capacity')
    for mode in previous.MODES:
        samples = p['timing_modes'][mode]['raw_samples']
        require(len(samples) == 3 and [r['sample_index'] for r in samples] == [0,1,2], 'Missing samples')
        for sample in samples:
            previous.validate_runtime(sample['runtime_gate'], 'page_gauge')
            require(sample['exact_prefix_canary']['passed'], 'Canary failure')
            for metric in previous.METRICS:
                require(math.isfinite(sample[metric]) and sample[metric] > 0, 'Invalid timing')
    accounting = previous.expected_cache('page_gauge')
    cache = p['cache_build']
    require(cache['selected_backend_cache_served_bytes_excluding_following_canary'] == accounting['served_bytes'],
            'Changed served cache')
    tensors = cache['selected_backend_cache']['tensors']
    require(set(tensors) == set(accounting['tensors']), 'Unexpected cache tensors')
    for name, fields in accounting['tensors'].items():
        require(all(tensors[name][k] == v for k,v in fields.items()), 'Cache layout mismatch '+name)
    require(not cache['opposite_backend_full_gpu_cache_allocated']
            and not p['exclusivity']['opposite_backend_full_gpu_cache_allocated']
            and p['exclusivity']['hf_dynamic_caches_released_before_decoder_construction'], 'Cache residency')
    hf = p['correctness']['backend_vs_hf_sdpa_fp16']
    require(completion['return_code'] == (0 if hf['passed'] else 2), 'Unexpected quality/exit status')
    return {'execution_passed':True, 'hf_diagnostic':hf, 'served_bytes':accounting['served_bytes']}


def reduce_results(payloads, blocks, bootstrap_samples=50000):
    if len(payloads) != 8 or len(blocks) != 8:
        raise ValueError('All eight fresh-process blocks required')
    pairs = []
    for i in range(0,8,2):
        c = next(j for j in (i,i+1) if blocks[j]['arm'] == 'register_control')
        f = next(j for j in (i,i+1) if blocks[j]['arm'] == 'factorized')
        for key in previous.MATCHED_FIELDS:
            previous.require(payloads[c]['pairing']['configuration'][key] == payloads[f]['pairing']['configuration'][key],
                             'Unmatched fixture '+key)
        previous.require(payloads[c]['timed_work'] == payloads[f]['timed_work'], 'Different timed work')
        pairs.append((c,f,blocks[i]))
    endpoints = {}
    for mode in previous.MODES:
        for metric in previous.METRICS:
            rows = []
            for c,f,block in pairs:
                means = [statistics.fmean(math.log(r[metric]) for r in payloads[j]['timing_modes'][mode]['raw_samples'])
                         for j in (c,f)]
                rows.append({'seed':block['seed'], 'pair_id':block['pair_id'], 'pair_order':block['pair_order'],
                    'log_speedup':means[0]-means[1], 'register_control_ms_per_step':math.exp(means[0])/1536,
                    'factorized_ms_per_step':math.exp(means[1])/1536})
            endpoints[mode+'.'+metric] = {'register_over_factorized':math.exp(statistics.fmean(r['log_speedup'] for r in rows)),
                **previous._hierarchical_bootstrap_ci(rows, bootstrap_samples, 5140), 'pairs':rows}
    return {'status':'complete', 'execution_passed':True, 'endpoints':endpoints,
        'scope':'Scale placement only; both arms retain common-center factoring and identical mixed INT8/FP16 storage.',
        'fixture_clusters':2, 'fresh_process_blocks':8, 'adjacent_pairs':4,
        'interpretation':'Ratio >1 favors factoring; report interval and negative/tied outcomes. Not a FlashInfer-relative speedup.',
        'quality':[p['correctness']['backend_vs_hf_sdpa_fp16'] for p in payloads]}


def main():
    import fcntl
    old = ROOT/'results/mlsys2027_rtx5090/02_fp16_performance/20260908T041311Z_076aaa6c'
    original = json.loads((old/'manifest.json').read_text())
    idle = base.idle_preflight(0)
    gpu = idle[-1]['uuid']
    with (Path('/tmp')/f'pagegauge-mlsys-{gpu}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        include, control_sha = prepare()
        names = set(original['source_sha256']) | {'scripts/page_gauge_value_conditioning.py',
            'experiments/mlsys2027/factorization_v1/control.py', 'experiments/mlsys2027/factorization_v1/full_model.py',
            'experiments/mlsys2027/factorization_v1/full_model_worker.py',
            'experiments/mlsys2027/factorization_v1/run.sh', 'experiments/mlsys2027/representation_v2/run.sh'}
        # Freeze all actually included control headers, not only the changed file.
        names.update(str(p.relative_to(ROOT)) for p in include.rglob('*') if p.is_file())
        manifest = {**original, 'experiment':'E1_centered_register_full_model', 'schedule':schedule(),
            'source_sha256':{name:base.sha256_file(ROOT/name) for name in sorted(names)},
            'control_header_sha256':control_sha, 'created_utc':datetime.now(timezone.utc).isoformat(),
            'scope':'TRAIN development; scale-placement contrast, both retain shared-center algebra; default policy unchanged',
            'statistics':{'primary':'cache_neutral.wall_ms', 'bootstrap_samples':50000, 'bootstrap_seed':5140,
                          'interpretation':'Report ratio and CI around 1, with no required positive outcome'},
            'quality_policy':'HF cosine descriptive; report all quality metrics; no PPL or final TEST claim',
            'config':{**original['config'], 'min_logits_cosine':-1.0}}
        manifest.pop('manifest_sha256', None)
        manifest['manifest_sha256'] = base.canonical_hash(manifest)
        previous.verify_frozen(manifest, full_inputs=True)
        out = ROOT/'results/mlsys2027_factorization_v1'/('full_model_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
        out.mkdir(parents=True)
        base.atomic_json(out/'manifest.json', manifest)
        base.atomic_json(out/'orchestrator.json', {'pid':os.getpid(), 'gpu':gpu})
        print('E1 full-model output: '+str(out), flush=True)
        payloads = []
        os.environ['PAGEGAUGE_VALUE_CONDITIONING'] = 'none'
        for block in manifest['schedule']:
            previous.verify_frozen(manifest)
            base.idle_preflight(0)
            os.environ['PAGEGAUGE_FACTORIZATION_ARM'] = block['arm']
            target = out/f"block_{block['index']}.json"
            cmd = previous.worker_command(block, manifest['inputs']['model_path'], target)
            cmd[2] = str(ROOT/'experiments/mlsys2027/factorization_v1/full_model_worker.py')
            cmd[cmd.index('--min-logits-cosine')+1] = '-1'
            base.atomic_json(out/f"block_{block['index']}_invocation.json", {'command':cmd, 'arm':block['arm']})
            print(f"E1 block {block['index']+1}/8: {block['arm']}", flush=True)
            completion = previous.run_process(cmd, out, block, gpu)
            base.atomic_json(out/f"block_{block['index']}_completion.json", completion)
            p = json.loads(target.read_text())
            assessment = assess(p, block, manifest, completion)
            previous.verify_frozen(manifest)
            base.atomic_json(out/f"block_{block['index']}_assessment.json", {**assessment,'result_sha256':base.sha256_file(target)})
            payloads.append(p)
        previous.verify_frozen(manifest, full_inputs=True)
        analysis = reduce_results(payloads, manifest['schedule'])
        analysis['manifest_sha256'] = manifest['manifest_sha256']
        base.atomic_json(out/'analysis.json', analysis)
        print(json.dumps(analysis, indent=2), flush=True)


if __name__ == '__main__':
    main()
