"""Balanced reference/A0 and FI/A0 development contrasts; no default promotion.

Each requested contrast uses eight fresh processes (ABBA/BAAB), two exposed
TRAIN fixture clusters and a hierarchical paired bootstrap. Run both contrasts
regardless of point-estimate direction. This does not establish final quality.
"""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import sys
import uuid

from regional_cost import ROOT, previous, base, assess, configuration, ADAPTER
import wsl_gpu_monitor

def schedule(contrast):
    if contrast not in ('reference_a0', 'fi_a0'):
        raise ValueError('Unsupported contrast')
    result = []
    for block in previous.schedule():
        reference = block['backend'] == 'flashinfer_fp16'
        backend = 'flashinfer_fp16' if reference and contrast == 'fi_a0' else 'page_gauge'
        result.append(dict(block, backend=backend, role='reference' if reference else 'a0',
            policy='reference_policy' if reference else 'without_static_suffix', exact_split_pages=32,
            history_split_pages=160, adapter=ADAPTER))
    return result


def reduce_results(payloads, blocks):
    if len(payloads) != 8 or len(blocks) != 8: raise ValueError('All eight fresh blocks required')
    endpoints = {}
    for mode in previous.MODES:
        for metric in previous.METRICS:
            rows = []
            for i in range(0, 8, 2):
                a = next(j for j in (i, i+1) if blocks[j]['role'] == 'reference')
                b = next(j for j in (i, i+1) if blocks[j]['role'] == 'a0')
                for key in previous.MATCHED_FIELDS:
                    previous.require(payloads[a]['pairing']['configuration'][key] ==
                        payloads[b]['pairing']['configuration'][key], 'Different paired fixture '+key)
                previous.require(payloads[a]['timed_work'] == payloads[b]['timed_work'], 'Different timed work')
                means = [statistics.fmean(math.log(s[metric]) for s in payloads[j]['timing_modes'][mode]['raw_samples']) for j in (a,b)]
                rows.append({'seed': blocks[i]['seed'], 'pair_id': blocks[i]['pair_id'], 'pair_order': blocks[i]['pair_order'],
                    'log_speedup': means[0]-means[1], 'reference_ms_per_step': math.exp(means[0])/1536,
                    'a0_ms_per_step': math.exp(means[1])/1536})
            endpoints[mode+'.'+metric] = {'reference_over_a0': math.exp(statistics.fmean(r['log_speedup'] for r in rows)),
                **previous._hierarchical_bootstrap_ci(rows, 50000, 2026090912), 'pairs': rows}
    return {'status': 'complete', 'execution_passed': True, 'endpoints': endpoints,
        'fresh_process_blocks': 8, 'fixture_clusters': 2, 'adjacent_pairs': 4,
        'scope': __doc__+' Includes planner, append/finalize, embedding, all decoder layers, final norm, LM head and argmax. Excludes prefill, capture, restore and scrub. Not online serving or final independent TEST.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contrast', choices=('reference_a0', 'fi_a0'), required=True)
    parser.add_argument('--pilot', type=Path, required=True)
    args = parser.parse_args()
    PILOT = args.pilot.resolve()
    pilot = json.loads((PILOT/'analysis.json').read_text())
    if pilot['status'] != 'complete' or not pilot['execution_passed']:
        raise ValueError('Complete execution-valid four-policy pilot required first')
    original = json.loads((PILOT/'manifest.json').read_text())
    import fcntl
    idle = base.idle_preflight(0); gpu = idle[-1]['uuid']
    with (Path('/tmp')/f'pagegauge-mlsys-{gpu}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        blocks = schedule(args.contrast)
        m = {**original, 'experiment': 'E3_regional_'+args.contrast+'_balanced_replication',
            'contrast': args.contrast, 'schedule': blocks, 'idle': idle,
            'pilot_analysis_sha256': base.sha256_file(PILOT/'analysis.json'),
            'statistics': {'bootstrap_samples': 50000, 'bootstrap_seed': 2026090912,
                'primary': 'cache_neutral.wall_ms', 'unit': 'two fixtures, paired processes; repeats nested within processes',
                'scope': 'Limited two-fixture generality; no unbounded repeats or outcome-based stopping'},
            'quality_policy': 'Retain descriptive HF metrics and gate flags; do not reinterpret them as independent TEST',
            'effective_worker_configurations': [configuration(b, original['inputs']['model_path']) if b['backend'] == 'page_gauge'
                else previous.worker_config(b, original['inputs']['model_path']) for b in blocks],
            'created_utc': datetime.now(timezone.utc).isoformat(), 'scope': __doc__}
        m['source_sha256'] = dict(original['source_sha256'])
        m['source_sha256'][str(Path(__file__).resolve().relative_to(ROOT))] = base.sha256_file(Path(__file__))
        shell = Path(__file__).with_name('run_clean_replication.sh')
        m['source_sha256'][str(shell.relative_to(ROOT))] = base.sha256_file(shell)
        m.pop('manifest_sha256', None); m['manifest_sha256'] = base.canonical_hash(m)
        previous.verify_frozen(m, full_inputs=True)
        out = ROOT/'results/mlsys2027_ablation_v1'/('regional_'+args.contrast+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
        out.mkdir(parents=True); base.atomic_json(out/'manifest.json', m)
        print('Regional replication: '+str(out), flush=True)
        os.environ['PAGEGAUGE_VALUE_CONDITIONING'] = 'none'
        os.environ['PAGEGAUGE_EXPERIMENT_EXACT_SPLIT'] = '32'
        payloads = []
        for block in blocks:
            previous.verify_frozen(m); base.idle_preflight(0)
            target = out/f"block_{block['index']}.json"
            command = previous.worker_command(block, m['inputs']['model_path'], target)
            if block['backend'] == 'page_gauge':
                command[2] = str(ROOT/ADAPTER)
                config = configuration(block, m['inputs']['model_path'])
                for flag, key in (('--exact-sink-pages','exact_sink_pages'), ('--exact-static-suffix-pages','exact_static_suffix_pages'),
                    ('--exact-tail','exact_tail_tokens'), ('--min-logits-cosine','min_logits_cosine'),
                    ('--candidate-split-pages','candidate_split_pages')):
                    command[command.index(flag)+1] = str(config[key])
            base.atomic_json(out/f"block_{block['index']}_invocation.json", {'command': command})
            c = wsl_gpu_monitor.run_process(previous, command, out, block, gpu)
            base.atomic_json(out/f"block_{block['index']}_completion.json", c)
            previous.require(c['sampled_exclusivity_passed'], 'Monitoring failure; timing ineligible')
            p = json.loads(target.read_text())
            check = assess(p, block, m, c) if block['backend'] == 'page_gauge' else previous.validate_worker(p, block, m, c['return_code'])
            previous.verify_frozen(m)
            base.atomic_json(out/f"block_{block['index']}_assessment.json", {**check, 'result_sha256': base.sha256_file(target)})
            payloads.append(p)
        result = reduce_results(payloads, blocks)
        result['contrast'] = args.contrast; result['manifest_sha256'] = m['manifest_sha256']
        base.atomic_json(out/'analysis.json', result)
        print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__': main()
