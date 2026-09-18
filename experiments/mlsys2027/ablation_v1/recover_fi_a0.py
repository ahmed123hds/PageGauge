"""CPU-only, separately attested repair of cross-backend timed-work comparison.

Preserves frozen launcher, measured JSON, assessments and original failure.
Only four backend-specific wrapper counters may differ; validate their exact
declared values instead of dropping them. All other work fields must match.
"""
import argparse
import json
import math
from pathlib import Path
import statistics
from datetime import datetime, timezone
from regional_replication import previous, base, assess

COUNTERS = {'wrapper_plan_invocations': 1536,
            'wrapper_last_page_len_device_fills': 1536,
            'flashinfer_full_plan_calls_all_wrappers': 96,
            'blocking_d2h_metadata_copies': 192}


def match_work(a, b):
    for p in (a, b):
        backend = p['backend']
        previous.require(backend in ('flashinfer_fp16', 'page_gauge'), 'Unknown backend')
        multiplier = 1 if backend == 'flashinfer_fp16' else 2
        for key, count in COUNTERS.items():
            previous.require(p['timed_work'][key] == multiplier*count, 'Wrong backend work '+key)
    previous.require({k:v for k,v in a['timed_work'].items() if k not in COUNTERS} ==
                     {k:v for k,v in b['timed_work'].items() if k not in COUNTERS},
                     'Different semantic timed work')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args(); run = args.run.resolve()
    m = json.loads((run/'manifest.json').read_text())
    previous.require(m['contrast'] == 'fi_a0', 'Only declared FI/A0 repair supported')
    previous.verify_frozen(m, full_inputs=True)
    blocks = m['schedule']; previous.require(len(blocks) == 8, 'Eight blocks required')
    payloads = []; evidence = {}; assessments = []
    for block in blocks:
        i = block['index']
        for suffix in ('.json', '_assessment.json', '_completion.json', '_telemetry.json', '.log', '_invocation.json'):
            path = run/f'block_{i}{suffix}'; evidence[path.name] = base.sha256_file(path)
        p = json.loads((run/f'block_{i}.json').read_text())
        c = json.loads((run/f'block_{i}_completion.json').read_text())
        a = json.loads((run/f'block_{i}_assessment.json').read_text())
        previous.require(a['result_sha256'] == evidence[f'block_{i}.json'], 'Result hash mismatch')
        previous.require(c['telemetry_sha256'] == evidence[f'block_{i}_telemetry.json'] and
                         c['log_sha256'] == evidence[f'block_{i}.log'], 'Monitoring hash mismatch')
        previous.require(c['return_code'] in (0,2) and c['own_pid_seen'] and c['sampled_exclusivity_passed'], 'Worker/ownership failed')
        if block['backend'] == 'page_gauge': check = assess(p, block, m, c)
        else: check = previous.validate_worker(p, block, m, c['return_code'])
        assessments.append({'block':i, 'return_code':c['return_code'], 'assessment':check})
        payloads.append(p)
    endpoints = {}
    for mode in previous.MODES:
        for metric in previous.METRICS:
            rows = []
            for i in range(0, 8, 2):
                a = next(j for j in (i,i+1) if blocks[j]['role'] == 'reference')
                b = next(j for j in (i,i+1) if blocks[j]['role'] == 'a0')
                for key in previous.MATCHED_FIELDS:
                    previous.require(payloads[a]['pairing']['configuration'][key] == payloads[b]['pairing']['configuration'][key], 'Different paired fixture '+key)
                match_work(payloads[a], payloads[b])
                means = [statistics.fmean(math.log(s[metric]) for s in payloads[j]['timing_modes'][mode]['raw_samples']) for j in (a,b)]
                rows.append({'seed': blocks[i]['seed'], 'pair_id': blocks[i]['pair_id'],
                    'pair_order': blocks[i]['pair_order'], 'log_speedup': means[0]-means[1],
                    'reference_ms_per_step': math.exp(means[0])/1536, 'a0_ms_per_step': math.exp(means[1])/1536})
            endpoints[mode+'.'+metric] = {'reference_over_a0': math.exp(statistics.fmean(r['log_speedup'] for r in rows)),
                **previous._hierarchical_bootstrap_ci(rows,50000,2026090912), 'pairs': rows}
    result = {'status':'complete', 'execution_passed':True, 'contrast':'fi_a0',
        'fresh_process_blocks':8, 'fixture_clusters':2, 'adjacent_pairs':4,
        'manifest_sha256':m['manifest_sha256'], 'endpoints':endpoints, 'block_assessments':assessments,
        'recovery': {'reason':'Original whole-dictionary equality rejects legitimate one-versus-two-wrapper costs.',
            'source_sha256':base.sha256_file(Path(__file__)), 'input_sha256':evidence,
            'created_utc':datetime.now(timezone.utc).isoformat(),
            'changed_measurements':False, 'original_launcher_unchanged':True},
        'scope':m['scope']+' Complete decoder only; prefill, capture, restore and scrub excluded. Two exposed TRAIN fixtures, not final TEST or serving.'}
    target = run/'recovered_analysis.json'
    previous.require(not target.exists(), 'Never overwrite a recovery artifact')
    base.atomic_json(target,result)
    print(json.dumps(result,indent=2))


if __name__ == '__main__': main()
