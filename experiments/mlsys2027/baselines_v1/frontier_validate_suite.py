"""Finish native B4 CPU-state validation after a retained FI control pilot."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'diagnostics'))
import mlsys_rtx5090_entry as base


def assess(directory, tokens_hash):
    m = json.loads((directory/'manifest.json').read_text())
    r = json.loads((directory/'analysis.json').read_text())
    c = json.loads((directory/'completion.json').read_text())
    if c['return_code'] or not c['sampled_exclusivity_passed']:
        raise ValueError('Invalid native validation completion')
    if (m['batch'], m['context'], m['decode_steps']) != (4, 4096, 785) or not m['validate']:
        raise ValueError('Wrong pilot shape/purpose')
    if base.sha256_file(directory/'tokens.json') != tokens_hash or m['tokens_sha256'] != tokens_hash:
        raise ValueError('Unmatched actual validation IDs')
    if not r['validation_only'] or len(r['rows']) != 1 or not r['rows'][0]['initial_state_validation']['bitwise_initial_state_match']:
        raise ValueError('Initial cache restoration check missing')
    q = json.loads((directory/'quality.json').read_text())
    if base.sha256_file(directory/'quality.json') != r['quality_sha256']:
        raise ValueError('Changed quality evidence')
    overall = q['distribution_quality']['overall']
    if overall['token_count'] != 3140:
        raise ValueError('Incomplete batched labels')
    return {'run': str(directory), 'analysis_sha256': base.sha256_file(directory/'analysis.json'),
        'quality_sha256': r['quality_sha256'], 'initial_state': r['rows'][0]['initial_state_validation'],
        'ppl_ratio_to_serial_hf': overall['candidate_to_reference_perplexity_ratio'],
        'top1_agreement_to_serial_hf': q['top1_agreement_fraction'],
        'minimum_cosine': q['minimum_logits_cosine'], 'cache': r['rows'][0]['cache']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', type=Path, required=True)
    args = parser.parse_args()
    pilot = args.pilot.resolve()
    m = json.loads((pilot/'manifest.json').read_text())
    if m['backend'] != 'flashinfer_fp16':
        raise ValueError('FI control pilot required')
    rows = {'flashinfer_fp16': assess(pilot, m['tokens_sha256'])}
    hashes = dict(m['source_sha256'])
    hashes[str(Path(__file__).resolve())] = base.sha256_file(Path(__file__))
    order = ['page_gauge', 'kivi_int4', 'bitdecode_int4', 'kivi_int2']
    out = ROOT/'results/mlsys2027_baselines_v1'/('frontier_validation_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True)
    base.atomic_json(out/'manifest.json', {'pilot': str(pilot), 'order': order, 'source_sha256': hashes,
        'orchestrator_pid': os.getpid(), 'tokens_sha256': m['tokens_sha256'],
        'scope': 'B4/C4096/D785 CPU-backed serving-state validation. Initial state bitwise checks plus descriptive HF fidelity, not a latency or predictive-quality acceptance result.'})
    print('Frontier validation suite: '+str(out), flush=True)
    for index, backend in enumerate(order):
        for path, digest in hashes.items():
            if base.sha256_file(Path(path)) != digest:
                raise ValueError('Source drift')
        command = [sys.executable, '-u', str(Path(__file__).with_name('frontier_run.py')),
                   '--backend', backend, '--batch', '4', '--context', '4096', '--steps', '785', '--validate']
        target = None
        with (out/f'{index}_{backend}.log').open('x') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            try:
                for line in process.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                    log.flush()
                    if line.startswith('Frontier output: '):
                        target = Path(line.strip().split(': ', 1)[1])
                code = process.wait()
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait()
        base.atomic_json(out/f'{index}_{backend}_completion.json', {'run': str(target), 'return_code': code, 'command': command})
        if code or target is None:
            raise ValueError('Validation worker failed; preserve and diagnose before timing')
        rows[backend] = assess(target, m['tokens_sha256'])
        base.atomic_json(out/'progress.json', {'rows': rows})
    for path, digest in hashes.items():
        if base.sha256_file(Path(path)) != digest:
            raise ValueError('Source drift')
    base.atomic_json(out/'analysis.json', {'rows': rows, 'initial_state_validation_passed': True,
        'scope': 'Five matched B4 native cache restorations and full785-step recurrences. Quality metrics are descriptive relative to serial HF; no performance claim.'})
    print(json.dumps({k: {f: r[f] for f in ('ppl_ratio_to_serial_hf', 'top1_agreement_to_serial_hf')} for k, r in rows.items()}, indent=2), flush=True)


if __name__ == '__main__':
    main()
