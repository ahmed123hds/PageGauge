"""Fixed B4/B8/B16 common-engine grid; lower bounds are not measured OOMs."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'diagnostics'))
import mlsys_rtx5090_entry as base
from kivi_quality import verify


def parameter_bytes(model):
    """Read safetensors shapes, not tensor data; all model arms load FP16."""
    config = json.loads((model/'config.json').read_text())
    if config.get('tie_word_embeddings') or config['model_type'] != 'mistral':
        raise ValueError('Only the pinned untied Mistral parameter layout is audited')
    index = json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
    tensors = {}
    for shard in sorted(set(index.values())):
        with (model/shard).open('rb') as stream:
            count = struct.unpack('<Q', stream.read(8))[0]
            if count > 100_000_000:
                raise ValueError('Invalid safetensors header size')
            header = json.loads(stream.read(count))
        for name, item in header.items():
            if name == '__metadata__':
                continue
            if name in tensors or index.get(name) != shard or item['dtype'] not in ('BF16', 'F16', 'F32'):
                raise ValueError('Unexpected checkpoint tensor/index')
            tensors[name] = math.prod(item['shape'])*2
    if set(tensors) != set(index):
        raise ValueError('Incomplete checkpoint headers')
    # Independently check the expected architecture, including both untied
    # embedding matrices and RMSNorm weights. No workspace is counted.
    h, n, kv, d, inter, vocab = (config[k] for k in
        ('hidden_size', 'num_hidden_layers', 'num_key_value_heads', 'head_dim', 'intermediate_size', 'vocab_size')) if 'head_dim' in config else (
        config['hidden_size'], config['num_hidden_layers'], config['num_key_value_heads'],
        config['hidden_size']//config['num_attention_heads'], config['intermediate_size'], config['vocab_size'])
    expected = 2*(2*vocab*h+n*(2*h*h+2*h*kv*d+3*h*inter+2*h)+h)
    if sum(tensors.values()) != expected:
        raise ValueError('Parameter shape count differs from audited architecture')
    return {'fp16_parameter_bytes': expected, 'tensor_count': len(tensors), 'layers': n, 'kv_heads': kv, 'head_dim': d}


def lower_bound(weights, batch, bits, layers=32, heads=8, dim=128, length=22016):
    # All five policies retain every position. Treat even exact FP16 regions
    # as the minimum bitwidth, and omit scales/centers/staging/temporary storage.
    code_bytes = batch*layers*2*heads*dim*length*bits//8
    return weights+code_bytes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot', type=Path)
    p.add_argument('--plan', type=Path)
    args = p.parse_args()
    if bool(args.pilot) == bool(args.plan):
        raise ValueError('Provide completed B4 pilot to freeze a plan, OR frozen plan to execute')
    if args.pilot:
        pilot = args.pilot.resolve()
        result = json.loads((pilot/'analysis.json').read_text())
        order = ('flashinfer_fp16', 'page_gauge', 'kivi_int4', 'bitdecode_int4', 'kivi_int2')
        if set(result['rows']) != set(order) or result['batch'] != 4:
            raise ValueError('Complete five-method B4 pilot required')
        evidence, first = {}, None
        for backend in order:
            path = Path(result['rows'][backend]['run'])
            m = json.loads((path/'manifest.json').read_text())
            c = json.loads((path/'completion.json').read_text())
            if c['return_code'] or not c['sampled_exclusivity_passed'] or base.sha256_file(path/'analysis.json') != result['rows'][backend]['analysis_sha256']:
                raise ValueError('Invalid original pilot evidence')
            verify(m)
            first = first or m
            if any(m[k] != first[k] for k in ('source_sha256', 'model', 'tokens_sha256', 'allocator_budget_bytes')):
                raise ValueError('Mismatched pilot inputs')
            evidence[str(path/'analysis.json')] = base.sha256_file(path/'analysis.json')
        headers = parameter_bytes(Path(first['model']))
        cells = []
        for batch in (8, 16):
            for backend in order:
                bits = 16 if backend == 'flashinfer_fp16' else 8 if backend == 'page_gauge' else 2 if backend == 'kivi_int2' else 4
                bound = lower_bound(headers['fp16_parameter_bytes'], batch, bits,
                    headers['layers'], headers['kv_heads'], headers['head_dim'])
                cells.append({'backend': backend, 'batch': batch, 'lower_bound_bytes': bound,
                    'action': 'analytically_infeasible' if bound > first['allocator_budget_bytes'] else 'run_full_recurrence'})
        out = ROOT/'results/mlsys2027_baselines_v1'/('capacity_grid_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
        out.mkdir(parents=True)
        hashes = dict(first['source_sha256'])
        for path in (Path(__file__), Path(__file__).with_name('CAPACITY_PROTOCOL.md')):
            hashes[str(path.resolve())] = base.sha256_file(path)
        plan = {'cells': cells, 'original_pilot': str(pilot), 'original_pilot_sha256': base.sha256_file(pilot/'analysis.json'),
            'input_evidence': evidence, 'parameter_header_accounting': headers, 'source_sha256': hashes,
            'input_file_evidence': first['input_file_evidence'], 'model': first['model'],
            'allocator_budget_bytes': first['allocator_budget_bytes'], 'context': 20480, 'decode_steps': 1536,
            'offset': 472000, 'repeats': 1,
            'scope': 'Fixed B4/B8/B16 TRAIN common-eager grid; reuse B4, full warmup plus one timed recurrence for new feasible points. No max-batch, replicated throughput CI, native-best or final TEST claim.'}
        base.atomic_json(out/'manifest.json', plan)
        print('Capacity plan: '+str(out), flush=True)
        print(json.dumps({'parameters': headers, 'cells': cells}, indent=2), flush=True)
        return
    out = args.plan.resolve()
    m = json.loads((out/'manifest.json').read_text())
    if (out/'progress.json').exists() or (out/'analysis.json').exists():
        raise ValueError('Plan already started; inspect retained progress, do not duplicate')
    rows = []
    for index, cell in enumerate(m['cells']):
        verify(m)
        if cell['action'] == 'analytically_infeasible':
            rows.append(dict(cell, outcome='analytically_infeasible'))
            continue
        command = [sys.executable, '-u', str(Path(__file__).with_name('frontier_run.py')),
            '--backend', cell['backend'], '--batch', str(cell['batch']), '--context', str(m['context']),
            '--steps', str(m['decode_steps']), '--repeats', str(m['repeats']), '--offset', str(m['offset'])]
        target = None
        with (out/f'cell_{index}.log').open('x') as log:
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
        record = dict(cell, run=str(target), return_code=code)
        if target is None or not (target/'completion.json').exists():
            base.atomic_json(out/'failure.json', record)
            raise ValueError('Host/preflight failure, not a measured capacity OOM')
        complete = json.loads((target/'completion.json').read_text())
        if not complete['sampled_exclusivity_passed']:
            base.atomic_json(out/'failure.json', record)
            raise ValueError('Exclusivity failure, not capacity evidence')
        if code:
            failure = json.loads((target/'failure.json').read_text()) if (target/'failure.json').exists() else {}
            record['failure'] = failure
            if failure.get('type') != 'OutOfMemoryError' or 'CUDA out of memory' not in failure.get('error', ''):
                base.atomic_json(out/'failure.json', record)
                raise ValueError('Non-CUDA-OOM failure; diagnose before continuing')
            record['outcome'] = 'measured_cuda_oom'
        else:
            r = json.loads((target/'analysis.json').read_text())
            if r['validation_only'] or r['warmup']['steps'] != 1536 or len(r['rows']) != 1 or r['rows'][0]['steps'] != 1536:
                raise ValueError('Incomplete capacity trajectory')
            record.update(outcome='verified_feasible', analysis_sha256=base.sha256_file(target/'analysis.json'),
                wall_ms_per_step=r['rows'][0]['wall_ms_per_step'],
                aggregate_tokens_per_second=r['rows'][0]['aggregate_tokens_per_second'],
                peak_allocated_bytes=r['rows'][0]['final_memory']['peak_allocated_bytes'])
        rows.append(record)
        base.atomic_json(out/'progress.json', {'rows': rows})
    verify(m)
    base.atomic_json(out/'analysis.json', {'rows': rows, 'original_pilot': m['original_pilot'], 'scope': m['scope']})
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == '__main__':
    main()
