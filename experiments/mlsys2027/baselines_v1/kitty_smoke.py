"""Native Kitty SM120 recurrence versus independently reconstructed cache."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace
import uuid

ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path('/home/anonymous/pagegauge_baselines/Kitty')
PYTHON = '/home/anonymous/pagegauge_baselines/kitty_sm120_env/bin/python'
sys.path.insert(0, str(ROOT/'diagnostics'))
import mlsys_rtx5090_entry as base
import mlsys_rtx5090_step02 as previous


def verify(manifest):
    for name, digest in manifest['source_sha256'].items():
        if base.sha256_file(Path(name)) != digest:
            raise RuntimeError('Source changed: '+name)


def assess(out):
    completion = json.loads((out/'completion.json').read_text())
    result = json.loads((out/'analysis.json').read_text()) if (out/'analysis.json').exists() else {}
    telemetry = json.loads((out/'block_0_telemetry.json').read_text())
    observations = result.get('worker_boundary_observations', [])
    combined = (len(observations) == 4 and all(
        completion['pid'] in row['pids'] and not row.get('error') and not row['unexpected_pids']
        for row in observations) and all(not row.get('error') and not row.get('unexpected_pids')
        for row in telemetry['observations']))
    passed = completion['return_code'] == 0 and result.get('execution_passed', False) and combined
    base.atomic_json(out/'assessment.json', {'execution_passed': passed,
        'combined_sampled_exclusivity_passed': combined,
        'boundary_observation_count': len(observations),
        'periodic_observer_own_pid_seen': completion['own_pid_seen'],
        'input_sha256': {name: base.sha256_file(out/name) for name in
            ('completion.json', 'analysis.json', 'block_0_telemetry.json') if (out/name).exists()},
        'scope': 'Short worker boundary observations plus periodic checks; not continuous proof of exclusivity.'})
    return passed


def worker(out):
    import torch
    from kitty.kvcache import get_kvcache_kitty
    from kitty.kvcache.kernels.kitty_attention import kitty_attention_forward
    from kitty_reference import attention as oracle
    manifest = json.loads((out/'manifest.json').read_text())
    verify(manifest)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if torch.cuda.get_device_capability() != (12, 0):
        raise ValueError('SM120 check only')
    cfg = SimpleNamespace(num_hidden_layers=1, head_dim=128, num_key_value_heads=8)
    module = SimpleNamespace(num_attention_heads=32, num_key_value_heads=8)
    context, steps = manifest['context'], manifest['decode_steps']
    rows, boundary_observations = [], []
    for batch in manifest['batches']:
        print('Kitty native recurrence B='+str(batch), flush=True)
        generator = torch.Generator(device='cuda').manual_seed(manifest['seed']+batch)
        cache = get_kvcache_kitty(cfg, batch, context+steps)
        boundary_observations.append(previous.process_snapshot(os.environ['CUDA_VISIBLE_DEVICES'], os.getpid()))
        shape = (batch, 8, context+steps, 128)
        key = (torch.randn(shape, generator=generator, device='cuda', dtype=torch.float16)*.08).contiguous()
        value = (torch.randn(shape, generator=generator, device='cuda', dtype=torch.float16)*.08).contiguous()
        if cache.update(key[:, :, :context], value[:, :, :context], 0) is not True:
            raise RuntimeError('Missing native prefill')
        cache.quantize_prefill(0)
        initial_k, initial_v = cache.kv_cache[0].PageCount_K, cache.kv_cache[0].PageCount_V
        consumed_k, consumed_v, errors = set(), set(), []
        for step in range(steps):
            position = context+step
            if cache.update(key[:, :, position:position+1], value[:, :, position:position+1], 0):
                raise RuntimeError('Decode misclassified as prefill')
            layer = cache.kv_cache[0]
            if layer.get_total_length() != position+1:
                raise RuntimeError('Native K/V length mismatch')
            query = (torch.randn((batch, 32, 1, 128), generator=generator, device='cuda', dtype=torch.float16)*.35).contiguous()
            output, _ = kitty_attention_forward(module, query, layer, 128**-.5)
            consumed_k.add(layer.PageCount_K)
            consumed_v.add(layer.PageCount_V)
            if step in manifest['reference_steps']:
                expected = oracle(query, layer)
                left, right = output.float(), expected.float()
                if not bool(left.isfinite().all() & right.isfinite().all()):
                    raise RuntimeError('Nonfinite attention output')
                relative = float(torch.linalg.vector_norm(left-right)/torch.linalg.vector_norm(right).clamp_min(1e-12))
                maximum = float((left-right).abs().max())
                row = {'step': step, 'relative_l2': relative, 'max_abs': maximum,
                    'key_packed_pages': layer.PageCount_K, 'value_packed_pages': layer.PageCount_V}
                errors.append(row)
                if relative > manifest['max_relative_l2'] or maximum > manifest['max_abs']:
                    base.atomic_json(out/f'failed_B{batch}.json', {'errors': errors})
                    raise RuntimeError('Native Kitty attention differs from reconstructed-cache reference')
            cache.quantize_decode(0)
            if cache.get_seq_length() != position+1:
                raise RuntimeError('Finalization changed logical length')
        if max(consumed_k) <= initial_k or max(consumed_v) <= initial_v:
            raise RuntimeError('New packed K/V were not consumed')
        rows.append({'batch': batch, 'errors': errors, 'final_length': cache.get_seq_length(),
            'consumed_key_page_counts': sorted(consumed_k), 'consumed_value_page_counts': sorted(consumed_v),
            'final_key_pages': layer.PageCount_K, 'final_value_pages': layer.PageCount_V})
        torch.cuda.synchronize()
        boundary_observations.append(previous.process_snapshot(os.environ['CUDA_VISIBLE_DEVICES'], os.getpid()))
        del cache, layer, key, value, query, output, expected
        torch.cuda.empty_cache()
    verify(manifest)
    boundary_pass = len(boundary_observations) == 4 and all(
        os.getpid() in row['pids'] and not row['unexpected_pids'] for row in boundary_observations)
    if not boundary_pass:
        raise RuntimeError('GPU boundary exclusivity failed')
    base.atomic_json(out/'analysis.json', {'execution_passed': True, 'rows': rows,
        'worker_boundary_observations': boundary_observations, 'worker_boundary_exclusivity_passed': boundary_pass,
        'scope': manifest['scope'], 'torch': torch.__version__, 'gpu': torch.cuda.get_device_name()})
    print('Kitty smoke complete: '+str(out), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--assess', type=Path, help='Reduce preserved short-run observations without rerunning GPU work')
    args = parser.parse_args()
    if args.assess:
        if not assess(args.assess):
            raise SystemExit(2)
        return
    if args.worker:
        try:
            worker(args.worker)
        except BaseException as exc:
            base.atomic_json(args.worker/'failure.json', {'error': str(exc), 'traceback': traceback.format_exc()})
            raise
        return
    import fcntl
    idle = base.idle_preflight(0)
    gpu = idle[-1]['uuid']
    with (Path('/tmp')/f'pagegauge-mlsys-{gpu}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        paths = set((SOURCE/'src/kitty').rglob('*.py'))
        paths.update((Path(__file__), Path(__file__).with_name('kitty_reference.py')))
        out = ROOT/'results/mlsys2027_baselines_v1'/('kitty_smoke_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
        out.mkdir(parents=True)
        manifest = {'context': 511, 'decode_steps': 257, 'batches': [1, 4], 'seed': 2026090823,
            'reference_steps': [0, 31, 32, 95, 96, 127, 128, 159, 160, 255, 256],
            'max_relative_l2': .005, 'max_abs': .02,
            'source_sha256': {str(p): base.sha256_file(p) for p in sorted(paths)}, 'idle': idle,
            'scope': 'Native Kitty K2V2+25%K4, sink32/page128 recurrence against same reconstructed quantized cache. Numerical execution check only; no original-FP16 predictive quality or speed claim.'}
        base.atomic_json(out/'manifest.json', manifest)
        print('Kitty smoke output: '+str(out), flush=True)
        command = [PYTHON, '-u', str(Path(__file__)), '--worker', str(out)]
        base.atomic_json(out/'invocation.json', {'command': command})
        completion = previous.run_process(command, out, {'index': 0}, gpu)
        base.atomic_json(out/'completion.json', completion)
        verify(manifest)
        if not assess(out):
            raise RuntimeError('Native Kitty smoke/exclusivity failed; inspect preserved files')


if __name__ == '__main__':
    main()
