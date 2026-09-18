"""Profile actual MLP-graph dispatch; instrumented times are not speed evidence."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import traceback
import uuid
import frontier_dense_timing as timing
from frontier_dense_timing import ROOT, base, previous, wsl_gpu_monitor, verify, PYTHON
import frontier_run


def worker(out):
    import torch
    m = json.loads((out/'manifest.json').read_text())
    original = frontier_run.restore
    counts = {'steps': 0, 'starts': 0, 'stops': 0, 'restores': 0}
    handles = []
    profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA])

    def before(module, args):
        if counts['steps'] == m['decode_steps']-129:
            torch.cuda.synchronize()
            profiler.start()
            counts['starts'] += 1

    def after(module, args, output):
        counts['steps'] += 1
        if counts['steps'] == m['decode_steps']:
            torch.cuda.synchronize()
            profiler.stop()
            counts['stops'] += 1

    def restore(model, *args, **kwargs):
        past = original(model, *args, **kwargs)
        counts['restores'] += 1
        handles.extend([model.register_forward_pre_hook(before), model.register_forward_hook(after)])
        return past

    frontier_run.restore = restore
    try:
        timing.worker(out)
    finally:
        frontier_run.restore = original
        for handle in handles:
            handle.remove()
    if counts != {'steps': 1536, 'starts': 1, 'stops': 1, 'restores': 1}:
        raise ValueError('Incomplete profiler trajectory')
    events = [{'name': e.key, 'device_type': str(e.device_type), 'count': e.count,
               'self_cpu_us': e.self_cpu_time_total, 'total_cpu_us': e.cpu_time_total,
               'self_device_us': e.self_device_time_total, 'total_device_us': e.device_time_total}
              for e in profiler.key_averages()]
    profiler.export_chrome_trace(str(out/'trace.json'))
    base.atomic_json(out/'profile.json', {'backend': m['backend'], 'batch': m['batch'],
        'profiled_steps': 129, 'counts': counts, 'events': events, 'scope': __doc__})
    r = json.loads((out/'analysis.json').read_text())
    r.update(profile_only=True, profile_sha256=base.sha256_file(out/'profile.json'),
             trace_sha256=base.sha256_file(out/'trace.json'))
    base.atomic_json(out/'analysis.json', r)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker', type=Path)
    p.add_argument('--pair', type=Path)
    args = p.parse_args()
    if args.worker:
        try:
            worker(args.worker)
        except BaseException as e:
            base.atomic_json(args.worker/'failure.json', {'type': type(e).__name__, 'error': str(e), 'traceback': traceback.format_exc()})
            raise
        return
    import fcntl
    pair = json.loads((args.pair/'analysis.json').read_text())
    for backend in ('flashinfer_fp16', 'page_gauge'):
        source = args.pair/backend
        m = json.loads((source/'manifest.json').read_text())
        c = json.loads((source/'completion.json').read_text())
        verify(m)
        if c['return_code'] or not c['sampled_exclusivity_passed']:
            raise ValueError('Invalid source timing')
        if base.sha256_file(source/'analysis.json') != pair['rows'][backend]['analysis_sha256']:
            raise ValueError('Changed source result')
        if base.sha256_file(source/'tokens.json') != m['tokens_sha256']:
            raise ValueError('Changed tokens')
        idle = base.idle_preflight(0); gpu = idle[-1]['uuid']
        with (Path('/tmp')/f'pagegauge-mlsys-{gpu}.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            out = ROOT/'results/mlsys2027_baselines_v1'/('dense_profile_'+backend+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
            out.mkdir()
            (out/'tokens.json').write_bytes((source/'tokens.json').read_bytes())
            m.update(repeats=0, profile_only=True, idle=idle, source_timing=str(source.resolve()), scope=__doc__)
            m['source_sha256'][str(Path(__file__).resolve())] = base.sha256_file(Path(__file__))
            base.atomic_json(out/'manifest.json', m)
            os.environ.pop('PYTORCH_CUDA_ALLOC_CONF', None)
            os.environ['PYTORCH_ALLOC_CONF'] = m['allocator_configuration']
            command = [PYTHON, '-u', str(Path(__file__).resolve()), '--worker', str(out)]
            base.atomic_json(out/'invocation.json', {'command': command})
            print('Dense profile: '+str(out), flush=True)
            c = wsl_gpu_monitor.run_process(previous, command, out, {'index': 0}, gpu)
            base.atomic_json(out/'completion.json', c)
            verify(m)
            if c['return_code'] or not c['sampled_exclusivity_passed']:
                raise ValueError('Profile failed; stop and retain')


if __name__ == '__main__':
    main()
