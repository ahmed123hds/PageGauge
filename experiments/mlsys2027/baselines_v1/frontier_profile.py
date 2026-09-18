"""Instrument the last129 decode steps of an existing CPU-backed B4 fixture.

An isolated worker installs forward hooks after cache restoration. Native math
and the frozen timing runner stay unchanged on disk. Instrumented timings are
diagnostic only and must never enter the uninstrumented timing pilot.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'diagnostics'))
import mlsys_rtx5090_entry as base
import mlsys_rtx5090_step02 as previous
import frontier_run
from kivi_quality import verify, PYTHON


def worker(out):
    import torch
    manifest = json.loads((out/'manifest.json').read_text())
    if not manifest['profile_only'] or manifest['repeats'] != 0 or manifest['validate']:
        raise ValueError('Single instrumented recurrence required')
    original_restore = frontier_run.restore
    counts = {'steps': 0, 'starts': 0, 'stops': 0, 'restores': 0}
    profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA])
    handles = []

    def before(model, arguments):
        if counts['steps'] == manifest['decode_steps']-129:
            torch.cuda.synchronize()
            profiler.start()
            counts['starts'] += 1

    def after(model, arguments, output):
        counts['steps'] += 1
        if counts['steps'] == manifest['decode_steps']:
            torch.cuda.synchronize()
            profiler.stop()
            counts['stops'] += 1

    def instrumented_restore(model, *args, **kwargs):
        counts['restores'] += 1
        if counts['restores'] != 1:
            raise ValueError('Unexpected second recurrence in profile worker')
        past = original_restore(model, *args, **kwargs)
        handles.extend((model.register_forward_pre_hook(before), model.register_forward_hook(after)))
        return past

    frontier_run.restore = instrumented_restore
    try:
        frontier_run.worker(out)
    finally:
        frontier_run.restore = original_restore
        for handle in handles:
            handle.remove()
    if counts != {'steps': manifest['decode_steps'], 'starts': 1, 'stops': 1, 'restores': 1}:
        raise ValueError('Incorrect profiler boundaries')
    events = [{'name': event.key, 'device_type': str(event.device_type), 'count': event.count,
        'self_cpu_us': event.self_cpu_time_total, 'total_cpu_us': event.cpu_time_total,
        'self_device_us': event.self_device_time_total, 'total_device_us': event.device_time_total}
        for event in profiler.key_averages()]
    profiler.export_chrome_trace(str(out/'trace.json'))
    base.atomic_json(out/'profile.json', {'backend': manifest['backend'], 'batch': manifest['batch'],
        'profiled_steps': 129, 'counts': counts, 'events': events,
        'model_position_start': manifest['context']+manifest['decode_steps']-129,
        'scope': manifest['scope']})
    result = json.loads((out/'analysis.json').read_text())
    result.update(profile_only=True, profile_sha256=base.sha256_file(out/'profile.json'),
                  trace_sha256=base.sha256_file(out/'trace.json'))
    base.atomic_json(out/'analysis.json', result)
    verify(manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--fixture', type=Path)
    args = parser.parse_args()
    if args.worker:
        try:
            worker(args.worker)
        except BaseException as error:
            base.atomic_json(args.worker/'failure.json', {'type': type(error).__name__, 'error': str(error), 'traceback': traceback.format_exc()})
            raise
        return
    if args.fixture is None:
        raise ValueError('Completed timing fixture required')
    source = args.fixture.resolve()
    m = json.loads((source/'manifest.json').read_text())
    completion = json.loads((source/'completion.json').read_text())
    if completion['return_code'] or not completion['sampled_exclusivity_passed'] or m['validate']:
        raise ValueError('Invalid timed source fixture')
    if (m['batch'], m['context'], m['decode_steps']) != (4, 20480, 1536):
        raise ValueError('This diagnostic is for the completed B4 negative/weak case')
    verify(m)
    if base.sha256_file(source/'tokens.json') != m['tokens_sha256']:
        raise ValueError('Changed source tokens')
    import fcntl
    idle = base.idle_preflight(0)
    gpu = idle[-1]['uuid']
    with (Path('/tmp')/f'pagegauge-mlsys-{gpu}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        out = ROOT/'results/mlsys2027_baselines_v1'/('frontier_profile_'+m['backend']+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
        out.mkdir(parents=True)
        # Preserve the exact bytes, including canonical JSON whitespace/hash.
        (out/'tokens.json').write_bytes((source/'tokens.json').read_bytes())
        m.update(repeats=0, profile_only=True, source_fixture=str(source),
            source_analysis_sha256=base.sha256_file(source/'analysis.json'),
            idle=idle, orchestrator_pid=os.getpid(),
            scope='Instrumented last129 steps of a full1536-step B4/C20480 recurrence, with original CPU-backed native cache preparation and eager model body. Forward hooks only start/stop the profiler; no native math change. Diagnostic kernel/CPU attribution, not a latency/speed or quality acceptance result.')
        m['source_sha256'][str(Path(__file__).resolve())] = base.sha256_file(Path(__file__))
        base.atomic_json(out/'manifest.json', m)
        print('Frontier profile output: '+str(out), flush=True)
        command = [PYTHON, '-u', str(Path(__file__)), '--worker', str(out)]
        base.atomic_json(out/'invocation.json', {'command': command})
        result = previous.run_process(command, out, {'index': 0}, gpu)
        base.atomic_json(out/'completion.json', result)
        verify(m)
        if result['return_code'] or not result['sampled_exclusivity_passed']:
            raise ValueError('Profile execution/exclusivity failed; preserve evidence')


if __name__ == '__main__':
    main()
