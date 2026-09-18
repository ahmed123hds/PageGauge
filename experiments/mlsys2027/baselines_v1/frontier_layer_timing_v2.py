"""Layer-segment timing pilot after validation; one process/arm, no CI."""
import argparse
import gc
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import traceback
import uuid
import frontier_combined_graphs as common
from frontier_combined_graphs import ROOT, base, previous, wsl_gpu_monitor, verify, PYTHON
import frontier_run
import attention_graph_dispatch as attention
import layer_segment_dispatch as segments
import segment_graph_dispatch
import mistral_dense_segments


def worker(out):
    m = json.loads((out/'manifest.json').read_text())
    original = frontier_run.restore
    originals, records = [], []
    def restore(model, *args, **kwargs):
        for layer, forward in originals:
            layer.forward = forward
        originals.clear()
        # Old per-layer forwards held the driver and its captured graphs alive
        # during the outer runner's GC. Collect only after releasing forwards.
        gc.collect()
        import torch
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        past = original(model, *args, **kwargs)
        stats = attention.prepare(past[0][0], m['context'], m['decode_steps'], False)
        saved, counters = segments.install(model, past[0][0], m['context'], m['decode_steps'], False)
        originals.extend(saved)
        records.append({'attention': stats, 'segments': counters})
        return past
    frontier_run.restore = restore
    try:
        frontier_run.worker(out)
        if len(records) != 4:
            raise ValueError('Missing validation round')
        for r in records:
            attention.verify_stats(r['attention'], 1536, 32, False)
            segments.verify(r['segments'], 1536, False)
        result = json.loads((out/'analysis.json').read_text())
        result['layer_segment_timing'] = records
        base.atomic_json(out/'analysis.json', result)
    finally:
        frontier_run.restore = original
        for layer, forward in originals:
            layer.forward = forward


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker', type=Path)
    p.add_argument('--fixtures', nargs=2, type=Path)
    args = p.parse_args()
    if args.worker:
        try:
            worker(args.worker)
        except BaseException as e:
            base.atomic_json(args.worker/'failure.json', {'type': type(e).__name__, 'error': str(e), 'traceback': traceback.format_exc()})
            raise
        return
    import fcntl
    for source in args.fixtures:
        m = json.loads((source/'manifest.json').read_text())
        c = json.loads((source/'completion.json').read_text())
        verify(m)
        if c['return_code'] or not c['sampled_exclusivity_passed'] or m['backend'] not in ('flashinfer_fp16', 'page_gauge'):
            raise ValueError('Invalid source')
        if base.sha256_file(source/'tokens.json') != m['tokens_sha256']:
            raise ValueError('Changed tokens')
        prior = json.loads((source/'analysis.json').read_text())
        if not m['validate'] or len(prior['layer_segment_validation']) != 1:
            raise ValueError('Completed segment validation required')
        for record in prior['layer_segment_validation']:
            attention.verify_stats(record['attention'], 1536, 32, True)
            segments.verify(record['segments'], 1536, True)
        if base.sha256_file(source/'quality.json') != prior['quality_sha256']:
            raise ValueError('Changed quality evidence')
        for name, key in [('block_0.log', 'log_sha256'), ('block_0_telemetry.json', 'telemetry_sha256')]:
            if base.sha256_file(source/name) != c[key]:
                raise ValueError('Changed monitoring')
        idle = base.idle_preflight(0); gpu = idle[-1]['uuid']
        with (Path('/tmp')/f'pagegauge-mlsys-{gpu}.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            out = ROOT/'results/mlsys2027_baselines_v1'/('layer_segment_timing_v2_'+m['backend']+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
            out.mkdir()
            (out/'tokens.json').write_bytes((source/'tokens.json').read_bytes())
            m.update(validate=False, repeats=3, idle=idle, source_fixture=str(source.resolve()), scope=__doc__+' Original reference policy and operations. Capture/restoration excluded; copies and guards included. Fixed FI then PG, warmup and three repeats; ranges only.')
            for module in (segments, segment_graph_dispatch, mistral_dense_segments):
                path = Path(module.__file__).resolve()
                m['source_sha256'][str(path)] = base.sha256_file(path)
            m['source_sha256'][str(Path(__file__).resolve())] = base.sha256_file(Path(__file__))
            base.atomic_json(out/'manifest.json', m)
            os.environ.pop('PYTORCH_CUDA_ALLOC_CONF', None)
            os.environ['PYTORCH_ALLOC_CONF'] = m['allocator_configuration']
            command = [PYTHON, '-u', str(Path(__file__).resolve()), '--worker', str(out)]
            base.atomic_json(out/'invocation.json', {'command': command})
            print('Layer-segment timing: '+str(out), flush=True)
            c = wsl_gpu_monitor.run_process(previous, command, out, {'index': 0}, gpu)
            base.atomic_json(out/'completion.json', c)
            verify(m)
            if c['return_code'] or not c['sampled_exclusivity_passed']:
                raise ValueError('Layer validation failed; stop and retain')


if __name__ == '__main__':
    main()
