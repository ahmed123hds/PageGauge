"""Execute the predeclared eight layer-segment timing processes, no retuning."""
import argparse
import json
import os
from pathlib import Path
import statistics
import frontier_layer_timing_v2 as timing
from frontier_layer_timing_v2 import base, previous, wsl_gpu_monitor, verify, PYTHON
import attention_graph_dispatch as attention
import layer_segment_dispatch as segments


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    out = p.parse_args().directory.resolve()
    m = json.loads((out/'manifest.json').read_text())
    verify(m)
    assert json.loads((out/'qualification.json').read_text())['second_fixture_execution_complete']
    if (out/'execution_manifest.json').exists():
        raise ValueError('Already started; never overwrite or silently restart')
    sources = {}
    hashes = {}
    for f in (0, 1):
        for b in ('flashinfer_fp16', 'page_gauge'):
            d = Path(m['fixture0_validations'][0 if b == 'flashinfer_fp16' else 1]) if f == 0 else out/('validation1_'+b)
            cm = json.loads((d/'manifest.json').read_text())
            c = json.loads((d/'completion.json').read_text())
            r = json.loads((d/'analysis.json').read_text())
            verify(cm)
            assert c['return_code'] == 0 and c['own_pid_seen'] and c['sampled_exclusivity_passed']
            assert cm['backend'] == b and cm['validate']
            assert base.sha256_file(d/'tokens.json') == cm['tokens_sha256']
            assert base.sha256_file(d/'quality.json') == r['quality_sha256']
            for name, key in [('block_0.log', 'log_sha256'), ('block_0_telemetry.json', 'telemetry_sha256')]:
                assert base.sha256_file(d/name) == c[key]
            assert len(r['layer_segment_validation']) == 1
            attention.verify_stats(r['layer_segment_validation'][0]['attention'], 1536, 32, True)
            segments.verify(r['layer_segment_validation'][0]['segments'], 1536, True)
            sources[f, b] = (d, cm)
            for name in ('manifest.json', 'analysis.json', 'completion.json', 'tokens.json', 'quality.json'):
                hashes[str(d/name)] = base.sha256_file(d/name)
    for f in (0, 1):
        assert sources[f, 'flashinfer_fp16'][1]['tokens_sha256'] == sources[f, 'page_gauge'][1]['tokens_sha256']
    base.atomic_json(out/'execution_manifest.json', {'parent_sha256': base.sha256_file(out/'manifest.json'),
        'validation_sha256': hashes, 'launcher_sha256': base.sha256_file(Path(__file__)), 'order': m['order']})
    print('Layer replication timing: '+str(out), flush=True)
    import fcntl
    rows = []
    for index, cell in enumerate(m['order']):
        verify(m)
        source, cm = sources[cell['fixture'], cell['backend']]
        cm = dict(cm)
        idle = base.idle_preflight(0); gpu = idle[-1]['uuid']
        with (Path('/tmp')/f'pagegauge-mlsys-{gpu}.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            target = out/f'block_{index}'; target.mkdir()
            (target/'tokens.json').write_bytes((source/'tokens.json').read_bytes())
            cm.update(validate=False, repeats=3, idle=idle, source_sha256=m['source_sha256'], scope=m['scope'])
            base.atomic_json(target/'manifest.json', cm)
            os.environ.pop('PYTORCH_CUDA_ALLOC_CONF', None)
            os.environ['PYTORCH_ALLOC_CONF'] = cm['allocator_configuration']
            command = [PYTHON, '-u', timing.__file__, '--worker', str(target)]
            base.atomic_json(target/'invocation.json', {'command': command})
            print('Replication block '+str(index)+' '+cell['backend'], flush=True)
            c = wsl_gpu_monitor.run_process(previous, command, target, {'index': 0}, gpu)
            base.atomic_json(target/'completion.json', c)
            verify(m)
            if c['return_code'] or not c['sampled_exclusivity_passed']:
                raise ValueError('Replication worker failed; retain and stop')
            r = json.loads((target/'analysis.json').read_text())
            assert len(r['rows']) == 3 and len(r['layer_segment_timing']) == 4
            for stats in r['layer_segment_timing']:
                attention.verify_stats(stats['attention'], 1536, 32, False)
                segments.verify(stats['segments'], 1536, False)
            rows.append({**cell, 'index': index, 'median_ms': statistics.median(v['wall_ms_per_step'] for v in r['rows']),
                         'analysis_sha256': base.sha256_file(target/'analysis.json')})
            base.atomic_json(out/'progress.json', {'rows': rows})
    base.atomic_json(out/'timing_complete.json', {'rows': rows, 'execution_complete': True})


if __name__ == '__main__':
    main()
