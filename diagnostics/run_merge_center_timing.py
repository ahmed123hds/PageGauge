"""Matched three-arm development timing; one fresh process per arm, no CI."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'experiments/mlsys2027/baselines_v1'))
import frontier_layer_timing_v2 as timing


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--output', type=Path)
    p.add_argument('--worker', type=Path)
    args = p.parse_args()
    if args.worker:
        m = json.loads((args.worker/'manifest.json').read_text())
        if m['candidate_merge']:
            import benchmark_pg19_external_quality as quality
            from install_merge_center_candidate import install
            install(quality.PG.TransformerDecoder)
        timing.worker(args.worker)
        return
    out = args.output.resolve()
    if out.exists():
        raise ValueError('Output must be new')
    root = ROOT/'results/mlsys2027_baselines_v1'
    sources = [('flashinfer', root/'layer_segments_flashinfer_fp16_20260909T190907Z_7b024ad7', False),
               ('original_pg', root/'layer_segments_page_gauge_20260909T191435Z_c7175f1b', False),
               ('candidate_pg', root/'merge_center_layer_validation_v2', True)]
    prepared = []
    for name, source, candidate in sources:
        m = json.loads((source/'manifest.json').read_text())
        c = json.loads((source/'completion.json').read_text())
        timing.verify(m)
        if c['return_code'] or not c['sampled_exclusivity_passed'] or not m['validate']:
            raise ValueError('Successful monitored validation required')
        if timing.base.sha256_file(source/'tokens.json') != m['tokens_sha256']:
            raise ValueError('Changed tokens')
        prepared.append((name, source, candidate, m))
    if len({m['tokens_sha256'] for _, _, _, m in prepared}) != 1:
        raise ValueError('Unmatched inputs')
    out.mkdir(parents=True)
    timing.base.atomic_json(out/'plan.json', {'order': [v[0] for v in prepared], 'scope': __doc__})
    import fcntl
    for name, source, candidate, m in prepared:
        idle = timing.base.idle_preflight(0)
        with (Path('/tmp')/('pagegauge-mlsys-'+idle[-1]['uuid']+'.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            target = out/name
            target.mkdir()
            shutil.copyfile(source/'tokens.json', target/'tokens.json')
            for file in (Path(__file__), Path(timing.__file__),
                         Path(__file__).with_name('merge_center_candidate.py'),
                         Path(__file__).with_name('install_merge_center_candidate.py')):
                m['source_sha256'][str(file.resolve())] = timing.base.sha256_file(file)
            m.update(validate=False, repeats=3, candidate_merge=candidate, scope=__doc__, idle=idle)
            timing.base.atomic_json(target/'manifest.json', m)
            os.environ['PYTORCH_ALLOC_CONF'] = m['allocator_configuration']
            command = [timing.PYTHON, '-u', str(Path(__file__).resolve()), '--worker', str(target)]
            timing.base.atomic_json(target/'invocation.json', {'command': command})
            print('Timing arm '+name, flush=True)
            c = timing.wsl_gpu_monitor.run_process(timing.previous, command, target, {'index': 0}, idle[-1]['uuid'])
            timing.base.atomic_json(target/'completion.json', c)
            timing.verify(m)
            if c['return_code'] or not c['sampled_exclusivity_passed']:
                raise RuntimeError('Timing failed; retained')


if __name__ == '__main__':
    main()
