"""Separate monitored B4 layer-graph development validation of fused merge."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'experiments/mlsys2027/baselines_v1'))
import frontier_layer_segments as validation


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--source', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--worker', type=Path)
    args = p.parse_args()
    if args.worker:
        import benchmark_pg19_external_quality as quality
        from install_merge_center_candidate import install
        install(quality.PG.TransformerDecoder)
        validation.worker(args.worker)
        return
    source, out = args.source.resolve(), args.output.resolve()
    m = json.loads((source/'manifest.json').read_text())
    validation.verify(m)
    if m['backend'] != 'page_gauge' or not m['validate']:
        raise ValueError('Retained PG validation fixture required')
    if validation.base.sha256_file(source/'tokens.json') != m['tokens_sha256']:
        raise ValueError('Changed tokens')
    if out.exists():
        raise ValueError('Output must be new')
    idle = validation.base.idle_preflight(0)
    import fcntl
    with (Path('/tmp')/('pagegauge-mlsys-'+idle[-1]['uuid']+'.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        out.mkdir(parents=True)
        shutil.copyfile(source/'tokens.json', out/'tokens.json')
        for name in ('run_merge_center_layer_validation.py', 'merge_center_candidate.py', 'install_merge_center_candidate.py'):
            file = Path(__file__).with_name(name).resolve()
            m['source_sha256'][str(file)] = validation.base.sha256_file(file)
        m.update(scope=__doc__, source_fixture=str(source), idle=idle)
        validation.base.atomic_json(out/'manifest.json', m)
        os.environ['PYTORCH_ALLOC_CONF'] = m['allocator_configuration']
        command = [validation.PYTHON, '-u', str(Path(__file__).resolve()), '--worker', str(out)]
        validation.base.atomic_json(out/'invocation.json', {'command': command})
        result = validation.wsl_gpu_monitor.run_process(validation.previous, command, out, {'index': 0}, idle[-1]['uuid'])
        validation.base.atomic_json(out/'completion.json', result)
        validation.verify(m)
        if result['return_code'] or not result['sampled_exclusivity_passed']:
            raise RuntimeError('Candidate validation failed; evidence retained')


if __name__ == '__main__':
    main()
