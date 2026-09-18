"""Qualify remaining low-bit arms sequentially in separate native processes."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import traceback
import uuid
from synthetic_generation import ROOT, base, previous, wsl_gpu_monitor, verify


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--worker', type=Path)
    args = parser.parse_args()
    if args.worker:
        try:
            from native_task_worker import worker
            worker(args.worker)
        except BaseException:
            base.atomic_json(args.worker/'failure.json', {'traceback': traceback.format_exc()})
            raise
        return
    import fcntl
    source = ROOT/'results/mlsys2027_tasks_v1/lowbit_smoke_20260910T092504Z_045c7533'
    for backend, bits in (('hf', 0), ('kivi', 4), ('bitdecode', 4)):
        m = json.loads((source/'manifest.json').read_text())
        verify(m)
        idle = base.idle_preflight(0)
        gpu = idle[-1]['uuid']
        with (Path('/tmp')/f'pagegauge-mlsys-{gpu}.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            out = ROOT/'results/mlsys2027_tasks_v1'/(backend+str(bits)+'_cohort_smoke_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
            out.mkdir()
            base.atomic_json(out/'fixtures.json', json.loads((ROOT/'results/mlsys2027_tasks_v1/task_fallback_smoke_20260910T081503Z_3aaa0d41/fixtures.json').read_text()))
            paths = [Path(__file__).resolve(), *[Path(__file__).with_name(n+'.py') for n in ('native_task_worker', 'reusable_lowbit_generation', 'native_hf_generation')]]
            if backend == 'bitdecode':
                paths.extend(ROOT/'experiments/mlsys2027/baselines_v1'/n for n in ('bitdecode_adapter.py', 'bitdecode_cache.py'))
                paths.extend(Path('/home/anonymous/pagegauge_baselines/bitdecode_sm120_env/lib/python3.12/site-packages')/n
                    for n in ('bit_decode_cuda.cpython-312-x86_64-linux-gnu.so', 'bit_decode/bit_decode_interface.py'))
            for path in paths:
                m['source_sha256'][str(path)] = base.sha256_file(path)
            m.update(baseline=backend, bits=bits, idle=idle, task_backend='hf' if backend == 'hf' else backend+'_int'+str(bits),
                     fixtures_sha256=base.sha256_file(out/'fixtures.json'))
            base.atomic_json(out/'manifest.json', m)
            command = ['/home/anonymous/pagegauge_baselines/kivi_mistral_sm120_env/bin/python', '-u', str(Path(__file__)), '--worker', str(out)]
            base.atomic_json(out/'invocation.json', {'command': command})
            print(str(out), flush=True)
            completion = wsl_gpu_monitor.run_process(previous, command, out, {'index': 0}, gpu)
            base.atomic_json(out/'completion.json', completion)
            verify(m)
            if completion['return_code'] or not completion['sampled_exclusivity_passed']:
                raise RuntimeError('Qualification failed; retain evidence before further runs')


if __name__ == '__main__':
    main()
