"""Long/short resident-model qualification for both full-native baseline stacks."""
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid
from synthetic_generation import ROOT, base, previous, wsl_gpu_monitor, verify


def main():
    import fcntl
    from transformers import AutoTokenizer
    specs = [('kitty', 'kitty2_kitty_generation_20260910T093848Z_18cb321f', 'kitty_pro')]
    for family, source_name, quantized in specs:
        source = ROOT/'results/mlsys2027_tasks_v1'/source_name
        for backend in (quantized,):
            m = json.loads((source/'manifest.json').read_text())
            verify(m)
            tokenizer = AutoTokenizer.from_pretrained(m['model'], local_files_only=True, trust_remote_code=False)
            kwargs = {'enable_thinking': False} if family == 'kitty' else {}
            short = tokenizer.apply_chat_template([{'role': 'user', 'content': 'Reply with only the word READY.'}],
                         tokenize=True, add_generation_prompt=True, **kwargs)
            fixtures = json.loads((source/'fixtures.json').read_text())
            fixtures['cases'].append({'case_id': 'short_native_request', 'task': 'qasper', 'prompt_ids': short})
            originals = fixtures['cases']
            fixtures['cases'] = [dict(case, case_id=case['case_id']+'_repeat'+str(i))
                                 for i in range(6) for case in originals]
            idle = base.idle_preflight(0)
            gpu = idle[-1]['uuid']
            with (Path('/tmp')/f'pagegauge-mlsys-{gpu}.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                out = ROOT/'results/mlsys2027_tasks_v1'/(family+'_'+backend+'_eos_v2_observed_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
                out.mkdir()
                base.atomic_json(out/'fixtures.json', fixtures)
                worker = Path(__file__).with_name('native_full_task_worker_v2.py')
                for path in (Path(__file__).resolve(), worker, Path(__file__).with_name('task_contract.py')):
                    m['source_sha256'][str(path)] = base.sha256_file(path)
                m.update(native_family=family, task_backend=backend, cache_capacity=32768,
                         fixtures_sha256=base.sha256_file(out/'fixtures.json'), idle=idle,
                         scope='Two synthetic requests with one model load; native short behavior, no implicit fallback or public task score.')
                base.atomic_json(out/'manifest.json', m)
                command = ['/home/anonymous/pagegauge_baselines/'+family+'_sm120_env/bin/python', '-u', str(worker), '--worker', str(out)]
                base.atomic_json(out/'invocation.json', {'command': command})
                print(str(out), flush=True)
                completion = wsl_gpu_monitor.run_process(previous, command, out, {'index': 0}, gpu)
                base.atomic_json(out/'completion.json', completion)
                verify(m)
                if completion['return_code'] or not completion['sampled_exclusivity_passed']:
                    raise RuntimeError('Native cohort failed; retain and diagnose before continuing')


if __name__ == '__main__':
    main()
