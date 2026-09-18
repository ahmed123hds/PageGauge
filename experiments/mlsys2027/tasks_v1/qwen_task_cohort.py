"""Qualify pinned Qwen3 own-generation on synthetic data only."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import uuid
import task_generation_worker as synthetic
from synthetic_generation import ROOT, base, previous, wsl_gpu_monitor, verify


def main():
    from transformers import AutoTokenizer
    import fcntl
    source = ROOT/'results/mlsys2027_tasks_v1/synthetic_20260909T025247Z_a44bf775/manifest.json'
    m = json.loads(source.read_text()); verify(m)
    model = Path(m['model'])
    evidence = m['input_file_evidence']
    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True, trust_remote_code=False)
    cases = [synthetic.make_case(tokenizer, 8192, depth, 2026091016+i, 'qwen3') for i, depth in enumerate((.5, .9))]
    cases = cases[:1]
    short_ids = tokenizer.apply_chat_template([{'role': 'user', 'content': 'Reply with only the word READY.'}], tokenize=True, add_generation_prompt=True, enable_thinking=False)
    cases.append({'case_id': 'short_native_fallback', 'prompt_ids': short_ids})
    for case in cases:
        case['task'] = 'qasper'
    idle = base.idle_preflight(0); gpu = idle[-1]['uuid']
    with (Path('/tmp')/f'pagegauge-mlsys-{gpu}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        out = ROOT/'results/mlsys2027_tasks_v1'/('qwen_task_cohort_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
        out.mkdir()
        base.atomic_json(out/'fixtures.json', {'cases': cases})
        m.update(model=str(model), model_family='qwen3', input_file_evidence=evidence,
            fixtures_sha256=base.sha256_file(out/'fixtures.json'), seed=2026091016,
            prompt_budgets=[8192], max_new_tokens=32,
            scope='Two synthetic retrieval prompts using qasper output budget only, NOT Qasper examples or scores; independent HF/FI/PG generation with first-case native HF-generate comparison. Reference policy S4/A128/T768. Not public-task score or optimized layer-runtime performance.')
        for path in (Path(__file__).resolve(), source, Path(synthetic.__file__).resolve(), Path(__file__).with_name('packed_hf_views.py'), Path(__file__).with_name('task_contract.py'), Path(__file__).with_name('short_prompt_fallback.py')):
            m['source_sha256'][str(path)] = base.sha256_file(path)
        base.atomic_json(out/'manifest.json', m)
        command = [sys.executable, '-u', synthetic.__file__, '--worker', str(out)]
        base.atomic_json(out/'invocation.json', {'command': command})
        print('Qwen task validation: '+str(out), flush=True)
        c = wsl_gpu_monitor.run_process(previous, command, out, {'index': 0}, gpu)
        base.atomic_json(out/'completion.json', c); verify(m)
        if c['return_code'] or not c['sampled_exclusivity_passed']:
            raise ValueError('Instruction qualification failed; retain and diagnose')


if __name__ == '__main__':
    main()
