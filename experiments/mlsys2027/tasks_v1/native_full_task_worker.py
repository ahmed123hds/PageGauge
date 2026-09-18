"""Resident native NSN/Kitty task worker. No PageGauge short-prefix fallback."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import traceback
from synthetic_generation import base, verify
from native_full_generation import load, generate
from task_contract import prepare


def worker(out):
    import torch
    m = json.loads((out/'manifest.json').read_text())
    verify(m)
    if base.sha256_file(out/'fixtures.json') != m['fixtures_sha256']:
        raise ValueError('Changed frozen prompts')
    cases = json.loads((out/'fixtures.json').read_text())['cases']
    if not cases or len({c['case_id'] for c in cases}) != len(cases):
        raise ValueError('Nonempty unique cohort required')
    torch.manual_seed(m['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    model, make_cache, account = load(m['model'], m['native_family'], m['task_backend'], m['cache_capacity'])
    eos = model.config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    rows = []
    for case in cases:
        contract = prepare(case['task'], case['prompt_ids'], model.config.max_position_embeddings)
        if len(contract.prompt_ids)+contract.max_new_tokens > m['cache_capacity']:
            raise ValueError('Native allocation capacity insufficient')
        result = generate(model, contract, eos, make_cache, account)
        result.update(executed_backend=m['task_backend'], fallback=False)
        # Prompt-budget policy is shared; PG's minimum prefix is not a native
        # method limitation. Native short inputs must execute or fail visibly.
        prompt = asdict(contract)
        prompt.pop('fp16_fallback')
        prompt.pop('fallback_reason')
        rows.append({'case_id': case['case_id'], 'task': case['task'],
                     'prompt_contract': prompt, 'result': result})
        base.atomic_json(out/'progress.json', {'execution_complete': False, 'results': rows})
    verify(m)
    base.atomic_json(out/'analysis.json', {'execution_complete': True, 'results': rows,
        'backend': m['task_backend'], 'family': m['native_family'], 'model_loads': 1,
        'cache_capacity': m['cache_capacity'],
        'scope': 'Native full-prefix independent generation. No implicit fallback, timing or benchmark score claim.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--worker', required=True, type=Path)
    args = parser.parse_args()
    try:
        worker(args.worker)
    except BaseException:
        base.atomic_json(args.worker/'failure.json', {'traceback': traceback.format_exc()})
        raise
