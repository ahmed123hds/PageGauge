"""Normalize native-worker IDs using the externally pinned modern tokenizer."""
from task_contract import prepare, validate_result


def normalize(analysis, fixtures, backend, tokenizer, context_limit, eos_ids):
    if not analysis.get('execution_complete'):
        raise ValueError('Incomplete native cohort')
    expected = {c['case_id']: c for c in fixtures['cases']}
    results = analysis['results']
    if len(expected) != len(fixtures['cases']) or len(results) != len(expected) or {r['case_id'] for r in results} != set(expected):
        raise ValueError('Incomplete/duplicate native cohort')
    normalized = []
    for row in results:
        case = expected[row['case_id']]
        contract = prepare(case['task'], case['prompt_ids'], context_limit)
        result = row['result']
        if row['requested_backend'] != backend or row['task'] != case['task']:
            raise ValueError('Native task/backend mismatch')
        if tuple(row['prompt_contract']['prompt_ids']) != contract.prompt_ids:
            raise ValueError('Prompt mismatch')
        if result.get('fallback') is not contract.fp16_fallback:
            raise ValueError('Fallback mismatch')
        executed = 'native_hf_fp16' if contract.fp16_fallback or backend == 'hf' else backend
        if result.get('executed_backend') != executed:
            raise ValueError('Execution backend mismatch')
        if contract.fp16_fallback and result.get('quantized_tokens_served') != 0:
            raise ValueError('Fallback quantized-token accounting mismatch')
        validate_result(contract, result['generated_ids'], result['stop_reason'], eos_ids)
        normalized.append({'id': case['case_id'], 'backend': backend,
            'text': tokenizer.decode(result['generated_ids'], skip_special_tokens=True),
            'status': 'complete', 'fallback': contract.fp16_fallback, 'executed_backend': executed})
    return normalized
