"""Normalize native full-prefix results without imposing PageGauge fallback."""
from task_contract import prepare, validate_result


def normalize(analysis, fixtures, backend, family, tokenizer, context_limit, eos_ids):
    if not analysis.get('execution_complete') or analysis['backend'] != backend or analysis['family'] != family:
        raise ValueError('Incomplete/wrong native arm')
    expected = {c['case_id']: c for c in fixtures['cases']}
    results = analysis['results']
    if len(expected) != len(fixtures['cases']) or len(results) != len(expected) or {r['case_id'] for r in results} != set(expected):
        raise ValueError('Native cohort coverage mismatch')
    rows = []
    for row in results:
        case = expected[row['case_id']]
        contract = prepare(case['task'], case['prompt_ids'], context_limit)
        result = row['result']
        if row['task'] != case['task'] or tuple(row['prompt_contract']['prompt_ids']) != contract.prompt_ids:
            raise ValueError('Native prompt/task mismatch')
        if result.get('fallback') is not False or result['executed_backend'] != backend:
            raise ValueError('Unexpected native substitution')
        validate_result(contract, result['generated_ids'], result['stop_reason'], eos_ids)
        expected_length = len(contract.prompt_ids)+len(result['generated_ids'])-1
        layers = {'nsn': 32, 'kitty': 36}[family]
        if result['final_lengths'] != [expected_length]*layers:
            raise ValueError('Native final cache length mismatch')
        rows.append({'id': case['case_id'], 'backend': backend,
            'text': tokenizer.decode(result['generated_ids'], skip_special_tokens=True),
            'status': 'complete', 'fallback': False, 'executed_backend': backend})
    return rows
