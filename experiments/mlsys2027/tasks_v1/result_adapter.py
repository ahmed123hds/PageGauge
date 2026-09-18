"""Normalize completed generation artifacts without inferring missing arms."""


def normalize(analysis, fixtures, backends):
    if not analysis.get('execution_complete'):
        raise ValueError('Incomplete generation run')
    expected = {c['case_id']: c for c in fixtures['cases']}
    if len(expected) != len(fixtures['cases']):
        raise ValueError('Duplicate fixture IDs')
    results = analysis['results']
    if len(results) != len(expected) or {r['case_id'] for r in results} != set(expected):
        raise ValueError('Missing, duplicate or unexpected generated case')
    rows = []
    for result in results:
        case = expected[result['case_id']]
        contract = result['prompt_contract']
        if contract['task'] != case['task'] or set(result['arms']) != set(backends):
            raise ValueError('Wrong task or backend coverage')
        if type(contract['fp16_fallback']) is not bool:
            raise ValueError('Missing fallback policy')
        for backend, arm in result['arms'].items():
            fallback = contract['fp16_fallback']
            if fallback:
                if arm.get('fallback') is not True or arm.get('executed_backend') != 'native_hf_fp16' or arm.get('quantized_tokens_served') != 0:
                    raise ValueError('Undisclosed native fallback')
            elif arm.get('fallback', False):
                raise ValueError('Unexpected fallback')
            if not isinstance(arm['generated_text'], str) or not arm['generated_ids'] or arm['stop_reason'] not in ('eos', 'max_new_tokens'):
                raise ValueError('Invalid generated output')
            rows.append({'id': result['case_id'], 'backend': backend, 'text': arm['generated_text'],
                         'status': 'complete', 'fallback': fallback,
                         'executed_backend': 'native_hf_fp16' if fallback else backend})
    return rows
