"""Report preprocessing from frozen fixtures, not re-tokenized worker inputs."""


def summarize(fixtures):
    cases = fixtures['cases']
    if not cases or len({c['case_id'] for c in cases}) != len(cases):
        raise ValueError('Unique nonempty prompt cohort required')
    total_original = total_served = truncated = 0
    for case in cases:
        original, removed = case['original_prompt_tokens'], case['removed_tokens']
        served = len(case['prompt_ids'])
        if type(original) is not int or type(removed) is not int or removed < 0 or original-served != removed:
            raise ValueError('Inconsistent frozen truncation accounting')
        total_original += original
        total_served += served
        truncated += int(removed > 0)
    return {'examples': len(cases), 'truncated_examples': truncated,
            'truncation_fraction': truncated/len(cases),
            'original_prompt_tokens': total_original, 'served_prompt_tokens': total_served,
            'removed_prompt_tokens': total_original-total_served,
            'scope': 'Frozen central preprocessing; not worker-side re-preparation counts.'}
