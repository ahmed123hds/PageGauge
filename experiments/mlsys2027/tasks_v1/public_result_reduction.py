"""Reduce all declared tasks/control groups without pooling software stacks."""
from cohort_scoring import score_cohort
from prompt_accounting import summarize
from public_matrix_spec import GROUPS, POLICY
from task_contract import TASK_OUTPUT_LIMITS


def reduce_complete(references, fixtures, predictions, metrics):
    expected = {(task, group['id']) for task in TASK_OUTPUT_LIMITS for group in GROUPS}
    if set(predictions) != expected:
        raise ValueError('All declared task/control-group predictions required')
    if set(references) != set(TASK_OUTPUT_LIMITS):
        raise ValueError('All declared reference tasks required')
    required_fixtures = {(task, g['model']) for task in TASK_OUTPUT_LIMITS for g in GROUPS}
    if set(fixtures) != required_fixtures:
        raise ValueError('Exact per-model fixture coverage required')
    results = {}
    for task in TASK_OUTPUT_LIMITS:
        refs = references[task]
        if any(r['task'] != task for r in refs):
            raise ValueError('Reference task mismatch')
        ref_ids = [r['id'] for r in refs]
        results[task] = {}
        for group in GROUPS:
            fixture = fixtures[task, group['model']]
            if [c['case_id'] for c in fixture['cases']] != ref_ids:
                raise ValueError('Fixture/reference IDs or order differ')
            stats = summarize(fixture)
            scored = score_cohort(task, refs, predictions[task, group['id']], group['arms'],
                metrics, POLICY['bootstrap']['seed'], POLICY['bootstrap']['draws'])
            scored['prompt_accounting'] = stats
            scored['model_family'] = group['model']
            scored['control_group'] = group['id']
            results[task][group['id']] = scored
    return {'tasks': results, 'scope': 'Complete declared cohorts, separate software-stack HF controls. '
            'No cross-stack pooled HF reference or speed inference. No quality acceptance threshold inferred.'}
