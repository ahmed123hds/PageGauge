"""Eight-task adapter to externally pinned official LongBench metric functions.

Reference: THUDM/LongBench commit2e00731f8d0bff23dc4325161044d0ed8af94c1e,
LongBench/eval.py and metrics.py. No benchmark data loaded by this module.
"""
METRICS = {'qasper': 'qa_f1_score', 'multifieldqa_en': 'qa_f1_score',
           'hotpotqa': 'qa_f1_score', 'triviaqa': 'qa_f1_score',
           'gov_report': 'rouge_score', 'qmsum': 'rouge_score',
           'lcc': 'code_sim_score', 'repobench-p': 'code_sim_score'}


def example_score(task, prediction, answers, official_metrics):
    if task not in METRICS or not isinstance(prediction, str) or not answers:
        raise ValueError('Declared task, prediction and references required')
    if any(not isinstance(answer, str) for answer in answers):
        raise ValueError('References must be strings')
    if task == 'triviaqa':
        prediction = prediction.lstrip('\n').split('\n')[0]
    metric = getattr(official_metrics, METRICS[task])
    values = [float(metric(prediction, answer, all_classes=[])) for answer in answers]
    if any(not 0 <= value <= 1 for value in values):
        raise ValueError('Invalid metric result')
    return max(values)


def aggregate(rows, expected_ids):
    if len(set(expected_ids)) != len(expected_ids) or not expected_ids:
        raise ValueError('Unique nonempty expected cohort required')
    if len(rows) != len(expected_ids) or {r['id'] for r in rows} != set(expected_ids):
        raise ValueError('Missing, duplicate or unexpected example; do not drop failures')
    if any(not 0 <= r['score'] <= 1 for r in rows):
        raise ValueError('Invalid per-example score')
    return {'score_percent': 100*sum(r['score'] for r in rows)/len(rows),
            'examples': len(rows), 'fallback_fraction': sum(bool(r['fallback']) for r in rows)/len(rows)}
