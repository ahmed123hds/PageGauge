"""Complete-cohort scoring and paired-example uncertainty, no dataset access."""
import numpy as np
from scoring_contract import example_score, aggregate


def score_cohort(task, examples, predictions, backends, metrics, seed, draws=10000):
    ids = [e['id'] for e in examples]
    if not ids or len(set(ids)) != len(ids) or len(set(backends)) != len(backends) or 'hf' not in backends:
        raise ValueError('Unique cohort/backends including HF required')
    if draws < 1:
        raise ValueError('Positive bootstrap count required')
    keyed = {}
    for row in predictions:
        key = (row['id'], row['backend'])
        if key in keyed:
            raise ValueError('Duplicate output')
        keyed[key] = row
    if set(keyed) != {(i, b) for i in ids for b in backends}:
        raise ValueError('Incomplete or unexpected outputs; retain failures, do not silently score a subset')
    summaries, scores = {}, {}
    for backend in backends:
        rows = []
        for e in examples:
            row = keyed[e['id'], backend]
            if row.get('status') != 'complete':
                raise ValueError('Unfinished/failed output prevents complete-cohort claim')
            if type(row.get('fallback')) is not bool:
                raise ValueError('Explicit fallback disclosure required')
            if row['fallback'] and row.get('executed_backend') != 'native_hf_fp16':
                raise ValueError('Fallback execution mismatch')
            rows.append({'id': e['id'], 'score': example_score(task, row['text'], e['answers'], metrics),
                         'fallback': row['fallback']})
        summaries[backend] = aggregate(rows, ids)
        scores[backend] = np.array([r['score'] for r in rows])
    rng = np.random.default_rng(seed)
    # Shared resampling indices preserve pairing across all backend comparisons.
    differences = {b: [] for b in backends if b != 'hf'}
    for _ in range(draws):
        indices = rng.integers(0, len(ids), size=len(ids))
        for backend in differences:
            differences[backend].append(float(100*(scores[backend][indices]-scores['hf'][indices]).mean()))
    for backend, samples in differences.items():
        summaries[backend]['difference_from_hf_percentage_points'] = float(100*(scores[backend]-scores['hf']).mean())
        summaries[backend]['paired_example_95_interval'] = np.quantile(samples, [.025, .975]).tolist()
    return {'task': task, 'rows': summaries, 'seed': seed, 'draws': draws,
            'scope': 'Complete declared cohort; paired-example intervals, fallback retained in overall score and disclosed separately.'}
