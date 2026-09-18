"""Fixed common-engine shape pilot: development only, no speed CI."""
CONTEXTS = (8192, 20480, 30720)
BATCHES = (1, 4, 8)
BACKENDS = ('flashinfer_fp16', 'page_gauge', 'kivi_int4', 'bitdecode_int4', 'kivi_int2')
STEPS = 1536
STRIDE = 35000
BUDGET = 28*1024**3


def schedule(weights, maximum_positions):
    from frontier_capacity import lower_bound
    if any(c+STEPS > maximum_positions or c+STEPS+1 > STRIDE for c in CONTEXTS):
        raise ValueError('Context overflow or overlapping independent request windows')
    cells = []
    # Full B1 validation of every new shape before any performance cells.
    for context in CONTEXTS:
        for backend in BACKENDS:
            cells.append({'backend': backend, 'batch': 1, 'context': context,
                'validate': True, 'repeats': 0, 'action': 'run'})
    for context in CONTEXTS:
        for batch in BATCHES:
            for backend in BACKENDS:
                bits = 16 if backend == 'flashinfer_fp16' else 8 if backend == 'page_gauge' else 2 if backend == 'kivi_int2' else 4
                bound = lower_bound(weights, batch, bits, length=context+STEPS)
                cells.append({'backend': backend, 'batch': batch, 'context': context,
                    'validate': False, 'repeats': 3, 'lower_bound_bytes': bound,
                    'action': 'analytically_infeasible' if bound > BUDGET else 'run'})
    return [dict(c, index=i) for i, c in enumerate(cells)]


def validate_result(result, manifest):
    if any(result[k] != manifest[k] for k in ('backend', 'batch', 'context', 'decode_steps')):
        raise ValueError('Wrong measured shape/backend')
    validation = manifest['validate']
    if result['validation_only'] != validation or len(result['rows']) != (1 if validation else 3):
        raise ValueError('Incomplete shape evaluation')
    rows = result['rows'] if validation else [result['warmup']]+result['rows']
    for row in rows:
        if row['steps'] != STEPS or row['request_tokens'] != manifest['batch']*STEPS:
            raise ValueError('Short or replicated capacity trajectory')
    if result['allocator_budget_bytes'] != BUDGET:
        raise ValueError('Different allocator budget')
