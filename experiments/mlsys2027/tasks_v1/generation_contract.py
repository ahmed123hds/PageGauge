"""CPU-testable prompt partition and autonomous greedy rollout contract."""


def aligned_prefix(prompt_length, page=16):
    if prompt_length < 2 or page <= 0:
        raise ValueError('A nonempty prompt remainder and prefix are required')
    context = ((prompt_length-1)//page)*page
    if context == 0:
        raise ValueError('Use declared native FP16 fallback for sub-page prompts')
    return context


def greedy_rollout(prompt_ids, context, max_new_tokens, eos_ids, step_token):
    """step_token consumes one real token at its absolute position.

    The caller prefills prompt_ids[:context]. Each arm supplies its own step
    function; no reference-generated token or reference transition logit enters.
    """
    if not 0 < context < len(prompt_ids) or context % 16 or max_new_tokens < 1:
        raise ValueError('Invalid page-aligned prefix, remainder or output limit')
    if any(not isinstance(token, int) for token in prompt_ids):
        raise ValueError('Prompt must contain integer token IDs')
    eos_ids = set(eos_ids)
    calls = 0
    next_token = None
    for position in range(context, len(prompt_ids)):
        next_token = int(step_token(prompt_ids[position], position))
        calls += 1
    generated = []
    for index in range(max_new_tokens):
        generated.append(next_token)
        if next_token in eos_ids:
            return {'generated_ids': generated, 'stop_reason': 'eos', 'decoder_step_calls': calls}
        if index+1 < max_new_tokens:
            next_token = int(step_token(next_token, len(prompt_ids)+index))
            calls += 1
    return {'generated_ids': generated, 'stop_reason': 'max_new_tokens', 'decoder_step_calls': calls}
