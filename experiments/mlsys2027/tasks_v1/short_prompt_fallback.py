"""Explicit short-prompt fallback; never label native FP16 as quantized work."""
from task_contract import validate_result


def run(contract, eos_ids, generate_native):
    if not contract.fp16_fallback:
        raise ValueError('Fallback only for declared unsupported short prefix')
    arms = {}
    for requested in ('hf', 'flashinfer_fp16', 'page_gauge'):
        # Separate invocations: never clone HF answers into another arm.
        result = generate_native(list(contract.prompt_ids), eos_ids, contract.max_new_tokens)
        validate_result(contract, result['generated_ids'], result['stop_reason'], eos_ids)
        arms[requested] = {**result, 'requested_backend': requested,
            'executed_backend': 'native_hf_fp16', 'fallback': True,
            'fallback_reason': contract.fallback_reason, 'quantized_tokens_served': 0}
    return arms
