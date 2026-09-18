"""Own-token KIVI/BitDecoding generation after shared FP16 prefix processing.

One fresh, unmodified native-stack model per call. This is a quality adapter,
not native-prefill timing or a best-serving implementation.
"""
from pathlib import Path
import sys
from generation_contract import greedy_rollout
from task_contract import validate_result


def generate(model, contract, backend, bits, eos_ids):
    import torch
    import transformers
    if transformers.__version__ != '4.36.2' or model.__class__.__name__ != 'MistralForCausalLM':
        raise ValueError('Fresh native-stack HF Mistral model required per invocation')
    if backend not in ('kivi', 'bitdecode') or bits not in (2, 4) or (backend == 'bitdecode' and bits != 4):
        raise ValueError('Unsupported baseline configuration')
    if contract.fp16_fallback or contract.prefill_tokens < 4096:
        raise ValueError('Short-prompt native fallback must be dispatched explicitly')
    if model.config.sliding_window is not None:
        raise ValueError('Full-context model required')
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'baselines_v1'))
    if backend == 'kivi':
        from kivi_adapter import configure_decode, pack_initial_cache
    else:
        from bitdecode_adapter import configure_decode, pack_initial_cache
    from kivi_quality import cache_accounting
    tokens = torch.tensor([contract.prompt_ids], dtype=torch.long, device=next(model.parameters()).device)
    past = None
    with torch.inference_mode():
        # Bounded eager prefill, same protocol as qualified corpus adapters.
        for start in range(0, contract.prefill_tokens, 256):
            output = model(tokens[:, start:min(start+256, contract.prefill_tokens)],
                           past_key_values=past, use_cache=True)
            past = output.past_key_values
        legacy = past.to_legacy_cache() if hasattr(past, 'to_legacy_cache') else past
        if any(k.shape[-2] != contract.prefill_tokens for k, v in legacy):
            raise ValueError('Incomplete prefix')
        configure_decode(model, bits)
        packed = pack_initial_cache(legacy, bits)
        del past, legacy, output
        past = packed
        del packed
        initial_bytes = cache_accounting(past)

        def step(token, position):
            nonlocal past
            if any(layer[-1] != position for layer in past):
                raise ValueError('Native cache position mismatch')
            output = model(torch.tensor([[token]], device=tokens.device), past_key_values=past, use_cache=True)
            past = output.past_key_values
            if not torch.isfinite(output.logits[:, -1]).all():
                raise ValueError('Nonfinite generation logits')
            return output.logits[:, -1].argmax(-1).item()

        result = greedy_rollout(contract.prompt_ids, contract.prefill_tokens,
                                contract.max_new_tokens, eos_ids, step)
        validate_result(contract, result['generated_ids'], result['stop_reason'], eos_ids)
        expected = len(contract.prompt_ids)+len(result['generated_ids'])-1
        if any(layer[-1] != expected for layer in past):
            raise ValueError('Incomplete final cache')
        result.update(executed_backend=backend+'_int'+str(bits), fallback=False,
                      cache_accounting={'initial': initial_bytes, 'final': cache_accounting(past)},
                      final_cache_tokens=expected,
                      scope='Own greedy continuation; shared FP16 prefix then native low-bit recurrence. No speed claim.')
    return result
