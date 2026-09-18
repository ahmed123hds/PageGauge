"""Native-stack HF control with the same prefix partition and own continuation."""
from generation_contract import greedy_rollout
from task_contract import validate_result


def generate(model, contract, eos_ids):
    import torch
    tokens = torch.tensor([contract.prompt_ids], dtype=torch.long, device=next(model.parameters()).device)
    past = None
    with torch.inference_mode():
        for start in range(0, contract.prefill_tokens, 256):
            output = model(tokens[:, start:min(start+256, contract.prefill_tokens)], past_key_values=past, use_cache=True)
            past = output.past_key_values

        def step(token, position):
            nonlocal past
            legacy = past.to_legacy_cache() if hasattr(past, 'to_legacy_cache') else past
            if any(k.shape[-2] != position for k, v in legacy):
                raise ValueError('HF cache position mismatch')
            output = model(torch.tensor([[token]], device=tokens.device), past_key_values=past, use_cache=True)
            past = output.past_key_values
            if not torch.isfinite(output.logits[:, -1]).all():
                raise ValueError('Nonfinite HF logits')
            return output.logits[:, -1].argmax(-1).item()

        result = greedy_rollout(contract.prompt_ids, contract.prefill_tokens, contract.max_new_tokens, eos_ids, step)
        validate_result(contract, result['generated_ids'], result['stop_reason'], eos_ids)
        result.update(executed_backend='native_hf_fp16', fallback=False)
        return result
