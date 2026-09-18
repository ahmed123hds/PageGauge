"""Native NSN/Kitty loader and autonomous full-prefix generation quality adapter."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'baselines_v1'))
from task_contract import validate_result


def load(model_path, family, backend, capacity):
    import torch
    if (family, backend) not in (('nsn', 'hf'), ('nsn', 'nsn_int2'), ('kitty', 'hf'), ('kitty', 'kitty_pro')):
        raise ValueError('Unsupported native family/backend')
    m = {'model': model_path, 'family': family, 'backend': backend}
    kwargs = dict(torch_dtype=torch.float16, local_files_only=True,
                  low_cpu_mem_usage=True, device_map={'': 'cuda:0'}, attn_implementation='sdpa')
    make_cache = lambda: None
    if m['family'] == 'nsn':
        from nsn_quality import SOURCE, cache_bytes
        sys.path.insert(0, str(SOURCE))
        if m['backend'] == 'hf':
            from transformers import MistralForCausalLM
            model = MistralForCausalLM.from_pretrained(m['model'], **kwargs).eval()
        else:
            from src.models.mistral import QuantizedMistralForCausalLM
            from src.utils import rotate_v_proj, rotate_o_proj
            from src.quantizers.nsn_quantizer import NSNQuantizer
            quant = {'name': 'NSNQuantizer', 'kwargs': {'n_bits': 2,
                'codebook_path': str(SOURCE/'codebooks/2bit_codebook.pt'),
                'window_size': 64, 'residual_size': 64, 'hadamard': True}}
            model = QuantizedMistralForCausalLM.from_pretrained(m['model'],
                quant_config=quant, forward_quant=False, **kwargs).half().eval()
            released = NSNQuantizer(**quant['kwargs']).half().cuda()
            expected = dict(released.named_buffers())
            for layer in model.model.layers:
                for name, value in layer.self_attn.quantizer.named_buffers():
                    if not torch.equal(value, expected[name]):
                        raise ValueError('Released NSN codebook mismatch')
                rotate_v_proj(layer.self_attn.v_proj, 128)
                rotate_o_proj(layer.self_attn.o_proj, 128)
            del released, expected
        account = lambda past: cache_bytes(model, past)
        if model.config.sliding_window is not None or len(model.model.layers) != 32:
            raise ValueError('Wrong Mistral configuration')
    else:
        from kitty_quality import cache_bytes
        from transformers import Qwen3ForCausalLM
        from kitty.models.qwen3 import Qwen3ForCausalLM_Kitty
        from kitty.kvcache import get_kvcache_kitty
        cls = Qwen3ForCausalLM if m['backend'] == 'hf' else Qwen3ForCausalLM_Kitty
        model, loading = cls.from_pretrained(m['model'], output_loading_info=True, **kwargs)
        if any(loading.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
            raise ValueError('Checkpoint loading mismatch: '+str(loading))
        model.eval()
        if m['backend'] == 'kitty_pro':
            make_cache = lambda: get_kvcache_kitty(model.config, 1, capacity)
        account = cache_bytes
        if (model.config.model_type, len(model.model.layers)) != ('qwen3', 36):
            raise ValueError('Wrong Qwen configuration')

    return model, make_cache, account


def generate(model, contract, eos_ids, make_cache, account):
    import torch
    ids = torch.tensor([contract.prompt_ids], device=next(model.parameters()).device, dtype=torch.long)
    generated = []
    with torch.inference_mode():
        # Native full-prefix cache initialization, not shared FP16 packing.
        output = model.model(ids, past_key_values=make_cache(), use_cache=True)
        past = output.past_key_values
        logits = model.lm_head(output.last_hidden_state[:, -1:])
        del output
        initial = account(past)
        for index in range(contract.max_new_tokens):
            if not torch.isfinite(logits).all():
                raise ValueError('Nonfinite native generation')
            token = logits[:, -1].argmax(-1).item()
            generated.append(token)
            if token in eos_ids or index+1 == contract.max_new_tokens:
                break
            output = model(torch.tensor([[token]], device=ids.device), past_key_values=past, use_cache=True)
            past, logits = output.past_key_values, output.logits
        expected = len(contract.prompt_ids)+len(generated)-1
        lengths = [past.get_seq_length(i) for i in range(len(model.model.layers))]
        if lengths != [expected]*len(lengths):
            raise ValueError('Incomplete native generated cache')
        result = {'generated_ids': generated,
                  'stop_reason': 'eos' if generated[-1] in eos_ids else 'max_new_tokens',
                  'cache_accounting': {'initial': initial, 'final': account(past)},
                  'final_lengths': lengths,
                  'scope': 'Native full-prefix own generation; no timing claim.'}
        validate_result(contract, generated, result['stop_reason'], eos_ids)
        return result
