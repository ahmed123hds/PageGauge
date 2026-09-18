"""Opt-in Qwen3 model integration; production files/defaults remain unchanged.

Install into one freshly imported PageGauge worker module before model checks
or projection packing. Both FI and PG execute the checkpoint's native per-head
Q/K normalization before the existing RoPE/append path. This is model semantics,
not a new quantizer or kernel. Full GPU/model validation is still required.
"""


def validate_qwen_model(model):
    config = model.config
    if config.model_type != 'qwen3':
        raise ValueError('Expected Qwen3')
    layers = int(config.num_hidden_layers)
    hq, hkv = int(config.num_attention_heads), int(config.num_key_value_heads)
    hidden, dim = int(config.hidden_size), int(config.head_dim)
    if dim != 128 or hq % hkv or hq//hkv not in (1, 2, 4, 8):
        raise ValueError('Unsupported Qwen GQA/head shape')
    if (hq, hkv) != (32, 8):
        raise ValueError('Current production graph wrappers require Hq32/Hkv8')
    if config.hidden_act != 'silu' or getattr(config, 'use_sliding_window', False):
        raise ValueError('Only full-context SiLU Qwen3 is audited')
    if getattr(config, 'sliding_window', None) is not None:
        raise ValueError('Sliding-window attention is not implemented here')
    if len(model.model.layers) != layers:
        raise ValueError('Layer-count mismatch')
    for layer in model.model.layers:
        for name in ('q_norm', 'k_norm'):
            norm = getattr(layer.self_attn, name, None)
            if norm is None or tuple(norm.weight.shape) != (dim,):
                raise ValueError('Missing checkpoint per-head normalization: '+name)
    return layers, hq, hkv, hidden


def install(production):
    """Return decoder class; idempotent in this one process, no on-disk mutation."""
    if getattr(production, '_pagegauge_qwen3_adapter', False):
        return production.TransformerDecoder
    original_check = production.BASE_E2E.check_model
    original_decoder = production.TransformerDecoder

    def check_model(model):
        if getattr(model.config, 'model_type', None) == 'qwen3':
            return validate_qwen_model(model)
        return original_check(model)

    class QwenAwareDecoder(original_decoder):
        def append(self, layer, query, key, value, position):
            if self.model.config.model_type == 'qwen3':
                attention = self.model.model.layers[layer].self_attn
                query = attention.q_norm(query).contiguous()
                key = attention.k_norm(key).contiguous()
            return super().append(layer, query, key, value, position)

    production.BASE_E2E.check_model = check_model
    production.TransformerDecoder = QwenAwareDecoder
    production._pagegauge_qwen3_adapter = True
    return QwenAwareDecoder
