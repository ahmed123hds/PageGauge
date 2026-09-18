"""Shared-FP16-prefill adapter to unmodified upstream KIVI recurrent decoding.

Use only with the pinned Mistral implementation / transformers 4.36.2. This
does not implement native prefill timing and must not be labeled as doing so.
"""
from pathlib import Path
import sys

SOURCE = Path('/home/anonymous/pagegauge_baselines/KIVI')


def imports():
    if str(SOURCE) not in sys.path:sys.path.insert(0,str(SOURCE))
    import models.mistral_kivi as native
    from quant.new_pack import triton_quantize_and_pack_along_last_dim
    return native, triton_quantize_and_pack_along_last_dim


def configure_decode(model, bits, group=32, residual=32):
    import transformers
    if transformers.__version__ != '4.36.2':
        raise ValueError('Pinned native Mistral stack requires transformers 4.36.2')
    if bits not in (2,4) or group not in (32,64,128) or residual < group or residual % group:
        raise ValueError('Unsupported native quantization configuration')
    if model.config.model_type != 'mistral':raise ValueError('Mistral-only adapter')
    native,_ = imports()
    # Reuse exactly the loaded FP16 parameters and rotary buffers. No duplicate
    # model, new initialization, folding or weight transformation is performed.
    before = {name:(p.data_ptr(),p.shape) for name,p in model.named_parameters()}
    model.__class__ = native.MistralForCausalLM_KIVI
    model.model.__class__ = native.MistralModel_KIVI
    model.config.use_flash = False
    model.config._flash_attn_2_enabled = False
    model.config.k_bits = model.config.v_bits = bits
    model.config.group_size,model.config.residual_length = group,residual
    for layer in model.model.layers:
        layer.__class__ = native.MistralDecoderLayer_KIVI
        layer.self_attn.__class__ = native.MistralAttention_KIVI
        layer.self_attn.k_bits = layer.self_attn.v_bits = bits
        layer.self_attn.group_size,layer.self_attn.residual_length = group,residual
    after = {name:(p.data_ptr(),p.shape) for name,p in model.named_parameters()}
    if before != after:raise RuntimeError('Adapter changed parameter storage')
    return model


def pack_initial_cache(legacy_cache, bits, group=32, residual=32):
    """Mirror upstream native prefill's cache packing, without rerunning prefill."""
    _,pack = imports()
    output = []
    for key,value in legacy_cache:
        length = key.shape[-2]
        if key.shape != value.shape or length <= residual or key.shape[-1] % group:
            raise ValueError('Unsupported prefill cache shape')
        key_count = length//residual*residual
        key_full = key[:,:,key_count:,:].contiguous() if key_count < length else None
        k_codes,k_scale,k_min = pack(key[:,:,:key_count,:].transpose(2,3).contiguous(),group,bits)
        v_codes,v_scale,v_min = pack(value[:,:,:length-residual,:].contiguous(),group,bits)
        v_full = value[:,:,length-residual:,:].contiguous()
        output.append((k_codes,key_full,k_scale,k_min,v_codes,v_full,v_scale,v_min,length))
    return tuple(output)
