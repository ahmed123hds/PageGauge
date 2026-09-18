"""Temporary zero-copy HF projection views over already packed FP16 weights.

Preparation optimization only. Never repack/refit weights or change the timed
decoder. Not enabled in a GPU runner until native-HF equivalence is verified.
"""
from contextlib import contextmanager


@contextmanager
def hf_projection_views(model):
    import torch
    replacements = []
    def linear_view(weight):
        module = torch.nn.Linear(weight.shape[1],weight.shape[0],bias=False,device='meta',dtype=weight.dtype)
        module.weight = torch.nn.Parameter(weight,requires_grad=False)
        if module.weight.untyped_storage().data_ptr() != weight.untyped_storage().data_ptr():
            raise ValueError('Projection wrapper allocated new weight storage')
        return module
    try:
        for layer in model.model.layers:
            attention, mlp = layer.self_attn, layer.mlp
            for owner,names,packed,widths,bias_name in (
                (attention,('q_proj','k_proj','v_proj'),attention.qkv_weight,
                    (attention._pkv_q_width,attention._pkv_kv_width,attention._pkv_kv_width),'qkv_bias'),
                (mlp,('gate_proj','up_proj'),mlp.gate_up_weight,
                    (mlp._pkv_intermediate,mlp._pkv_intermediate),'gate_up_bias')):
                if packed.dtype != torch.float16 or packed.ndim != 2 or not packed.is_contiguous() or sum(widths) != packed.shape[0]:
                    raise ValueError('Unexpected packed projection contract')
                if getattr(owner,bias_name,None) is not None or any(getattr(owner,name) is not None for name in names):
                    raise ValueError('Only bias-free already-packed projections are supported')
                offset = 0
                for name,width in zip(names,widths):
                    setattr(owner,name,linear_view(packed[offset:offset+width]))
                    replacements.append((owner,name))
                    offset += width
        yield
    finally:
        for owner,name in reversed(replacements): setattr(owner,name,None)
