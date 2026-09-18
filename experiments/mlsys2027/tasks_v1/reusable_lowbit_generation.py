"""Restore native HF dispatch after each low-bit request; retain model weights."""
from copy import deepcopy
from lowbit_generation import generate as generate_once


def generate(model, contract, backend, bits, eos_ids):
    modules = [model, model.model]
    for layer in model.model.layers:
        modules.extend((layer, layer.self_attn))
    classes = [(module, type(module)) for module in modules]
    config = deepcopy(model.config.__dict__)
    names = ('k_bits', 'v_bits', 'group_size', 'residual_length')
    attributes = [(layer.self_attn, {name: (name in layer.self_attn.__dict__, layer.self_attn.__dict__.get(name))
                                   for name in names}) for layer in model.model.layers]
    weights = {n: (p.data_ptr(), tuple(p.shape), p._version) for n, p in model.named_parameters()}
    try:
        return generate_once(model, contract, backend, bits, eos_ids)
    finally:
        for module, cls in classes:
            module.__class__ = cls
        model.config.__dict__.clear()
        model.config.__dict__.update(config)
        for module, values in attributes:
            for name, (existed, value) in values.items():
                if existed:
                    setattr(module, name, value)
                else:
                    module.__dict__.pop(name, None)
        observed = {n: (p.data_ptr(), tuple(p.shape), p._version) for n, p in model.named_parameters()}
        if observed != weights:
            raise RuntimeError('Baseline request changed model parameter storage or version')
