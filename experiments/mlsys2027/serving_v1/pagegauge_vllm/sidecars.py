"""Allocate the exact/center tensors reserved outside the engine code pool.

The caller supplies a cache-only budget AFTER weights, workspace, graphs and
allocator overhead are reserved. This does not install an engine allocator hook.
"""
from .partition import PAGE


def allocate_sidecars(layout, *, device, cache_budget_bytes):
    import torch

    if type(cache_budget_bytes) is not int or cache_budget_bytes < 0:
        raise ValueError('Nonnegative integer cache-only byte budget required')
    accounting = layout.accounting()
    if accounting['total_served_capacity_bytes'] > cache_budget_bytes:
        raise MemoryError('Code pool plus all reserved sidecars exceed cache budget')
    exact_shape = (layout.layers, layout.maximum_requests * layout.policy.storage_pages,
                   PAGE, layout.kv_heads, layout.head_dim)
    center_shape = (layout.layers, layout.maximum_requests,
                    layout.kv_heads, layout.head_dim)
    shapes = {
        'exact_key': exact_shape,
        'exact_value': exact_shape,
        'key_center': center_shape,
        'value_center': center_shape,
        'output_center': (layout.layers, layout.maximum_requests,
                          layout.query_heads, layout.head_dim),
    }
    # Startup initialization only. Request reuse still requires ordered repacking.
    result = {name: torch.zeros(shape, dtype=torch.float16, device=device)
              for name, shape in shapes.items()}
    for name, tensor in result.items():
        if tensor.numel() * tensor.element_size() != accounting['tensors'][name]:
            raise RuntimeError('Sidecar allocation differs from declared accounting')
    return result
