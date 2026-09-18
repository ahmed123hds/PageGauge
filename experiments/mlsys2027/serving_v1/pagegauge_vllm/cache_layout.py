"""Deterministic serving-pool plus fully reserved sidecar accounting."""
from dataclasses import dataclass
from .partition import Policy, PAGE


@dataclass(frozen=True)
class Layout:
    layers: int
    physical_pages: int
    maximum_requests: int
    kv_heads: int = 8
    query_heads: int = 32
    head_dim: int = 128
    policy: Policy = Policy()

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in (self.layers,self.physical_pages,self.maximum_requests,self.kv_heads,self.query_heads,self.head_dim)) or self.query_heads%self.kv_heads:
            raise ValueError('Invalid cache geometry')

    def accounting(self):
        l,n,r,h,q,d = self.layers,self.physical_pages,self.maximum_requests,self.kv_heads,self.query_heads,self.head_dim
        tensors = {'key_codes': l*n*PAGE*h*d, 'value_codes': l*n*PAGE*h*d,
            'key_scales': 2*l*n*h, 'value_scales': 2*l*n*h,
            'exact_key': 2*l*r*self.policy.storage_pages*PAGE*h*d,
            'exact_value': 2*l*r*self.policy.storage_pages*PAGE*h*d,
            'key_center': 2*l*r*h*d, 'value_center': 2*l*r*h*d, 'output_center': 2*l*r*q*d}
        page_pool = sum(tensors[k] for k in ('key_codes','value_codes','key_scales','value_scales'))
        sidecar = sum(tensors.values())-page_pool
        return {'tensors': tensors, 'page_pool_bytes': page_pool, 'reserved_sidecar_bytes': sidecar,
            'total_served_capacity_bytes': sum(tensors.values()),
            'per_layer_page_bytes': 2*PAGE*h*d+4*h,
            'scope': 'Declared persistent code/metadata pool and all admitted-request sidecar slots, including empty reserved slots. Excludes weights, workspace, graph pools, input/output buffers and allocator padding; those must be separately reserved/measured before admission.'}

    def admissible_pages(self, cache_budget_bytes):
        accounting = self.accounting()
        remaining = cache_budget_bytes-accounting['reserved_sidecar_bytes']
        if remaining < 0: raise MemoryError('Budget cannot hold reserved exact/center sidecars')
        return remaining//(self.layers*accounting['per_layer_page_bytes'])
