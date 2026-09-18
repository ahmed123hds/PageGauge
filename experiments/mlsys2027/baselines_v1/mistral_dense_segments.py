"""Stateless segments around append/attention; original unmodified projections."""
import torch


class BeforeAttention(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.norm = layer.input_layernorm
        self.q = layer.self_attn.q_proj
        self.k = layer.self_attn.k_proj
        self.v = layer.self_attn.v_proj

    def forward(self, hidden):
        normalized = self.norm(hidden)
        return self.q(normalized), self.k(normalized), self.v(normalized)


class AfterAttention(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.output = layer.self_attn.o_proj
        self.norm = layer.post_attention_layernorm
        self.mlp = layer.mlp

    def forward(self, attended, residual):
        hidden = residual+self.output(attended)
        return (hidden+self.mlp(self.norm(hidden)),)
