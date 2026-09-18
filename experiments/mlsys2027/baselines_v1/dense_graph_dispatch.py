"""Opt-in fixed-shape graph for a stateless dense submodule, not KV mutation.

Keeps original weights and operations. Input copy is part of every invocation.
Output is borrowed until the next call; callers must consume it immediately.
Only inference on the capture stream is supported. No automatic eager fallback.
"""
import torch


class DenseGraph:
    def __init__(self, module, example):
        if module.training or example.device.type != 'cuda' or example.requires_grad:
            raise ValueError('CUDA inference-only module required')
        self.module = module
        self.stream = torch.cuda.current_stream(example.device)
        self.signature = (example.shape, example.dtype, example.device)
        self.parameters = [(p, p.data_ptr(), p._version) for p in module.parameters()]
        if list(module.buffers()):
            raise ValueError('Buffered/stateful modules are not supported')
        self.input = torch.empty_like(example)
        self.input.copy_(example)
        warm = torch.cuda.Stream(device=example.device)
        warm.wait_stream(self.stream)
        with torch.cuda.stream(warm), torch.inference_mode():
            for _ in range(3):
                module(self.input)
        self.stream.wait_stream(warm)
        self.graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(self.graph):
            self.output = module(self.input)
        if not isinstance(self.output, torch.Tensor):
            raise ValueError('Single tensor output required')
        self.calls = 0

    def __call__(self, value):
        if torch.is_grad_enabled() or self.module.training:
            raise ValueError('Inference-only graph')
        if (value.shape, value.dtype, value.device) != self.signature:
            raise ValueError('Uncaptured dense shape/dtype/device')
        if torch.cuda.current_stream(value.device) != self.stream:
            raise ValueError('Cross-stream reuse is unsupported')
        current = list(self.module.parameters())
        if len(current) != len(self.parameters) or any(
                p is not old or p.data_ptr() != ptr or p._version != version
                for p, (old, ptr, version) in zip(current, self.parameters)):
            raise ValueError('Captured weights changed')
        self.input.copy_(value)
        self.graph.replay()
        self.calls += 1
        return self.output
