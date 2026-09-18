"""Fixed-shape multi-tensor graph for stateless decoder segments.

Outputs are borrowed until the next invocation. Input copies are always timed.
Only same-stream inference with immutable parameters and no buffers is allowed.
"""
import torch


class SegmentGraph:
    def __init__(self, module, examples):
        if module.training or list(module.buffers()) or not examples:
            raise ValueError('Stateless inference module required')
        if any(x.device.type != 'cuda' or x.device != examples[0].device or x.requires_grad for x in examples):
            raise ValueError('Same-device CUDA inputs required')
        self.module = module
        self.stream = torch.cuda.current_stream(examples[0].device)
        self.signatures = [(x.shape, x.dtype, x.device) for x in examples]
        self.weights = [(p, p.data_ptr(), p._version) for p in module.parameters()]
        self.inputs = tuple(torch.empty_like(x) for x in examples)
        for target, source in zip(self.inputs, examples):
            target.copy_(source)
        warm = torch.cuda.Stream(device=examples[0].device)
        warm.wait_stream(self.stream)
        with torch.cuda.stream(warm), torch.inference_mode():
            for _ in range(3):
                module(*self.inputs)
        self.stream.wait_stream(warm)
        self.graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(self.graph):
            self.outputs = module(*self.inputs)
        if not isinstance(self.outputs, tuple) or not self.outputs or any(not isinstance(x, torch.Tensor) for x in self.outputs):
            raise ValueError('Nonempty tensor tuple output required')
        self.calls = 0

    def __call__(self, *values):
        if torch.is_grad_enabled() or self.module.training:
            raise ValueError('Inference only')
        if [(x.shape, x.dtype, x.device) for x in values] != self.signatures:
            raise ValueError('Uncaptured inputs')
        if torch.cuda.current_stream(values[0].device) != self.stream:
            raise ValueError('Cross-stream replay unsupported')
        current = list(self.module.parameters())
        if len(current) != len(self.weights) or any(p is not old or p.data_ptr() != ptr or p._version != version for p, (old, ptr, version) in zip(current, self.weights)):
            raise ValueError('Weights changed')
        # Reject borrowed outputs as inputs: copying one input could otherwise
        # corrupt another aliased source before all copies have completed.
        storage = {x.untyped_storage().data_ptr() for x in (*self.inputs, *self.outputs)}
        if any(x.untyped_storage().data_ptr() in storage for x in values):
            raise ValueError('Replay-owned storage cannot be reused as input')
        for target, source in zip(self.inputs, values):
            target.copy_(source)
        self.graph.replay()
        self.calls += 1
        return self.outputs
