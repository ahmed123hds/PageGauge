"""One deterministic failed-cache replay; separate native QK and SV differences."""
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import uuid

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'diagnostics'))
import mlsys_rtx5090_entry as base


def main():
    idle = base.idle_preflight(0)
    import torch
    from kitty.kvcache import get_kvcache_kitty
    from kitty_reference import reconstruct
    native = importlib.import_module('kitty.kvcache.kernels.kitty_attention')
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    out = ROOT/'results/mlsys2027_baselines_v1'/('kitty_diagnostic_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True)
    base.atomic_json(out/'manifest.json', {'idle': idle, 'seed': 2026090824,
        'source_sha256': base.sha256_file(Path(__file__)),
        'scope': 'Single failed synthetic-cache replay, instrumentation only; no baseline performance or quality result'})
    generator = torch.Generator(device='cuda').manual_seed(2026090824)
    cfg = SimpleNamespace(num_hidden_layers=1, head_dim=128, num_key_value_heads=8)
    cache = get_kvcache_kitty(cfg, 1, 768)
    shape = (1, 8, 768, 128)
    key = (torch.randn(shape, generator=generator, device='cuda', dtype=torch.float16)*.08).contiguous()
    value = (torch.randn(shape, generator=generator, device='cuda', dtype=torch.float16)*.08).contiguous()
    cache.update(key[:, :, :511], value[:, :, :511], 0)
    cache.quantize_prefill(0)
    cache.update(key[:, :, 511:512], value[:, :, 511:512], 0)
    query = (torch.randn((1, 32, 1, 128), generator=generator, device='cuda', dtype=torch.float16)*.35).contiguous()
    captured = {}
    original = native.qk_kernel
    class CaptureScores:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                result = original[grid](*args, **kwargs)
                captured['scores'] = args[original.arg_names.index('output_ptr')].clone()
                return result
            return launch
    native.qk_kernel = CaptureScores()
    try:
        output, _ = native.kitty_attention_forward(SimpleNamespace(num_attention_heads=32, num_key_value_heads=8), query, cache.kv_cache[0], 128**-.5)
    finally:
        native.qk_kernel = original
    k, v = reconstruct(cache.kv_cache[0])
    k, v = k.repeat_interleave(4, 1), v.repeat_interleave(4, 1)
    scores = (query.float() @ k.float().transpose(-1, -2)/128**.5).half().squeeze(-2)
    probabilities = captured['scores'].float().softmax(-1).half()
    native_scores = captured['scores']
    sv_reference = (probabilities.float().unsqueeze(-2) @ v.float()).half().transpose(1, 2)
    reference = (scores.float().softmax(-1).half().float().unsqueeze(-2) @ v.float()).half().transpose(1, 2)
    # One-hot queries recover the native kernel's effective reconstructed K
    # exactly (unit scale), without creating or modifying a native kernel.
    native_k = torch.empty(1, 8, 512, 128, device='cuda', dtype=torch.float16)
    native.qk_kernel = CaptureScores()
    try:
        for begin in range(0, 128, 4):
            basis = torch.zeros_like(query)
            for head in range(8):
                for group in range(4):
                    basis[0, head*4+group, 0, begin+group] = 1
            native.kitty_attention_forward(SimpleNamespace(num_attention_heads=32, num_key_value_heads=8), basis, cache.kv_cache[0], 1.0)
            native_k[..., begin:begin+4] = captured['scores'].reshape(1, 8, 4, 512).transpose(-1, -2)
    finally:
        native.qk_kernel = original
    def metrics(left, right):
        left, right = left.float(), right.float()
        return {'max_abs': float((left-right).abs().max()),
            'relative_l2': float(torch.linalg.vector_norm(left-right)/torch.linalg.vector_norm(right).clamp_min(1e-12))}
    result = {'qk': metrics(native_scores, scores), 'sv_with_native_probabilities': metrics(output, sv_reference),
        'full': metrics(output, reference),
        'native_basis_keys_vs_oracle': metrics(native_k, k[:, ::4]),
        'basis_regions': {name: metrics(native_k[..., a:b, :], k[:, ::4, a:b])
                       for name, a, b in [('sink', 0, 32), ('packed', 32, 416), ('residual', 416, 512)]}}
    base.atomic_json(out/'analysis.json', result)
    torch.save({'native_output': output.cpu(), 'oracle_output': reference.cpu(),
        'native_scores': native_scores.cpu(), 'oracle_scores': scores.cpu(),
        'query': query.cpu(), 'reconstructed_k': k.cpu(), 'reconstructed_v': v.cpu(),
        'basis_native_k': native_k.cpu(), 'key_cache': cache.kv_cache[0].KeyCache.cpu(),
        'key_metadata': cache.kv_cache[0].KeyCache_metadata.cpu()}, out/'diagnostic_tensors.pt')
    print('Kitty diagnostic: '+str(out), flush=True)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
