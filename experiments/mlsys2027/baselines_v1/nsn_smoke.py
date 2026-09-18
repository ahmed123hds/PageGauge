"""NSN native INT2 recurrence against independently unpacked same-cache attention.

Use nsn_sm120_env after its build/import succeeds. No speed or model-quality claim.
"""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path('/home/anonymous/pagegauge_baselines/NSNQuant')
sys.path.insert(0, str(ROOT/'diagnostics'))
import mlsys_rtx5090_entry as base


def hadamard_reference(x):
    """Independent FP32 butterfly, normalized; does not call the native extension."""
    shape = x.shape
    out = x.float()
    width = 1
    while width < shape[-1]:
        pairs = out.reshape(*shape[:-1], -1, 2, width)
        a, b = pairs[..., 0, :], pairs[..., 1, :]
        import torch
        out = torch.stack((a+b, a-b), dim=-2).reshape(shape)
        width *= 2
    return out/math.sqrt(shape[-1])


def unpack4(packed, scale, offset):
    import torch
    values = torch.stack([((packed >> (4*i)) & 15) for i in range(8)], -1)
    values = values.reshape(*scale.shape, -1).float()
    # Match upstream's two separately rounded half operations for stored metadata.
    values = (values*scale.float().unsqueeze(-1)).half()
    values = (values+offset.unsqueeze(-1)).half()
    return values.reshape(*packed.shape[:-1], packed.shape[-1]*8).float()


def code_vectors(packed, codebook):
    import torch
    indices = torch.stack((packed & 255, (packed >> 8) & 255), -1).long()
    values = codebook[indices].reshape(*packed.shape[:-1], -1)
    signs = torch.stack([((packed >> (16+i)) & 1) for i in range(16)], -1)
    signs = signs.reshape_as(values)
    return values*(signs.float()*2-1)


def rotary(x, cos, sin):
    import torch
    half = x.shape[-1]//2
    rotated = torch.cat((-x[..., half:], x[..., :half]), -1)
    return x.float()*cos.float().unsqueeze(1)+rotated.float()*sin.float().unsqueeze(1)


def reconstruct(quantizer, cache, cos, sin):
    import torch
    cell = cache[0]
    codebook = unpack4(quantizer.codebook_idx, quantizer.codebook_scale, quantizer.codebook_offset)
    packed_len = cell['quantized_key_cache'].shape[-2]
    restored = {}
    for prefix in ('key', 'value'):
        codes = code_vectors(cell[f'quantized_{prefix}_cache'], codebook)
        norm = unpack4(cell[prefix+'_norm_idx'], cell[prefix+'_norm_scale'], cell[prefix+'_norm_offset'])
        norm = norm.reshape(*codes.shape[:-1], 1)
        mean = unpack4(cell[prefix+'_mean_idx'], cell[prefix+'_mean_scale'], cell[prefix+'_mean_offset'])
        mean = mean.expand(*mean.shape[:-2], 64, mean.shape[-1]).reshape_as(codes)
        residual = codes*cell[prefix+'_norm2'].float()
        if prefix == 'key':
            residual = hadamard_reference(residual)
            mean = rotary(mean, cos[:, :packed_len], sin[:, :packed_len])
        values = norm*(residual+mean)
        exact = cell[f'full_{prefix}_cache']
        if len(exact):
            tail = rotary(exact, cell['cos'], cell['sin']) if prefix == 'key' else exact.float()
            values = torch.cat((values, tail), -2)
        restored[prefix] = values.repeat_interleave(4, 1)
    return restored['key'], restored['value']


def main():
    import fcntl
    expected = '604db3ca34e8de7026b404048eca58b894769701'
    if subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip() != expected:
        raise RuntimeError('NSN source revision changed')
    dirty = subprocess.check_output(['git', '-C', str(SOURCE), 'diff', '--ignore-submodules=all', '--name-only', 'HEAD'], text=True)
    if dirty.strip():
        raise RuntimeError('Undisclosed NSN source changes: '+dirty)
    idle = base.idle_preflight(0)
    gpu = idle[-1]['uuid']
    with (Path('/tmp')/f'pagegauge-mlsys-{gpu}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        sys.path.insert(0, str(SOURCE))
        import torch
        import nsn_tools
        import fast_hadamard_transform_cuda
        from src.quantizers.nsn_quantizer import NSNQuantizer
        from src.models.cache import NSNCache
        torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.manual_seed(2026090820)
        out = ROOT/'results/mlsys2027_baselines_v1'/('nsn_smoke_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
        out.mkdir(parents=True)
        source_paths = [Path(__file__), Path(nsn_tools.__file__), Path(fast_hadamard_transform_cuda.__file__),
                        SOURCE/'codebooks/2bit_codebook.pt', SOURCE/'3rdparty/fast-hadamard-transform/setup.py']
        source_paths += sorted((SOURCE/'src').rglob('*.py'))+sorted((SOURCE/'src/csrc').rglob('*.cu'))
        hashes = {str(p): base.sha256_file(p) for p in source_paths}
        manifest = {'source_commit': expected, 'source_sha256': hashes, 'idle': idle,
                    'cases': {'batch': [1, 4], 'prefill': 255, 'append': 129, 'hq': 32, 'hkv': 8,
                              'head_dim': 128, 'bits': 2, 'window': 64, 'residual': 64},
                    'execution_tolerances': {'relative_l2': .005, 'absolute': .02},
                    'scope': 'Independent same-quantized-cache native execution check; no model quality or timing claim',
                    'portability': 'SM120 arch-only Hadamard setup.py override; modern Torch, upstream NSN numerical code unchanged'}
        base.atomic_json(out/'manifest.json', manifest)
        print('NSN native smoke: '+str(out), flush=True)
        rows = []
        for batch in (1, 4):
            quantizer = NSNQuantizer(2, str(SOURCE/'codebooks/2bit_codebook.pt'), 64, 64, True).cuda().half()
            cache = NSNCache()
            total = 384
            inv = (1/(1000000.0**(torch.arange(0, 128, 2, device='cuda').float()/128))).half()
            angles = torch.arange(total, device='cuda').float().unsqueeze(-1)*inv.float()
            angles = torch.cat((angles, angles), -1)
            cos = angles.cos().half().unsqueeze(0).expand(batch, -1, -1).contiguous()
            sin = angles.sin().half().unsqueeze(0).expand(batch, -1, -1).contiguous()
            keys = torch.randn(batch, 8, total, 128, device='cuda', dtype=torch.float16)
            values = torch.randn_like(keys)
            cache.update(keys[:, :, :255].contiguous(), values[:, :, :255].contiguous(), 0,
                         {'cos': cos[:, :255].contiguous(), 'sin': sin[:, :255].contiguous()})
            quantizer.update_cache(cache, 0, True)
            observed = set()
            maximum_l2 = maximum_abs = 0.0
            for pos in range(255, total):
                cache.update(keys[:, :, pos:pos+1].contiguous(), values[:, :, pos:pos+1].contiguous(), 0,
                             {'cos': cos[:, pos:pos+1].contiguous(), 'sin': sin[:, pos:pos+1].contiguous()})
                query = torch.randn(batch, 32, 1, 128, device='cuda', dtype=torch.float16)
                query = rotary(query, cos[:, pos:pos+1], sin[:, pos:pos+1]).half()
                actual, probabilities = quantizer.self_attn(query, cache, 0, None, 1/math.sqrt(128), 4,
                                                             inv_freq=inv, offset=torch.zeros(batch, device='cuda', dtype=torch.int32))
                restored_k, restored_v = reconstruct(quantizer, cache, cos, sin)
                reference_p = ((query.float() @ restored_k.transpose(-2, -1))/math.sqrt(128)).softmax(-1)
                reference = (reference_p @ restored_v).transpose(1, 2)
                for observed_tensor, target in ((actual, reference), (probabilities, reference_p)):
                    diff = observed_tensor.float()-target
                    l2 = float(diff.norm()/target.norm().clamp_min(1e-12))
                    absolute = float(diff.abs().max())
                    maximum_l2, maximum_abs = max(maximum_l2, l2), max(maximum_abs, absolute)
                    if not torch.isfinite(observed_tensor).all() or l2 > .005 or absolute > .02:
                        base.atomic_json(out/'failure.json', {'batch': batch, 'position': pos, 'l2': l2, 'abs': absolute})
                        raise RuntimeError('Native NSN differs from independently reconstructed cache')
                observed.add(cache[0]['quantized_key_cache'].shape[-2])
                quantizer.update_cache(cache, 0, False)
                if cache.get_seq_length() != pos+1:
                    raise RuntimeError('Recurrent length mismatch')
            if observed != {192, 256, 320} or cache[0]['quantized_key_cache'].shape[-2] != 384:
                raise RuntimeError('Missing finalized-page consumption')
            rows.append({'batch': batch, 'max_relative_l2': maximum_l2, 'max_abs': maximum_abs,
                         'consumed_packed_lengths': sorted(observed), 'final_packed_length': 384, 'passed': True})
            print(json.dumps(rows[-1]), flush=True)
        if any(base.sha256_file(Path(p)) != digest for p, digest in hashes.items()):
            raise RuntimeError('Source changed during smoke')
        base.atomic_json(out/'analysis.json', {'passed': True, 'rows': rows, 'scope': manifest['scope']})


if __name__ == '__main__':
    main()
