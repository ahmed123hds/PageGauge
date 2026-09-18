"""Opt-in V/O diagonal reparameterization; all work happens before decoding.

Gains are shared across requests, fitted on request zero's prefill only.
Original HF oracle weights stay untouched until all oracle construction finishes.
"""
import os
import hashlib
import time
from pathlib import Path
import torch

MODE = 'folded_prefill_rms'
FIXED_MODE = 'folded_fixed_rms'


def enabled():
    mode = os.environ.get('PAGEGAUGE_VALUE_CONDITIONING', 'none')
    if mode not in ('none', MODE, FIXED_MODE):
        raise ValueError('Unknown PAGEGAUGE_VALUE_CONDITIONING mode: '+mode)
    return mode != 'none'


@torch.no_grad()
def condition_prefill(cache, layer, request, value):
    if not enabled():
        return value
    started = time.perf_counter()
    if not hasattr(cache, 'value_channel_gain'):
        cache.value_channel_gain = torch.ones((cache.value_center.shape[0], *value.shape[1:]), dtype=torch.float16, device=value.device)
        cache.value_conditioning_fitted = set()
        cache.value_conditioning_setup_seconds = 0.0
        if os.environ.get('PAGEGAUGE_VALUE_CONDITIONING') == FIXED_MODE:
            path = Path(os.environ['PAGEGAUGE_FIXED_GAIN_PATH'])
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != os.environ['PAGEGAUGE_FIXED_GAIN_SHA256']:
                raise ValueError('Fixed calibration artifact hash mismatch')
            import io
            artifact = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
            gain = artifact['gain']
            if (gain.shape != cache.value_channel_gain.shape or gain.dtype != torch.float16
                or not torch.isfinite(gain).all() or not (gain > 0).all()
                or not torch.equal(gain.float().log2(), gain.float().log2().round())
                or not ((gain >= 2.0**-8) & (gain <= 2.0**8)).all()):
                raise ValueError('Invalid fixed gain')
            cache.value_channel_gain.copy_(gain)
            cache.value_conditioning_fitted = set(range(len(gain)))
            cache.value_conditioning_artifact_sha256 = hashlib.sha256(raw).hexdigest()
    if os.environ.get('PAGEGAUGE_VALUE_CONDITIONING') == FIXED_MODE:
        pass  # Immutable calibration: no request may refit model-wide gains.
    elif request == 0:
        if layer in cache.value_conditioning_fitted:
            raise ValueError('Attempted gain refit')
        center = value.float().mean(0).half()
        residual = value.double()-center.double()
        rms = residual.square().mean(0).sqrt().clamp_min(2.0**-20)
        geom = rms.log().mean(-1,keepdim=True).exp()
        gain = torch.exp2(torch.round(torch.log2(rms/geom)).clamp(-8,8)).half()
        cache.value_channel_gain[layer].copy_(gain)
        cache.value_conditioning_fitted.add(layer)
    elif layer not in cache.value_conditioning_fitted:
        raise ValueError('Request zero prefill must fit gains before other requests')
    scaled = (value.float()/cache.value_channel_gain[layer].float()).half()
    if not torch.isfinite(scaled).all():
        raise ValueError('V conditioning overflow')
    if value.is_cuda:
        torch.cuda.synchronize(value.device)
    cache.value_conditioning_setup_seconds += time.perf_counter()-started
    return scaled


@torch.no_grad()
def fold_packed_weights(model, cache):
    if not enabled():
        if hasattr(cache, 'value_channel_gain'):
            raise ValueError('Conditioned cache cannot be consumed without folded weights')
        return None
    if not hasattr(cache, 'value_channel_gain') or len(cache.value_conditioning_fitted) != len(model.model.layers):
        raise ValueError('This opt-in path requires all layers populated by the exclusive cache builder')
    if getattr(model, '_pagegauge_value_weights_folded', False):
        raise ValueError('Refusing repeated weight folding')
    started = time.perf_counter()
    roundtrip_changed = 0
    for index,layer in enumerate(model.model.layers):
        attn = layer.self_attn
        gain = cache.value_channel_gain[index]
        q_width,kv_width = attn._pkv_q_width,attn._pkv_kv_width
        weight = attn.qkv_weight[q_width+kv_width:]
        if weight.shape[0] != gain.numel():
            raise ValueError('Unexpected packed V projection shape')
        transformed = (weight.float()/gain.flatten().float()[:,None]).half()
        expanded = gain.repeat_interleave(q_width//kv_width,dim=0).flatten()
        output = (attn.o_proj.weight.float()*expanded.float()[None]).half()
        if not torch.isfinite(transformed).all() or not torch.isfinite(output).all():
            raise ValueError('Projection conditioning overflow')
        roundtrip_changed += int(((transformed.float()*gain.flatten().float()[:,None]).half()!=weight).sum())
        roundtrip_changed += int(((output.float()/expanded.float()[None]).half()!=attn.o_proj.weight).sum())
        weight.copy_(transformed)
        attn.o_proj.weight.copy_(output)
        if getattr(attn,'qkv_bias',None) is not None:
            attn.qkv_bias[q_width+kv_width:].div_(gain.flatten())
    model._pagegauge_value_weights_folded = True
    if cache.value_channel_gain.is_cuda:
        torch.cuda.synchronize(cache.value_channel_gain.device)
    export = os.environ.get('PAGEGAUGE_EXPORT_GAIN_PATH')
    if export:
        with Path(export).open('xb') as stream:
            torch.save({'gain': cache.value_channel_gain.cpu(), 'source': 'development_initial_prefill'}, stream)
    fixed = os.environ.get('PAGEGAUGE_VALUE_CONDITIONING') == FIXED_MODE
    return {'mode': FIXED_MODE if fixed else MODE,
            'gain_source': 'frozen_calibration_artifact' if fixed else 'request_zero_initial_prefill_only_shared_across_batch',
            'calibration_artifact_sha256': getattr(cache, 'value_conditioning_artifact_sha256', None),
            'gain_sha256':hashlib.sha256(cache.value_channel_gain.cpu().numpy().tobytes()).hexdigest(),
            'gain_bytes':cache.value_channel_gain.numel()*cache.value_channel_gain.element_size(),
            'prefill_conditioning_seconds':cache.value_conditioning_setup_seconds,
            'weight_fold_seconds':time.perf_counter()-started,
            'fp16_weight_roundtrip_changed_elements':roundtrip_changed,
            'decode_extra_operations':0,
            'scope':'opt-in development reparameterization, original HF oracle, no general quality claim'}
