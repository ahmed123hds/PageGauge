"""Small native Mistral integration and FP16 V/O rotation control. No PPL claim."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path('/home/anonymous/pagegauge_baselines/NSNQuant')
sys.path.insert(0, str(ROOT/'diagnostics'))
import mlsys_rtx5090_entry as base


def main():
    import fcntl
    idle = base.idle_preflight(0)
    gpu = idle[-1]['uuid']
    with (Path('/tmp')/f'pagegauge-mlsys-{gpu}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        import torch
        from transformers import MistralConfig, MistralForCausalLM
        sys.path.insert(0, str(SOURCE))
        from src.models.mistral import QuantizedMistralForCausalLM
        from src.utils import rotate_v_proj, rotate_o_proj
        torch.manual_seed(2026090822)
        torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = False
        config = MistralConfig(vocab_size=256, hidden_size=1024, intermediate_size=1024,
            num_hidden_layers=2, num_attention_heads=8, num_key_value_heads=2,
            max_position_embeddings=512, sliding_window=None, rope_theta=1000000.)
        config._attn_implementation = 'sdpa'
        out = ROOT/'results/mlsys2027_baselines_v1'/('nsn_model_smoke_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
        out.mkdir(parents=True)
        base.atomic_json(out/'manifest.json', {'config': config.to_dict(), 'seed': 2026090822,
            'prefill': 255, 'decode': 129, 'idle': idle,
            'source_sha256': {str(p): base.sha256_file(p) for p in
                              [Path(__file__), SOURCE/'src/models/mistral.py', SOURCE/'src/utils.py',
                               SOURCE/'src/quantizers/nsn_quantizer.py', SOURCE/'codebooks/2bit_codebook.pt']},
            'identity_rotation_tolerances': {'relative_l2': .005, 'max_abs': .02},
            'scope': 'Random tiny model integration/control only. NSN fidelity descriptive, not a quality acceptance test.'})
        print('NSN model smoke: '+str(out), flush=True)
        original = MistralForCausalLM(config).eval().half().cuda()
        QuantizedMistralForCausalLM.forward_quant = False
        QuantizedMistralForCausalLM.quant_config = {'name': 'Identity', 'kwargs': None}
        control = QuantizedMistralForCausalLM(config).eval().half().cuda()
        control.load_state_dict(original.state_dict(), strict=True)
        QuantizedMistralForCausalLM.quant_config = {'name': 'NSNQuantizer', 'kwargs': {
            'n_bits': 2, 'codebook_path': str(SOURCE/'codebooks/2bit_codebook.pt'),
            'window_size': 64, 'residual_size': 64, 'hadamard': True}}
        native = QuantizedMistralForCausalLM(config).eval().half().cuda()
        missing, unexpected = native.load_state_dict(original.state_dict(), strict=False)
        expected = {f'model.layers.{i}.self_attn.quantizer.{name}' for i in range(2)
                    for name in ('codebook', 'codebook_idx', 'codebook_scale', 'codebook_offset')}
        if set(missing) != expected or unexpected:
            raise RuntimeError('Unexpected weight-loading mismatch')
        for model in (control, native):
            for layer in model.model.layers:
                rotate_v_proj(layer.self_attn.v_proj, 128)
                rotate_o_proj(layer.self_attn.o_proj, 128)
        tokens = torch.randint(0, 256, (1, 384), device='cuda')
        rows = {}
        observed = {}
        with torch.inference_mode():
            for name, model in [('hf', original), ('rotated_identity', control), ('nsn', native)]:
                output = model(tokens[:, :255], use_cache=True)
                past = output.past_key_values
                logits = []
                packed_lengths = set()
                for position in range(255, 384):
                    if name == 'nsn':
                        packed_lengths.add(past[0]['quantized_key_cache'].shape[-2])
                    output = model(tokens[:, position:position+1], past_key_values=past, use_cache=True)
                    past = output.past_key_values
                    logits.append(output.logits[:, -1].float())
                    if past.get_seq_length() != position+1:
                        raise RuntimeError('Native model cache length mismatch')
                observed[name] = torch.stack(logits)
                if not bool(observed[name].isfinite().all()):
                    raise RuntimeError('Nonfinite model logits')
                if name == 'nsn':
                    if packed_lengths != {192, 256, 320}:
                        raise RuntimeError('Missing new packed history consumption')
                    if any(past[i]['quantized_key_cache'].shape[-2] != 384 for i in range(2)):
                        raise RuntimeError('Incomplete finalization')
                    rows[name] = {'consumed_packed_lengths': sorted(packed_lengths), 'final_length': 384}
            for name in ('rotated_identity', 'nsn'):
                diff = observed[name]-observed['hf']
                l2 = float(diff.norm()/observed['hf'].norm())
                absolute = float(diff.abs().max())
                rows.setdefault(name, {}).update(relative_l2_to_hf=l2, max_abs_to_hf=absolute,
                    top1_to_hf=float((observed[name].argmax(-1)==observed['hf'].argmax(-1)).float().mean()))
            passed = rows['rotated_identity']['relative_l2_to_hf'] <= .005 and rows['rotated_identity']['max_abs_to_hf'] <= .02
            result = {'passed': passed, 'rows': rows,
                'scope': 'Native model recurrence plus unquantized rotation control; NSN-vs-HF differences descriptive only.'}
            base.atomic_json(out/'analysis.json', result)
            print(json.dumps(result, indent=2), flush=True)
            if not passed:
                raise SystemExit(2)


if __name__ == '__main__':
    main()
