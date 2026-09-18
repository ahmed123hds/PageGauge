"""Small random-model integration test, not pretrained-model quality evidence."""
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import sys
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path('/home/anonymous/pagegauge_baselines/KIVI')
sys.path.insert(0, str(ROOT/'diagnostics'))
import mlsys_rtx5090_entry as base


def main():
    import fcntl
    idle = base.idle_preflight(0)
    gpu = idle[-1]['uuid']
    with (Path('/tmp')/f'pagegauge-mlsys-{gpu}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        sys.path.insert(0,str(SOURCE))
        import torch
        import transformers
        from transformers import MistralConfig, MistralForCausalLM
        from models.mistral_kivi import MistralForCausalLM_KIVI
        out = ROOT/'results/mlsys2027_baselines_v1'/('kivi_model_smoke_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
        out.mkdir(parents=True)
        paths = [Path(__file__), SOURCE/'models/mistral_kivi.py', SOURCE/'quant/matmul.py', SOURCE/'quant/new_pack.py']
        base.atomic_json(out/'manifest.json', {'torch':torch.__version__, 'transformers':transformers.__version__,
            'source_sha256':{str(p):base.sha256_file(p) for p in paths},'idle':idle,
            'scope':'Random tiny GQA Mistral, B1/C128/D64, INT2/INT4, group/residual32; integration only'})
        print('KIVI model smoke: '+str(out), flush=True)
        try:
            torch.set_grad_enabled(False)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.manual_seed(2026090813)
            tokens = torch.randint(3,256,(1,192),device='cuda')
            rows = []
            for bits in (2,4):
                config = MistralConfig(vocab_size=256,hidden_size=512,intermediate_size=1024,
                    num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,
                    max_position_embeddings=512,sliding_window=None)
                config.k_bits = config.v_bits = bits
                config.group_size = config.residual_length = 32
                config.use_flash = False
                config._attn_implementation = 'eager'
                candidate = MistralForCausalLM_KIVI(config).eval().half().cuda()
                reference = MistralForCausalLM(config).eval().half().cuda()
                reference.load_state_dict(candidate.state_dict(),strict=True)
                with torch.inference_mode():
                    a = reference(tokens[:,:128],use_cache=True)
                    b = candidate(tokens[:,:128],use_cache=True)
                    # Prefill attention is FP16 in both arms; quantization only
                    # determines the returned compressed cache for later steps.
                    prefill_error = float((a.logits-b.logits).abs().max())
                    torch.testing.assert_close(a.logits,b.logits,rtol=.005,atol=.005)
                    past_a,past_b = a.past_key_values,b.past_key_values
                    agreement = 0
                    for position in range(128,192):
                        a = reference(tokens[:,position:position+1],past_key_values=past_a,use_cache=True)
                        b = candidate(tokens[:,position:position+1],past_key_values=past_b,use_cache=True)
                        if not bool(torch.isfinite(b.logits).all()):raise RuntimeError('Nonfinite native output')
                        past_a,past_b = a.past_key_values,b.past_key_values
                        agreement += int((a.logits[:,-1].argmax(-1)==b.logits[:,-1].argmax(-1)).sum())
                    for layer in past_b:
                        if layer[-1] != 192 or layer[0].dtype != torch.int32 or layer[4].dtype != torch.int32:
                            raise RuntimeError('Native quantized cache/length missing')
                        if layer[0].shape[-1]*(32//bits) != 192 or layer[1] is not None:
                            raise RuntimeError('Key residual finalization mismatch')
                        if layer[4].shape[-2] != 160 or layer[5].shape[-2] != 32:
                            raise RuntimeError('Value residual finalization mismatch')
                    row = {'bits':bits,'passed':True,'prefill_max_abs':prefill_error,
                        'descriptive_random_model_top1':agreement/64,
                        'cache_bytes':sum(t.numel()*t.element_size() for layer in past_b for t in layer if isinstance(t,torch.Tensor))}
                    rows.append(row)
                    print(json.dumps(row),flush=True)
                del candidate,reference,a,b,past_a,past_b
                gc.collect();torch.cuda.empty_cache()
            base.atomic_json(out/'analysis.json',{'passed':True,'rows':rows,
                'scope':'Random tiny-model integration only; no pretrained quality/performance claim'})
        except BaseException as error:
            base.atomic_json(out/'failure.json',{'passed':False,'error_type':type(error).__name__,
                'error':str(error),'traceback':traceback.format_exc()})
            raise


if __name__ == '__main__':main()
