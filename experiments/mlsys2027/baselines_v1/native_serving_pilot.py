"""Native NSN/Kitty request-cost pilot with an own-stack HF control.

One method per fresh process; original completed TRAIN quality fixture only.
Native full-prefix processing, cache allocation/packing, and full recurrent
decode are measured separately. No PageGauge comparison or final speed CI.
"""
import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'diagnostics'))
import mlsys_rtx5090_entry as base
import mlsys_rtx5090_step02 as previous
from kivi_quality import verify


def contract(m, family, backend):
    allowed = {'nsn': ('hf', 'nsn_int2'), 'kitty': ('hf', 'kitty_pro')}
    if family not in allowed or backend not in allowed[family]:
        raise ValueError('Wrong native model/backend pairing')
    if (m['context'], m['decode_steps'], m['token_provenance']['split']) != (20480, 1536, 'train'):
        raise ValueError('Completed full recurrent TRAIN fixture required')


def worker(out):
    import torch
    import transformers
    m = json.loads((out/'manifest.json').read_text())
    verify(m)
    contract(m, m['family'], m['backend'])
    if os.environ.get('PYTORCH_ALLOC_CONF') != m['allocator_configuration'] or os.environ.get('PYTORCH_CUDA_ALLOC_CONF'):
        raise ValueError('Allocator configuration differs from manifest')
    if base.sha256_file(out/'tokens.json') != m['tokens_sha256']:
        raise ValueError('Changed tokens')
    torch.set_grad_enabled(False)
    torch.manual_seed(m['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(m['allocator_budget_bytes']/total)
    tokens = torch.tensor(json.loads((out/'tokens.json').read_text())['ids'], device='cuda', dtype=torch.long)
    context, steps = m['context'], m['decode_steps']
    if tokens.shape != (1, context+steps+1):
        raise ValueError('Pilot requires one unpadded request')
    kwargs = dict(torch_dtype=torch.float16, local_files_only=True,
                  low_cpu_mem_usage=True, device_map={'': 'cuda:0'}, attn_implementation='sdpa')
    load_start = time.perf_counter()
    make_cache = lambda: None
    if m['family'] == 'nsn':
        from nsn_quality import SOURCE, cache_bytes
        sys.path.insert(0, str(SOURCE))
        if m['backend'] == 'hf':
            from transformers import MistralForCausalLM
            model = MistralForCausalLM.from_pretrained(m['model'], **kwargs).eval()
        else:
            from src.models.mistral import QuantizedMistralForCausalLM
            from src.utils import rotate_v_proj, rotate_o_proj
            from src.quantizers.nsn_quantizer import NSNQuantizer
            quant = {'name': 'NSNQuantizer', 'kwargs': {'n_bits': 2,
                'codebook_path': str(SOURCE/'codebooks/2bit_codebook.pt'),
                'window_size': 64, 'residual_size': 64, 'hadamard': True}}
            model = QuantizedMistralForCausalLM.from_pretrained(m['model'],
                quant_config=quant, forward_quant=False, **kwargs).half().eval()
            released = NSNQuantizer(**quant['kwargs']).half().cuda()
            expected = dict(released.named_buffers())
            for layer in model.model.layers:
                for name, value in layer.self_attn.quantizer.named_buffers():
                    if not torch.equal(value, expected[name]):
                        raise ValueError('Released NSN codebook mismatch')
                rotate_v_proj(layer.self_attn.v_proj, 128)
                rotate_o_proj(layer.self_attn.o_proj, 128)
            del released, expected
        account = lambda past: cache_bytes(model, past)
        if model.config.sliding_window is not None or len(model.model.layers) != 32:
            raise ValueError('Wrong Mistral configuration')
    else:
        from kitty_quality import cache_bytes
        from transformers import Qwen3ForCausalLM
        from kitty.models.qwen3 import Qwen3ForCausalLM_Kitty
        from kitty.kvcache import get_kvcache_kitty
        cls = Qwen3ForCausalLM if m['backend'] == 'hf' else Qwen3ForCausalLM_Kitty
        model, loading = cls.from_pretrained(m['model'], output_loading_info=True, **kwargs)
        if any(loading.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
            raise ValueError('Checkpoint loading mismatch: '+str(loading))
        model.eval()
        if m['backend'] == 'kitty_pro':
            make_cache = lambda: get_kvcache_kitty(model.config, 1, context+steps)
        account = cache_bytes
        if (model.config.model_type, len(model.model.layers)) != ('qwen3', 36):
            raise ValueError('Wrong Qwen configuration')
    torch.cuda.synchronize()
    load_seconds = time.perf_counter()-load_start
    def memory():
        free, total_bytes = torch.cuda.mem_get_info()
        return dict(allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved(),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            device_used_bytes_at_boundary=total_bytes-free)
    rows = []
    with torch.inference_mode():
        for repeat in range(m['repeats']+1):
            print(f"Native {m['family']}/{m['backend']} request round {repeat}", flush=True)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            begin = time.perf_counter()
            start.record()
            past = make_cache()
            output = model.model(tokens[:, :context], past_key_values=past, use_cache=True)
            past = output.past_key_values
            # Only the final prompt logits are needed for serving, not C LM heads.
            prompt_logits = model.lm_head(output.last_hidden_state[:, -1:])
            del output
            end.record()
            torch.cuda.synchronize()
            prefill_ms = (time.perf_counter()-begin)*1000
            prefill_cuda_ms = start.elapsed_time(end)
            prefill_memory, initial_cache = memory(), account(past)
            if not torch.isfinite(prompt_logits).all() or past.get_seq_length() != context:
                raise ValueError('Invalid native prefill')
            del prompt_logits
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            begin = time.perf_counter()
            start.record()
            for position in range(context, context+steps):
                output = model(tokens[:, position:position+1], past_key_values=past, use_cache=True)
                past = output.past_key_values
            end.record()
            torch.cuda.synchronize()
            decode_ms = (time.perf_counter()-begin)*1000
            decode_cuda_ms = start.elapsed_time(end)
            decode_memory = memory()
            lengths = [past.get_seq_length(layer) for layer in range(len(model.model.layers))]
            if lengths != [context+steps]*len(model.model.layers) or not torch.isfinite(output.logits).all():
                raise ValueError('Incomplete/nonfinite recurrent decode')
            final_cache = account(past)
            row = dict(repeat=repeat, steps=steps, prefill_wall_ms=prefill_ms,
                prefill_cuda_ms=prefill_cuda_ms, decode_wall_ms_per_step=decode_ms/steps,
                decode_cuda_ms_per_step=decode_cuda_ms/steps,
                timed_segment_sum_ms=prefill_ms+decode_ms, decode_tokens_per_second=1000*steps/decode_ms,
                initial_cache=initial_cache, final_cache=final_cache, prefill_memory=prefill_memory,
                decode_memory=decode_memory, final_lengths=lengths)
            rows.append(row)
            base.atomic_json(out/'progress.json', {'rows': rows})
            del past, output
    verify(m)
    base.atomic_json(out/'analysis.json', dict(family=m['family'], backend=m['backend'],
        warmup=rows[0], rows=rows[1:], model_load_and_static_setup_seconds=load_seconds,
        torch=torch.__version__, transformers=transformers.__version__, scope=m['scope']))
    print('Native serving pilot complete: '+str(out), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--fixture', type=Path)
    parser.add_argument('--family', choices=('nsn', 'kitty'))
    parser.add_argument('--backend', choices=('hf', 'nsn_int2', 'kitty_pro'))
    args = parser.parse_args()
    if args.worker:
        try:
            worker(args.worker)
        except BaseException as error:
            base.atomic_json(args.worker/'failure.json', dict(type=type(error).__name__, error=str(error), traceback=traceback.format_exc()))
            raise
        return
    if args.fixture is None:
        raise ValueError('Completed native quality fixture required')
    source = args.fixture.resolve()
    original = json.loads((source/'manifest.json').read_text())
    completion = json.loads((source/'completion.json').read_text())
    contract(original, args.family, args.backend)
    if completion['return_code'] or not completion['sampled_exclusivity_passed']:
        raise ValueError('Incomplete quality fixture')
    verify(original)
    result = json.loads((source/'analysis.json').read_text())
    required_arm = 'int2' if args.family == 'nsn' else 'kitty_pro'
    for arm in ('hf', required_arm):
        if base.sha256_file(source/(arm+'.json')) != result['results'][arm]['sha256']:
            raise ValueError('Changed quality evidence')
    if base.sha256_file(source/'tokens.json') != original['tokens_sha256']:
        raise ValueError('Changed quality tokens')
    import fcntl
    idle = base.idle_preflight(0)
    gpu = idle[-1]['uuid']
    with (Path('/tmp')/f'pagegauge-mlsys-{gpu}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        out = ROOT/'results/mlsys2027_baselines_v1'/('native_serving_'+args.family+'_'+args.backend+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
        out.mkdir(parents=True)
        base.atomic_json(out/'tokens.json', json.loads((source/'tokens.json').read_text()))
        hashes = dict(original['source_sha256'])
        hashes[str(Path(__file__).resolve())] = base.sha256_file(Path(__file__))
        m = dict(original, family=args.family, backend=args.backend, source_sha256=hashes,
            seed=2026090831, repeats=3, allocator_budget_bytes=28*1024**3,
            allocator_configuration='expandable_segments:True', quality_fixture=str(source),
            quality_analysis_sha256=base.sha256_file(source/'analysis.json'),
            quality_completion_sha256=base.sha256_file(source/'completion.json'), idle=idle, orchestrator_pid=os.getpid(),
            scope='One fresh process per native method on a completed B1/C20480/D1536 TRAIN quality fixture; one full-request warmup and three repeats. Native cache allocation, packing and full-prefix SDPA plus last prompt LM head included in prefill. Resident-model load/static rotations excluded and reported. Decode includes all layers, recurrent cache updates and LM head, but teacher-forced inputs and no sampling or network/scheduler. Timed-segment sum excludes validation gaps and is not outer request latency. Own-stack HF control; not a matched PageGauge comparison, native-best optimization, maximum capacity, final TEST or replicated CI.')
        base.atomic_json(out/'manifest.json', m)
        os.environ.pop('PYTORCH_CUDA_ALLOC_CONF', None)
        os.environ['PYTORCH_ALLOC_CONF'] = m['allocator_configuration']
        python = '/home/anonymous/pagegauge_baselines/'+('nsn_sm120_env' if args.family == 'nsn' else 'kitty_sm120_env')+'/bin/python'
        command = [python, '-u', str(Path(__file__)), '--worker', str(out)]
        base.atomic_json(out/'invocation.json', {'command': command})
        print('Native serving output: '+str(out), flush=True)
        completed = previous.run_process(command, out, {'index': 0}, gpu)
        base.atomic_json(out/'completion.json', completed)
        verify(m)
        if completed['return_code'] or not completed['sampled_exclusivity_passed']:
            raise RuntimeError('Native serving pilot failed; preserve exact evidence')


if __name__ == '__main__':
    main()
