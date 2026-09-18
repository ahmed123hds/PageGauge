"""Resident-model native-stack quality worker; explicit frozen prompt IDs only."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import traceback
from synthetic_generation import base, verify
from task_contract import prepare, validate_result
from reusable_lowbit_generation import generate as lowbit_generate
from native_hf_generation import generate as hf_generate


def worker(out):
    import torch
    import transformers
    from transformers import MistralForCausalLM
    m = json.loads((out/'manifest.json').read_text())
    verify(m)
    if transformers.__version__ != '4.36.2':
        raise ValueError('Pinned native stack required')
    if base.sha256_file(out/'fixtures.json') != m['fixtures_sha256']:
        raise ValueError('Changed frozen fixtures')
    cases = json.loads((out/'fixtures.json').read_text())['cases']
    if not cases or len({c['case_id'] for c in cases}) != len(cases):
        raise ValueError('Nonempty unique cases required')
    backend = m['task_backend']
    choices = {'hf': None, 'kivi_int2': ('kivi', 2), 'kivi_int4': ('kivi', 4), 'bitdecode_int4': ('bitdecode', 4)}
    if backend not in choices:
        raise ValueError('Unsupported task backend')
    torch.manual_seed(m['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    model = MistralForCausalLM.from_pretrained(m['model'], torch_dtype=torch.float16,
        local_files_only=True, low_cpu_mem_usage=True, device_map={'': 'cuda:0'}, attn_implementation='eager').eval()
    eos = model.config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    results = []
    for case in cases:
        contract = prepare(case['task'], case['prompt_ids'], model.config.max_position_embeddings)
        if contract.fp16_fallback:
            ids = torch.tensor([contract.prompt_ids], dtype=torch.long, device='cuda')
            with torch.inference_mode():
                generated = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False,
                    max_new_tokens=contract.max_new_tokens, eos_token_id=eos, pad_token_id=eos[0], use_cache=True)
            generated = generated[0, ids.shape[1]:].tolist()
            result = {'generated_ids': generated, 'stop_reason': 'eos' if generated[-1] in eos else 'max_new_tokens',
                      'fallback': True, 'executed_backend': 'native_hf_fp16', 'quantized_tokens_served': 0}
        elif backend == 'hf':
            result = hf_generate(model, contract, eos)
        else:
            family, bits = choices[backend]
            result = lowbit_generate(model, contract, family, bits, eos)
        validate_result(contract, result['generated_ids'], result['stop_reason'], eos)
        # Decode centrally with the pinned modern tokenizer: older tokenizers
        # cannot necessarily load this instruction checkpoint's tokenizer JSON.
        row = {'case_id': case['case_id'], 'task': case['task'], 'requested_backend': backend,
               'prompt_contract': asdict(contract), 'result': result}
        results.append(row)
        base.atomic_json(out/'progress.json', {'results': results, 'execution_complete': False})
    verify(m)
    base.atomic_json(out/'analysis.json', {'results': results, 'execution_complete': True,
        'model_loads': 1, 'transformers': transformers.__version__,
        'scope': 'Own-generation quality worker; native-stack HF control required. No timing claim.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--worker', type=Path, required=True)
    args = parser.parse_args()
    try:
        worker(args.worker)
    except BaseException:
        base.atomic_json(args.worker/'failure.json', {'traceback': traceback.format_exc()})
        raise
