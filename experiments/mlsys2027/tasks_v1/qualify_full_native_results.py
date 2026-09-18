"""Audit and decode the completed synthetic native cohort, no task scores."""
import copy
import hashlib
import json
from pathlib import Path
from transformers import AutoTokenizer
from full_native_result_adapter import normalize

ROOT = Path(__file__).resolve().parents[3]


def main():
    names = {('nsn', 'hf'): 'nsn_hf_cohort_20260910T094041Z_7d48c1d6',
             ('nsn', 'nsn_int2'): 'nsn_nsn_int2_cohort_20260910T094104Z_045085c9',
             ('kitty', 'hf'): 'kitty_hf_cohort_20260910T094120Z_24742355',
             ('kitty', 'kitty_pro'): 'kitty_kitty_pro_cohort_20260910T094135Z_2ebcd357'}
    rows, hashes = [], {}
    for (family, backend), name in names.items():
        directory = ROOT/'results/mlsys2027_tasks_v1'/name
        data = {key: json.loads((directory/(key+'.json')).read_text())
                for key in ('analysis', 'manifest', 'fixtures', 'completion')}
        if data['completion']['return_code'] or not data['completion']['sampled_exclusivity_passed']:
            raise ValueError('Incomplete/exclusivity failure')
        if data['analysis']['model_loads'] != 1:
            raise ValueError('Unexpected model reload')
        tokenizer = AutoTokenizer.from_pretrained(data['manifest']['model'], local_files_only=True, trust_remote_code=False)
        eos = [2] if family == 'nsn' else [151645]
        normalized = normalize(data['analysis'], data['fixtures'], backend, family, tokenizer, 32768, eos)
        if len(normalized) != 2 or sum(r['fallback'] for r in normalized) != 0:
            raise ValueError('Synthetic coverage mismatch')
        bad = copy.deepcopy(data['analysis'])
        bad['results'][1]['result']['executed_backend'] = 'not_native_hf'
        try:
            normalize(bad, data['fixtures'], backend, family, tokenizer, 32768, eos)
        except ValueError:
            pass
        else:
            raise AssertionError('Mislabeled fallback accepted')
        rows.extend(dict(row, family=family) for row in normalized)
        for key in data:
            path = directory/(key+'.json')
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = {'passed': True, 'rows': rows, 'input_sha256': hashes,
              'scope': 'Synthetic serialization/decoding/fallback qualification, not benchmark accuracy.'}
    target = ROOT/'results/mlsys2027_tasks_v1/full_native_cohort_qualification.json'
    with target.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
