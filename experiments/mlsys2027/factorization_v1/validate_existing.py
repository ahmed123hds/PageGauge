"""CPU integration check of new validator against an existing production worker.

In-memory experiment labels/config stand in for the adapter. This validates
schema compatibility only, never constitutes a new E1 measurement.
"""
import json
from full_model import ROOT, assess, schedule, base


def main():
    old = ROOT/'results/mlsys2027_rtx5090/02_fp16_performance/20260908T041311Z_076aaa6c'
    manifest = json.loads((old/'manifest.json').read_text())
    p = json.loads((old/'block_1_page_gauge.json').read_text())
    completion = json.loads((old/'block_1_completion.json').read_text())
    adapter = 'experiments/mlsys2027/factorization_v1/full_model_worker.py'
    manifest['source_sha256'][adapter] = base.sha256_file(ROOT/adapter)
    p['factorization_experiment'] = {'arm':'factorized', 'adapter_sha256':manifest['source_sha256'][adapter]}
    p['configuration']['min_logits_cosine'] = -1.0
    p['configuration']['value_conditioning_mode'] = 'none'
    result = assess(p, schedule()[1], manifest, completion)
    assert result['execution_passed']
    print('Existing production schema validates; CPU-only check, no new performance result.')


if __name__ == '__main__':
    main()
