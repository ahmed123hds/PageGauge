"""Attest the six-case synthetic generation smoke; not benchmark evidence."""
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT/'diagnostics'))
import mlsys_rtx5090_entry as base

RUN = ROOT/'results/mlsys2027_tasks_v1/synthetic_20260909T025247Z_a44bf775'


def main():
    inputs = {}
    def read(name):
        path = RUN/name; inputs[str(path)] = base.sha256_file(path)
        return json.loads(path.read_text())
    m = read('manifest.json'); r = read('analysis.json'); c = read('completion.json')
    f = read('fixtures.json'); native = read('hf_generate_validation.json'); telemetry = read('block_0_telemetry.json')
    if not r['execution_complete'] or r['cases'] != 6 or len(r['results']) != 6 or len(f['cases']) != 6:
        raise ValueError('Incomplete generation smoke')
    if c['return_code'] or not c['sampled_exclusivity_passed'] or not native['passed']:
        raise ValueError('Generation/monitor/native adapter validation failed')
    if c['telemetry_sha256'] != inputs[str(RUN/'block_0_telemetry.json')] or m['fixtures_sha256'] != inputs[str(RUN/'fixtures.json')]:
        raise ValueError('Changed generation inputs or telemetry')
    if native['manual_ids'] != native['official_generate_ids']:
        raise ValueError('Wrong native-generate comparison')
    if not any(c['pid'] in row.get('pids', []) for row in telemetry['observations']) or any(
        row.get('error') or row.get('unexpected_pids') or row.get('unexpected_windows_compute_pids') for row in telemetry['observations']):
        raise ValueError('Invalid sampled ownership evidence')
    for index, (case, row) in enumerate(zip(f['cases'], r['results'])):
        if row != read(f'case_{index}.json') or case['case_id'] != row['case_id']:
            raise ValueError('Changed/mismatched synthetic case')
        for arm in row['arms'].values():
            if not 1 <= len(arm['generated_ids']) <= m['max_new_tokens']:
                raise ValueError('Wrong output budget')
            matches = re.findall(r'(?<!\d)\d{6}(?!\d)', arm['generated_text'])
            expected = {'exact_stripped_match': arm['generated_text'].strip() == case['expected_code'],
                'first_six_digit_code_match': bool(matches) and matches[0] == case['expected_code']}
            if arm['scores'] != expected or not all(expected.values()):
                raise ValueError('The proposed all-correct smoke summary is not supported')
            if arm['generated_ids'] != row['arms']['hf']['generated_ids']:
                raise ValueError('The proposed exact-generation match is not supported')
        probes = row['arms']['page_gauge']['selected_execution_probes']
        if len(probes) != 6 or any(p['relative_l2'] > .005 or p['absolute_error'] > .02 for p in probes):
            raise ValueError('Missing or failed selected execution probes')
    output = HERE/'generated/synthetic_generation_summary.tex'
    output.write_text('As a generation-adapter smoke test, six synthetic Qwen3 archive-code '
        'retrieval prompts span 8,173 and 20,473 tokens with needle depths 10/50/90\\%. '
        'HF, FlashInfer and PageGauge each generate their own greedy answers; all six '
        'answers are exactly correct and all generated token sequences match HF. The '
        'first manual HF rollout matches native \\texttt{generate}, and all 36 selected '
        'PageGauge head-group execution checks pass. This small development smoke '
        'uses S4/A128/T768 and at most 32 output tokens; it is not a public benchmark '
        'score or evidence of generated tokens aging past T768.\n')
    base.atomic_json(HERE/'generated/synthetic_generation_evidence_manifest.json', {
        'input_sha256': inputs, 'builder_sha256': base.sha256_file(Path(__file__)),
        'output_sha256': {str(output): base.sha256_file(output)}, 'scope': m['scope']})


if __name__ == '__main__': main()
