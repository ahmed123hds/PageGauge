"""Publish only the complete, execution-validated four-policy pilot."""
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT/'experiments/mlsys2027/ablation_v1'))
from regional_cost import assess, reduce_results, previous

RUN = ROOT/'results/mlsys2027_ablation_v1/regional_cost_20260909T025538Z_20682b18'


def validate(out):
    inputs = {}
    def read(name):
        path = out/name
        inputs[str(path)] = previous.entry.sha256_file(path)
        return json.loads(path.read_text())
    m = read('manifest.json'); r = read('analysis.json')
    if r['status'] != 'complete' or not r['execution_passed']:
        raise ValueError('Partial or failed regional cost suite')
    payloads = []
    for block in m['schedule']:
        index = block['index']
        p = read(f'block_{index}.json'); c = read(f'block_{index}_completion.json')
        a = read(f'block_{index}_assessment.json'); t = read(f'block_{index}_telemetry.json')
        if a['result_sha256'] != inputs[str(out/f'block_{index}.json')]:
            raise ValueError('Changed timed payload')
        if c['telemetry_sha256'] != inputs[str(out/f'block_{index}_telemetry.json')]:
            raise ValueError('Changed monitoring evidence')
        log = out/f'block_{index}.log'; inputs[str(log)] = previous.entry.sha256_file(log)
        if c['log_sha256'] != inputs[str(log)]: raise ValueError('Changed worker log')
        contract = m['process_monitor']['contract']
        observations = t['observations']
        if c['monitor_contract'] != contract or not observations:
            raise ValueError('Missing explicit monitoring contract')
        if any(s.get('monitor_contract') != contract or s.get('error') or s.get('unexpected_pids')
               or s.get('unexpected_windows_compute_pids') for s in observations):
            raise ValueError('Invalid process samples')
        if not any(c['pid'] in s['pids'] for s in observations):
            raise ValueError('Own process never observed')
        checks = assess(p, block, m, c)
        if not checks['execution_passed']: raise ValueError('Execution failure')
        payloads.append(p)
    rebuilt = reduce_results(payloads, m['schedule'])
    if rebuilt['rows'] != r['rows']: raise ValueError('Pilot reduction does not reproduce')
    return r, inputs


def main():
    r, inputs = validate(RUN)
    rows = r['rows']; fp16_bytes = previous.expected_cache('flashinfer_fp16')['served_bytes']
    labels = (('reference_policy', 'S4/A128/T768'), ('without_prefix', 'S0/A128/T768'),
        ('without_static_suffix', 'S4/A0/T768'), ('minimum_page_tail', 'S4/A128/T16'))
    lines = []
    for key, label in labels:
        row = rows[key]
        cells = []
        for mode in ('cache_neutral', 'cache_hot'):
            value = row[mode]['wall_ms']; lo, hi = value['range_ms_per_step']
            if not all(math.isfinite(x) and x > 0 for x in (lo, hi, value['median_ms_per_step'])):
                raise ValueError('Invalid latency')
            cells.append(f'{value["median_ms_per_step"]:.3f} [{lo:.3f}, {hi:.3f}]')
        lines.append(f'{label} & {100*(1-row["served_bytes"]/fp16_bytes):.2f}\\% & '+
                     ' & '.join(cells)+' \\\\')
    table = HERE/'generated/regional_cost_rows.tex'
    table.write_text('\\begin{tabular}{lrrr}\n\\toprule\nPolicy & KV reduction & Neutral ms/step [range] & Hot ms/step [range]\\\\\\midrule\n'+
        '\n'.join(lines)+'\n\\bottomrule\n\\end{tabular}\n')
    original, candidate = rows['reference_policy'], rows['without_static_suffix']
    neutral = original['cache_neutral']['wall_ms']['median_ms_per_step']/candidate['cache_neutral']['wall_ms']['median_ms_per_step']
    hot = original['cache_hot']['wall_ms']['median_ms_per_step']/candidate['cache_hot']['wall_ms']['median_ms_per_step']
    summary = HERE/'generated/regional_cost_summary.tex'
    summary.write_text(f'The clean four-policy pilot measures reference/A0 median ratios of {neutral:.4f} '
        f'(cache-neutral) and {hot:.4f} (cache-hot), with unchanged quantizer and kernels '
        '(Table~\\ref{tab:regional-cost}). These are fixed-order, one-process-per-policy '
        'observations, not confidence intervals or a new FlashInfer speedup. Balanced '
        'fresh-process replication is required before promotion.\n')
    previous.entry.atomic_json(HERE/'generated/regional_cost_evidence_manifest.json', {
        'input_sha256': inputs, 'builder_sha256': previous.entry.sha256_file(Path(__file__)),
        'output_sha256': {str(p): previous.entry.sha256_file(p) for p in (table, summary)},
        'scope': r['scope']})
    print('Validated all four regional timing policies, telemetry, accounting and reduction.', flush=True)


if __name__ == '__main__': main()
