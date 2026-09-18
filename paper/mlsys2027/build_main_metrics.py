"""Validate and render the original optimized FI/PG development contrast."""
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT/'diagnostics'))
import mlsys_rtx5090_step02 as protocol


def main():
    out = ROOT/'results/mlsys2027_rtx5090/02_fp16_performance/20260908T041311Z_076aaa6c'
    result = json.loads((out/'analysis.json').read_text())
    manifest = json.loads((out/'manifest.json').read_text())
    if not result['execution_gates_passed'] or result['fresh_process_blocks'] != 8:
        raise ValueError('Incomplete original optimized contrast')
    inputs = {str(p): protocol.entry.sha256_file(p) for p in (out/'analysis.json', out/'manifest.json')}
    payloads = []
    for block, record in zip(manifest['schedule'], result['processes']):
        if record['block'] != block or not record['sampled_exclusivity_passed']:
            raise ValueError('Changed block/exclusivity')
        path = out/f"block_{block['index']}_{block['backend']}.json"
        digest = protocol.entry.sha256_file(path)
        if digest != record['result_sha256']: raise ValueError('Changed original timing payload')
        inputs[str(path)] = digest
        completion = out/f"block_{block['index']}_completion.json"
        inputs[str(completion)] = protocol.entry.sha256_file(completion)
        c = json.loads(completion.read_text())
        if c['return_code'] != record['return_code'] or not c['sampled_exclusivity_passed']:
            raise ValueError('Changed completion/exclusivity')
        payload = json.loads(path.read_text())
        protocol.validate_worker(payload, block, manifest, record['return_code'])
        payloads.append(payload)
    actual = protocol.reduce_results(payloads, manifest)
    if actual['endpoints'] != result['endpoints']:
        raise ValueError('Original timing reduction does not reproduce')
    lines = []
    for mode, label in [('cache_neutral', 'Cache-neutral'), ('cache_hot', 'Cache-hot')]:
        row = actual['endpoints'][mode+'.wall_ms']; lo, hi = row['speedup_95_ci']
        lines.append(f'{label} & {row["point_speedup"]:.5f} & [{lo:.5f}, {hi:.5f}] \\\\')
    target = HERE/'generated/original_optimized_rows.tex'
    target.write_text('\\begin{tabular}{lrr}\n\\toprule\nMode & FI/PG ratio & 95\\% interval\\\\\\midrule\n'+
        '\n'.join(lines)+'\n\\bottomrule\n\\end{tabular}\n')
    transfer_path = ROOT/'results/mlsys2027_ablation_v1/qwen_suffix_20260909T014639Z_bd96c608/analysis.json'
    transfer = json.loads(transfer_path.read_text())
    inputs[str(transfer_path)] = protocol.entry.sha256_file(transfer_path)
    for name, digest in transfer['input_sha256'].items():
        if protocol.entry.sha256_file(Path(name)) != digest: raise ValueError('Changed Qwen transfer evidence')
        inputs[name] = digest
    if len(transfer['books']) != 8 or len(transfer['runs']) != 8: raise ValueError('Incomplete Qwen transfer')
    reference, candidate = (transfer['rows'][name] for name in ('reference_policy', 'without_static_suffix'))
    for row, exact_pages in ((reference, 180), (candidate, 52)):
        expected_bytes = (2*36*1376*16*8*128+4*36*1376*8+4*36*exact_pages*16*8*128+
                          4*36*8*128+2*36*32*128)
        if (row['windows'], row['tokens'], row['cache_bytes']) != (8, 12288, expected_bytes):
            raise ValueError('Qwen transfer cohort/accounting mismatch')
    if not math.isclose(candidate['ppl']/reference['ppl'], transfer['no_suffix_over_reference_ppl'], rel_tol=1e-12):
        raise ValueError('Inconsistent Qwen transfer PPL')
    for directory in transfer['runs']:
        path = Path(directory); completion = json.loads((path/'completion.json').read_text())
        if completion['return_code'] or not completion['sampled_exclusivity_passed']: raise ValueError('Qwen worker failed')
        probes = json.loads((path/'execution_probes.json').read_text())['policies']
        for values in probes.values():
            if len(values) != 18 or any(p['relative_l2'] > .005 or p['absolute_error'] > .02 for p in values):
                raise ValueError('Qwen execution probes failed')
    lo, hi = transfer['descriptive_paired_book_ratio95']
    transfer_target = HERE/'generated/qwen_suffix_summary.tex'
    transfer_target.write_text(
        f'On the same eight Qwen3 PG19 TRAIN books, removing A128 reduces PageGauge KV by '
        f'{100*(1-candidate["cache_bytes"]/reference["cache_bytes"]):.2f}\\%, from '
        f'{reference["cache_bytes"]/2**20:.2f} to {candidate["cache_bytes"]/2**20:.2f}\\,MiB. '
        f'PPL/HF changes from {reference["ppl_ratio_to_native_hf"]:.6f} to {candidate["ppl_ratio_to_native_hf"]:.6f}, '
        f'and top-1 agreement from {100*reference["top1_agreement_to_native_hf"]:.3f}\\% to '
        f'{100*candidate["top1_agreement_to_native_hf"]:.3f}\\%. '
        f'The paired-book A0/reference PPL ratio is {transfer["no_suffix_over_reference_ppl"]:.6f} '
        f'[{lo:.6f}, {hi:.6f}]. This development transfer does not establish general quality superiority; '
        'the separate runtime pilot below does not replace balanced replication before promotion.\n')
    protocol.entry.atomic_json(HERE/'generated/main_evidence_manifest.json', {'input_sha256': inputs,
        'builder_sha256': protocol.entry.sha256_file(Path(__file__)),
        'output_sha256': {str(p): protocol.entry.sha256_file(p) for p in (target, transfer_target)},
        'scope': 'Original optimized S4/A128/T768 exact split128 timing and Qwen development A128/A0 quality; no A0 or common-engine speed inference'})
    print('Validated eight original optimized blocks and reproduced hierarchical reduction.', flush=True)


if __name__ == '__main__': main()
