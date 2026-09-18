"""Deterministic LaTeX tables from completed development evidence (no GPU use).

Run under WSL so the original evidence paths resolve. This creates table
artifacts, not a PDF; pdflatex typesets the accompanying development report.
"""
import hashlib
import json
import math
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
QUALITY = ROOT/'results/mlsys2027_baselines_v1/quality_frontier_20260908T111245Z_780773ad/analysis.json'
CAUSAL = ROOT/'results/mlsys2027_factorization_v1/full_model_20260908T081709Z_c032efb0/analysis.json'
PROFILE = ROOT/'results/mlsys2027_baselines_v1/profile_attribution_20260908T120217Z_93cf930d/analysis.json'
SPLIT = ROOT/'results/mlsys2027_baselines_v1/optimized_split_20260908T120901Z_3d8fd90a/analysis.json'
NSN = ROOT/'results/mlsys2027_baselines_v1/nsn_suite_20260908T134313Z_aa4f6ab7/analysis.json'
QWEN = ROOT/'results/mlsys2027_generalization_v1/qwen_suite_20260908T142521Z_58a37e17/analysis.json'
QWEN_SHORT = ROOT/'results/mlsys2027_generalization_v1/qwen_quality_20260908T141819Z_3f640f8a/analysis.json'
KITTY = ROOT/'results/mlsys2027_baselines_v1/kitty_suite_20260908T151119Z_e04e8815/analysis.json'
FRONTIER = ROOT/'results/mlsys2027_baselines_v1/frontier_pilot_b4_20260908T154840Z_e4fffbdc/analysis.json'
BOOKS = ROOT/'results/mlsys2027_generalization_v1/book_suite_mistral_20260908T163058Z_c15822be/analysis.json'
QWEN_BOOKS = ROOT/'results/mlsys2027_generalization_v1/book_suite_qwen3_20260908T183459Z_18348a6c/analysis.json'
NATIVE_ALLOCATOR = ROOT/'results/mlsys2027_baselines_v1/native_allocator_suite_20260908T181545Z_3d274195/analysis.json'
NATIVE_COST = ROOT/'results/mlsys2027_baselines_v1/native_serving_recovery_20260909T010428Z_3907ea45/analysis.json'
REGIONAL = ROOT/'results/mlsys2027_ablation_v1/regional_suite_20260909T011414Z_35cd0f1d/analysis.json'
GRAPHS = ROOT/'results/mlsys2027_baselines_v1/graph_pair_20260908T170120Z_a997e664/analysis.json'
CAPACITY = ROOT/'results/mlsys2027_baselines_v1/capacity_grid_20260908T170443Z_b462eb10/analysis.json'
EAGER_VALIDATION = ROOT/'results/mlsys2027_baselines_v1/eager_plan_page_gauge_20260908T175928Z_ee0ead38'
ALLOCATOR_DEFAULT = ROOT/'results/mlsys2027_baselines_v1/allocator_default_page_gauge_20260908T173051Z_8fd203dd'
EAGER_TIMING = ROOT/'results/mlsys2027_baselines_v1/eager_plan_page_gauge_20260908T181055Z_b24a2eb1'
ORDER = [('flashinfer_fp16', 'FlashInfer FP16'), ('pagegauge_int8', 'PageGauge INT8'),
         ('kivi_int4', 'KIVI INT4'), ('kivi_int2', 'KIVI INT2'),
         ('bitdecode_int4', 'BitDecoding INT4$^*$')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quality_rows(q, nsn=None):
    if not q['matched_actual_token_ids']:
        raise ValueError('Unmatched quality inputs')
    rows = []
    reference_bytes = q['rows']['flashinfer_fp16']['cache_bytes']
    entries = [(q['rows'][key], label) for key, label in ORDER]
    if nsn is not None:
        if not nsn['matched_actual_token_ids_to_pagegauge'] or nsn['rows']['hf']['cache_bytes'] != reference_bytes:
            raise ValueError('Unmatched NSN cohort/cache reference')
        entries.extend([(nsn['rows']['rotated_identity'], 'NSN unquantized control'),
                        (nsn['rows']['int2'], 'NSNQuant INT2')])
    for r, label in entries:
        if (r['windows'], r['tokens']) != (8, 12288):
            raise ValueError('Changed evaluation cohort')
        ratio = r['ppl']/r['native_hf_ppl']
        reduction = 1-r['cache_bytes']/reference_bytes
        if not math.isclose(ratio, r['ppl_ratio_to_native_hf'], rel_tol=1e-10):
            raise ValueError('Inconsistent PPL ratio')
        if not math.isclose(reduction, r['kv_reduction_fraction'], abs_tol=1e-12):
            raise ValueError('Inconsistent cache reduction')
        lo, hi = r['descriptive_window_bootstrap95']
        rows.append(f'{label} & {r["ppl"]:.5f} & {r["native_hf_ppl"]:.5f} & '
                    f'{ratio:.5f} & [{lo:.5f}, {hi:.5f}] & '
                    f'{r["cache_bytes"]/2**20:.2f} & {100*reduction:.2f}\\% \\\\')
    return '\n'.join(rows)+'\n'


def qwen_rows(qw, kitty):
    if not kitty['matched_actual_token_ids_to_qwen_pagegauge']:
        raise ValueError('Unmatched Kitty/Qwen requests')
    reference_bytes = qw['rows']['flashinfer_fp16']['cache_bytes']
    if kitty['rows']['hf']['cache_bytes'] != reference_bytes:
        raise ValueError('Different native FP16 cache reference')
    entries = [(qw['rows'][key], label) for key, label in
        [('flashinfer_fp16', 'FlashInfer FP16'), ('page_gauge', 'PageGauge INT8')]]
    entries.append((kitty['rows']['kitty_pro'], 'Kitty-Pro$^\\dagger$'))
    lines = []
    for r, label in entries:
        if (r['windows'], r['tokens']) != (8, 12288):
            raise ValueError('Incomplete Qwen/Kitty cohort')
        if not math.isclose(r['ppl']/r['native_hf_ppl'], r['ppl_ratio_to_native_hf'], rel_tol=1e-10):
            raise ValueError('Inconsistent Qwen/Kitty PPL')
        reduction = 1-r['cache_bytes']/reference_bytes
        if not math.isclose(reduction, r['kv_reduction_fraction'], abs_tol=1e-12):
            raise ValueError('Inconsistent Qwen/Kitty reduction')
        lo, hi = r['descriptive_window_bootstrap95']
        lines.append(f'{label} & {r["ppl"]:.5f} & {r["native_hf_ppl"]:.5f} & '
            f'{r["ppl_ratio_to_native_hf"]:.5f} & [{lo:.5f}, {hi:.5f}] & '
            f'{r["cache_bytes"]/2**20:.2f} & {100*reduction:.2f}\\% \\\\')
    return '\n'.join(lines)+'\n'


def frontier_rows(result):
    if result['batch'] != 4:
        raise ValueError('Expected common-engine batch-four pilot')
    names = [('flashinfer_fp16', 'FlashInfer FP16'), ('page_gauge', 'PageGauge INT8 (split32)'),
             ('kivi_int4', 'KIVI INT4'), ('kivi_int2', 'KIVI INT2'), ('bitdecode_int4', 'BitDecoding INT4')]
    if set(result['rows']) != {key for key, _ in names}:
        raise ValueError('Incomplete five-method pilot')
    lines = []
    for key, label in names:
        row = result['rows'][key]
        median = row['wall_ms_per_step_median']
        lo, hi = row['wall_repeat_range_ms']
        if not 0 < lo <= median <= hi:
            raise ValueError('Invalid repeat range')
        rate = row['aggregate_tokens_per_second_median']
        if not math.isclose(rate, 4000/median, rel_tol=1e-10):
            raise ValueError('Batch-token throughput inconsistency')
        lines.append(f'{label} & {median:.2f} & [{lo:.2f}, {hi:.2f}] & {rate:.2f} & '
            f'{row["cache"]["unique_storage_bytes"]/2**30:.2f} & {row["decode_peak_allocated_bytes"]/2**30:.2f} \\\\')
    return '\n'.join(lines)+'\n'


def book_rows(result, model_label=''):
    if result['model_family'] not in ('mistral', 'qwen3') or len(set(result['book_objects'])) != 8:
        raise ValueError('Incomplete matched book cohort')
    lines = []
    for key, label in [('flashinfer_fp16', 'FI FP16'), ('page_gauge', 'PG INT8')]:
        row = result['rows'][key]
        if (row['windows'], row['tokens']) != (8, 12288):
            raise ValueError('Incomplete book labels')
        ratio = row['ppl']/row['native_hf_ppl']
        if not math.isclose(ratio, row['ppl_ratio_to_native_hf'], rel_tol=1e-10):
            raise ValueError('Inconsistent book PPL ratio')
        lo, hi = row['descriptive_window_bootstrap95']
        lines.append(f'{model_label}{label} & {row["ppl"]:.5f} & {ratio:.5f} [{lo:.5f}, {hi:.5f}] \\\\')
    return '\n'.join(lines)+'\n'


def capacity_rows(result):
    names = [('flashinfer_fp16', 'FI FP16'), ('page_gauge', 'PG INT8'),
             ('kivi_int4', 'KIVI INT4'), ('kivi_int2', 'KIVI INT2'), ('bitdecode_int4', 'BitDecoding INT4')]
    cells = {(r['backend'], r['batch']): r for r in result['rows']}
    if len(result['rows']) != 10 or set(cells) != {(key, b) for key, _ in names for b in (8, 16)}:
        raise ValueError('Incomplete/repeated capacity grid cell')
    labels = {'analytically_infeasible': 'Bound', 'measured_cuda_oom': 'OOM', 'verified_feasible': 'OK'}
    return '\n'.join(label+' & OK & '+labels[cells[key, 8]['outcome']]+' & '+labels[cells[key, 16]['outcome']]+r' \\'
                     for key, label in names)+'\n'


def native_cost_rows(result):
    order = [('nsn', 'hf'), ('nsn', 'nsn_int2'), ('kitty', 'hf'), ('kitty', 'kitty_pro')]
    if [(r['family'], r['backend']) for r in result['rows']] != order or not result.get('interrupted_run'):
        raise ValueError('Incomplete native cost recovery/provenance')
    reference = {r['family']: r['summary']['cache_bytes'] for r in result['rows'] if r['backend'] == 'hf'}
    labels = {('nsn', 'hf'): ('Mistral', 'HF 4.48.1'), ('nsn', 'nsn_int2'): ('Mistral', 'NSN INT2'),
        ('kitty', 'hf'): ('Qwen3', 'HF 4.53.2'), ('kitty', 'kitty_pro'): ('Qwen3', 'Kitty-Pro')}
    lines = []
    for row in result['rows']:
        s = row['summary']
        family, method = labels[row['family'], row['backend']]
        lo, hi = s['decode_wall_ms_per_step']['range']
        lines.append(f'{family} & {method} & {s["prefill_wall_ms"]["median"]/1000:.3f} & '
            f'{s["decode_wall_ms_per_step"]["median"]:.2f} [{lo:.2f}, {hi:.2f}] & '
            f'{s["cache_bytes"]/2**20:.2f} & {100*(1-s["cache_bytes"]/reference[row["family"]]):.2f}\\% & '
            f'{s["peak_allocated_bytes"]/2**30:.2f} \\\\')
    return '\n'.join(lines)+'\n'


def regional_rows(result):
    policies = [('reference_policy', 'S4/A128/T768'), ('without_prefix', 'S0/A128/T768'),
        ('without_static_suffix', 'S4/A0/T768'), ('minimum_page_tail', 'S4/A128/T16')]
    if len(result['books']) != 8 or set(result['rows']) != {p[0] for p in policies}:
        raise ValueError('Incomplete regional cohort/policies')
    reference = result['rows']['reference_policy']
    lines = []
    for name, label in policies:
        r = result['rows'][name]
        if (r['windows'], r['tokens'], r['selected_execution_probe_calls']) != (8, 12288, 144):
            raise ValueError('Incomplete regional labels/execution probes')
        if r['maximum_relative_l2'] > .005 or r['maximum_absolute_error'] > .02:
            raise ValueError('Failed regional execution')
        ratio = r['ppl']/reference['ppl']
        if not math.isclose(ratio, r['ppl_ratio_to_reference_policy'], rel_tol=1e-10):
            raise ValueError('Inconsistent regional PPL ratio')
        lo, hi = r['descriptive_paired_book_ratio95']
        lines.append(f'{label} & {r["ppl_ratio_to_native_hf"]:.6f} & '
            f'{ratio:.6f} [{lo:.6f}, {hi:.6f}] & {100*r["top1_agreement_to_native_hf"]:.3f} & '
            f'{r["cache_bytes"]/2**20:.2f} & {100*(1-r["cache_bytes"]/2885681152):.2f}\\% \\\\')
    return '\n'.join(lines)+'\n'


def main():
    inputs = {}

    def checked(path, expected=None):
        digest = sha(path)
        if expected is not None and digest != expected:
            raise ValueError(f'Evidence changed: {path}')
        inputs[str(path)] = digest
        return json.loads(Path(path).read_text())

    q, c, p, s, nsn = [checked(path) for path in (QUALITY, CAUSAL, PROFILE, SPLIT, NSN)]
    for payload in (q, p, nsn):
        for path, expected in payload['input_sha256'].items():
            checked(path, expected)
    if c['status'] != 'complete' or not c['execution_passed']:
        raise ValueError('Causal contrast incomplete')
    if (c['fresh_process_blocks'], c['fixture_clusters'], c['adjacent_pairs']) != (8, 2, 4):
        raise ValueError('Unexpected causal replication')
    for index in range(8):
        a = checked(CAUSAL.parent/f'block_{index}_assessment.json')
        if not a['execution_passed']:
            raise ValueError('Failed causal block')
        checked(CAUSAL.parent/f'block_{index}.json', a['result_sha256'])
    if (s['fresh_process_blocks'], s['fixture_clusters']) != (8, 2):
        raise ValueError('Incomplete split confirmation')
    split_manifest = checked(SPLIT.parent/'manifest.json')
    if s['manifest_sha256'] != split_manifest['manifest_sha256']:
        raise ValueError('Split manifest mismatch')
    for index in range(8):
        a = checked(SPLIT.parent/f'block_{index}_assessment.json')
        if not a['execution_passed']:
            raise ValueError('Failed split block')
        block = checked(SPLIT.parent/f'block_{index}.json', a['result_sha256'])
        actual = block['exact_split_experiment']
        expected = split_manifest['schedule'][index]['exact_split_pages']
        if actual['exact_split_pages'] != expected or actual['observed_planning']['effective_exact_splits'] != [expected]:
            raise ValueError('Wrong actual split')
        if block['cache_build']['selected_backend_cache_served_bytes_excluding_following_canary'] != 7288520704:
            raise ValueError('Unmatched split cache storage')
    for directory in nsn['runs']:
        completion = checked(Path(directory)/'completion.json')
        if completion['return_code'] or not completion['sampled_exclusivity_passed']:
            raise ValueError('Invalid NSN process')
        manifest = checked(Path(directory)/'manifest.json')
        if (manifest['context'], manifest['decode_steps']) != (20480, 1536):
            raise ValueError('Changed NSN shape')
    tables = {'quality_rows.tex': quality_rows(q, nsn)}
    qw = checked(QWEN)
    for path, digest in qw['input_sha256'].items():
        checked(path, digest)
    for directory in qw['runs']:
        completion = checked(Path(directory)/'completion.json')
        if completion['return_code'] or not completion['sampled_exclusivity_passed']:
            raise ValueError('Invalid Qwen process')
    qshort = checked(QWEN_SHORT)
    for name, row in qshort['results'].items():
        checked(QWEN_SHORT.parent/(name+'.json'), row['sha256'])
    kitty = checked(KITTY)
    for path, digest in kitty['input_sha256'].items():
        checked(path, digest)
    for directory in kitty['runs']:
        completion = checked(Path(directory)/'completion.json')
        if completion['return_code'] or not completion['sampled_exclusivity_passed']:
            raise ValueError('Invalid Kitty process')
    tables['qwen_rows.tex'] = qwen_rows(qw, kitty)
    frontier = checked(FRONTIER)
    fm = checked(FRONTIER.parent/'manifest.json')
    checked(Path(fm['validation'])/'analysis.json', fm['validation_sha256'])
    if (fm['batch'], fm['context'], fm['decode_steps'], fm['timed_repeats'], fm['warmup_steps']) != (4, 20480, 1536, 3, 1536):
        raise ValueError('Wrong common-engine pilot contract')
    for backend, row in frontier['rows'].items():
        directory = Path(row['run'])
        raw = checked(directory/'analysis.json', row['analysis_sha256'])
        run_manifest = checked(directory/'manifest.json')
        completed = checked(directory/'completion.json')
        checked(directory/'tokens.json', frontier['tokens_sha256'])
        if completed['return_code'] or not completed['sampled_exclusivity_passed'] or raw['validation_only'] or len(raw['rows']) != 3:
            raise ValueError('Invalid timing execution')
        if run_manifest['allocator_budget_bytes'] != 28*2**30 or raw['warmup']['steps'] != 1536:
            raise ValueError('Changed memory/warmup policy')
        median = statistics.median(r['wall_ms_per_step'] for r in raw['rows'])
        if not math.isclose(median, row['wall_ms_per_step_median'], rel_tol=1e-12):
            raise ValueError('Changed timing reduction')
    tables['frontier_rows.tex'] = frontier_rows(frontier)
    book_table = []
    cohorts = []
    for source, label in ((BOOKS, 'M7 '), (QWEN_BOOKS, 'Q8 ')):
        books = checked(source)
        bm = checked(source.parent/'manifest.json')
        checked(Path(bm['cohort'])/'selection.json', bm['selection_sha256'])
        cohorts.append((bm['selection_sha256'], books['book_objects']))
        for path, digest in books['input_sha256'].items():
            checked(path, digest)
        for directory in books['runs']:
            path = Path(directory)
            completed = checked(path/'completion.json')
            m = checked(path/'manifest.json')
            if completed['return_code'] or not completed['sampled_exclusivity_passed'] or m['token_provenance']['split'] != 'train':
                raise ValueError('Invalid development book process/split')
            checked(path/'tokens.json', m['tokens_sha256'])
            if (m['model_family'], m['context'], m['decode_steps'], m['exact_split_pages']) != (books['model_family'], 20480, 1536, 32):
                raise ValueError('Changed book configuration')
            pg = json.loads((path/'page_gauge.json').read_text())
            if pg['consumed_new_history_pages'] != list(range(1280, 1328)):
                raise ValueError('Incomplete book recurrence')
        book_table.append(book_rows(books, label))
    if cohorts[0] != cohorts[1]:
        raise ValueError('Different PG19 book selection across models')
    tables['book_rows.tex'] = ''.join(book_table)
    regional = checked(REGIONAL)
    checked(REGIONAL.parent/'manifest.json')
    for path, digest in regional['input_sha256'].items():
        checked(path, digest)
    if sorted(regional['books']) != sorted(cohorts[0][1]):
        raise ValueError('Regional cohort differs from retained PG19 books')
    for directory in regional['runs']:
        path = Path(directory)
        rm, rc = checked(path/'manifest.json'), checked(path/'completion.json')
        if rc['return_code'] or not rc['sampled_exclusivity_passed'] or rm['token_provenance']['split'] != 'train':
            raise ValueError('Failed/nondevelopment regional process')
        for name, _, _, tail in regional['policies']:
            raw = checked(path/(name+'.json'))
            if raw['consumed_new_history_pages'] != list(range(1280, 1376-tail//16)):
                raise ValueError('Incomplete regional new-history consumption')
    tables['regional_rows.tex'] = regional_rows(regional)
    native_cost = checked(NATIVE_COST)
    for path, digest in native_cost['input_sha256'].items():
        if sha(path) != digest:
            raise ValueError('Native pilot provenance changed: '+path)
        inputs[path] = digest
    for row in native_cost['rows']:
        directory = Path(row['run'])
        raw = checked(directory/'analysis.json', row['analysis_sha256'])
        nc = checked(directory/'completion.json')
        nm = checked(directory/'manifest.json')
        checked(directory/'tokens.json', nm['tokens_sha256'])
        if nc['return_code'] or not nc['sampled_exclusivity_passed'] or (nm['context'], nm['decode_steps'], nm['repeats']) != (20480, 1536, 3):
            raise ValueError('Incomplete native cost process/recurrence')
        if raw['warmup']['steps'] != 1536 or len(raw['rows']) != 3:
            raise ValueError('Missing native timing warmup/repeats')
        summary = row['summary']
        for field in ('prefill_wall_ms', 'decode_wall_ms_per_step', 'timed_segment_sum_ms', 'decode_tokens_per_second'):
            values = [r[field] for r in raw['rows']]
            if summary[field] != {'median': statistics.median(values), 'range': [min(values), max(values)]}:
                raise ValueError('Changed native timing reduction')
        if any(r['steps'] != 1536 or r['final_lengths'] != [22016]*(32 if row['family'] == 'nsn' else 36) for r in raw['rows']):
            raise ValueError('Missing complete native layer trajectory')
        if {r['final_cache']['unique_storage_bytes'] for r in raw['rows']} != {summary['cache_bytes']}:
            raise ValueError('Changed native cache storage')
        peak = max(r[phase]['peak_allocated_bytes'] for r in raw['rows'] for phase in ('prefill_memory', 'decode_memory'))
        if peak != summary['peak_allocated_bytes']:
            raise ValueError('Changed native peak accounting')
    tables['native_cost_rows.tex'] = native_cost_rows(native_cost)
    graph = checked(GRAPHS)
    gm = checked(GRAPHS.parent/'manifest.json')
    lines = []
    for backend, label in [('flashinfer_fp16', 'FI FP16'), ('page_gauge', 'PG INT8')]:
        row = graph['rows'][backend]
        directory = Path(row['run'])
        raw = checked(directory/'analysis.json', row['analysis_sha256'])
        completed = checked(directory/'completion.json')
        checked(directory/'tokens.json', gm['tokens_sha256'])
        if completed['return_code'] or not completed['sampled_exclusivity_passed'] or raw['validation_only']:
            raise ValueError('Invalid graph timing process')
        if len(raw['rows']) != 3 or raw['warmup']['steps'] != 1536:
            raise ValueError('Incomplete graph timing')
        validation = Path(gm['validation'][backend])
        for name in ('manifest.json', 'analysis.json', 'completion.json', 'quality.json'):
            checked(validation/name, gm['input_sha256'][str(validation/name)])
        vr = json.loads((validation/'analysis.json').read_text())
        vc = json.loads((validation/'completion.json').read_text())
        if vc['return_code'] or not vc['sampled_exclusivity_passed'] or not vr['validation_only']:
            raise ValueError('Invalid graph validation')
        for stats in vr['attention_graph_dispatch']+raw['attention_graph_dispatch']:
            if (stats['served_plan_calls'], stats['graph_replays'], stats['missing_bank_count']) != (1536, 49152, 0):
                raise ValueError('Incomplete graph recurrence')
        oracle = vr['attention_graph_dispatch'][0]
        if oracle['same_cache_oracle_calls'] != 352 or oracle['max_same_cache_abs'] != 0:
            raise ValueError('Changed graph oracle evidence')
        values = [r['wall_ms_per_step'] for r in raw['rows']]
        median = statistics.median(values)
        if not math.isclose(median, row['wall_ms_per_step_median'], rel_tol=1e-12):
            raise ValueError('Changed graph timing reduction')
        lines.append(f'{label} & {median:.2f} & [{min(values):.2f}, {max(values):.2f}] \\\\')
    tables['graph_rows.tex'] = '\n'.join(lines)+'\n'
    capacity = checked(CAPACITY)
    cm = checked(CAPACITY.parent/'manifest.json')
    checked(Path(cm['original_pilot'])/'analysis.json', cm['original_pilot_sha256'])
    if cm['allocator_budget_bytes'] != 28*2**30 or cm['parameter_header_accounting']['fp16_parameter_bytes'] != 14496047104:
        raise ValueError('Changed capacity budget/parameter accounting')
    for row in capacity['rows']:
        if row['outcome'] == 'analytically_infeasible':
            bits = 16 if row['backend'] == 'flashinfer_fp16' else 8
            bound = 14496047104+row['batch']*32*2*8*128*22016*bits//8
            if bound != row['lower_bound_bytes'] or bound <= cm['allocator_budget_bytes']:
                raise ValueError('Unproven analytical exclusion')
            continue
        directory = Path(row['run'])
        completed = checked(directory/'completion.json')
        m = checked(directory/'manifest.json')
        checked(directory/'tokens.json', m['tokens_sha256'])
        if not completed['sampled_exclusivity_passed'] or (m['batch'], m['backend'], m['context'], m['decode_steps']) != (row['batch'], row['backend'], 20480, 1536):
            raise ValueError('Invalid capacity process/configuration')
        if row['outcome'] == 'measured_cuda_oom':
            failure = checked(directory/'failure.json')
            if not completed['return_code'] or failure != row['failure'] or failure['type'] != 'OutOfMemoryError' or 'CUDA out of memory' not in failure['error']:
                raise ValueError('Incorrect OOM classification')
        else:
            raw = checked(directory/'analysis.json', row['analysis_sha256'])
            if completed['return_code'] or raw['validation_only'] or raw['warmup']['steps'] != 1536 or len(raw['rows']) != 1 or raw['rows'][0]['steps'] != 1536:
                raise ValueError('Incomplete feasible capacity point')
    tables['capacity_rows.tex'] = capacity_rows(capacity)
    native_allocator = checked(NATIVE_ALLOCATOR)
    expected_cells = {(backend, batch, policy) for backend, batch in
        (('kivi_int4', 8), ('kivi_int4', 16), ('bitdecode_int4', 16), ('kivi_int2', 16))
        for policy in ('default', 'expandable')}
    if len(native_allocator['rows']) != 8 or {(v['backend'], v['batch'], v['allocator']) for v in native_allocator['rows']} != expected_cells:
        raise ValueError('Incomplete native allocator comparison')
    checked(NATIVE_ALLOCATOR.parent/'manifest.json')
    for row in native_allocator['rows']:
        directory = Path(row['run'])
        m = checked(directory/'manifest.json')
        native_completion = checked(directory/'completion.json')
        checked(directory/'tokens.json', m['tokens_sha256'])
        d = checked(directory/'allocator_diagnostics.json', row['allocator_diagnostics_sha256'])
        checked(directory/'allocator_snapshot.json', d['snapshot_sha256'])
        f = checked(directory/'failure.json', row['failure_sha256'])
        if not native_completion['return_code'] or not native_completion['sampled_exclusivity_passed'] or row['outcome'] != 'measured_cuda_oom' or f['type'] != 'OutOfMemoryError' or 'CUDA out of memory' not in f['error']:
            raise ValueError('Invalid native allocator failure classification')
        if (m['backend'], m['batch'], m['context'], m['decode_steps'], m['allocator_budget_bytes']) != (row['backend'], row['batch'], 20480, 1536, 28*1024**3):
            raise ValueError('Native allocator fixture/budget changed')
        if m['allocator_configuration'] != 'expandable_segments:'+('True' if row['allocator'] == 'expandable' else 'False'):
            raise ValueError('Wrong declared native allocator policy')
    vd = EAGER_VALIDATION
    vr, vm, vc = (checked(vd/name) for name in ('analysis.json', 'manifest.json', 'completion.json'))
    checked(vd/'quality.json', vr['quality_sha256'])
    checked(vd/'tokens.json', vm['tokens_sha256'])
    checked(vd/'allocator_diagnostics.json')
    if vc['return_code'] or not vc['sampled_exclusivity_passed'] or not vr['validation_only'] or vr['batch'] != 8 or vr['decode_steps'] != 1536:
        raise ValueError('Invalid corrected B8 validation')
    if vm['allocator_configuration'] != 'expandable_segments:True' or vm['flashinfer_cuda_graph_planning']:
        raise ValueError('Incorrect corrected planning/allocator policy')
    oracle = vr['non_graph_planning'][0]
    if oracle['oracle_calls'] != 36 or oracle['maximum_relative_l2'] > .005 or oracle['maximum_absolute_error'] > .02 or oracle['graph_enabled_count']:
        raise ValueError('Incomplete/failed corrected execution checks')
    for digest in vm['source_sha256'].values():
        source_blob = ROOT/'artifacts/mlsys2027_source_pool'/digest
        if sha(source_blob) != digest:
            raise ValueError('Missing/changed archived validation source')
        inputs[str(source_blob)] = digest
    for name in ('completion.json', 'failure.json', 'allocator_diagnostics.json'):
        checked(ALLOCATOR_DEFAULT/name)
    ad = json.loads((ALLOCATOR_DEFAULT/'allocator_diagnostics.json').read_text())
    checked(ALLOCATOR_DEFAULT/'allocator_snapshot.json', ad['snapshot_sha256'])
    tr, tm, tc = (checked(EAGER_TIMING/name) for name in ('analysis.json', 'manifest.json', 'completion.json'))
    td = checked(EAGER_TIMING/'allocator_diagnostics.json')
    checked(EAGER_TIMING/'tokens.json', vm['tokens_sha256'])
    if tc['return_code'] or not tc['sampled_exclusivity_passed'] or tr['validation_only'] or tr['batch'] != 8 or tr['warmup']['steps'] != 1536 or len(tr['rows']) != 1:
        raise ValueError('Invalid corrected B8 timing')
    if tm['source_sha256'] != vm['source_sha256'] or len(td['restored_state_checks']) != 2 or not all(v['bitwise_initial_state_match'] for v in td['restored_state_checks']):
        raise ValueError('Corrected timing differs from validated source/state')
    timed = tr['rows'][0]
    if timed['steps'] != 1536 or not math.isclose(timed['aggregate_tokens_per_second'], 8000/timed['wall_ms_per_step'], rel_tol=1e-12):
        raise ValueError('Incorrect corrected timing units')
    lines = []
    for mode, name in [('cache_neutral', 'Cache-neutral'), ('cache_hot', 'Cache-hot')]:
        e = c['endpoints'][mode+'.wall_ms']
        lo, hi = e['speedup_95_ci']
        lines.append(f'{name} & {e["register_over_factorized"]:.5f} & [{lo:.5f}, {hi:.5f}] \\\\')
    tables['causal_rows.tex'] = '\n'.join(lines)+'\n'
    lines = []
    for mode, name in [('cache_neutral', 'Cache-neutral'), ('cache_hot', 'Cache-hot')]:
        e = s['endpoints'][mode+'.wall_ms']
        lo, hi = e['speedup_95_ci']
        lines.append(f'{name} & {e["exact128_over_exact32"]:.5f} & [{lo:.5f}, {hi:.5f}] \\\\')
    tables['split_rows.tex'] = '\n'.join(lines)+'\n'
    lines = []
    for key, label in [('fi', 'FlashInfer'), ('pg_exact128', 'PG split128'), ('pg_exact32', 'PG split32')]:
        r = p['rows'][key]
        v = r['gpu_self_ms_per_step']
        lines.append(f'{label} & {v["projection_and_lm_gemv"]:.3f} & '
                     f'{v.get("int8_history_attention", 0):.3f} & {v["fp16_attention"]:.3f} & '
                     f'{v["merge"]:.3f} & {r["gpu_total_self_ms_per_step"]:.3f} \\\\')
    tables['profile_rows.tex'] = '\n'.join(lines)+'\n'
    layouts = {
        'quality_rows.tex': ('lrrrrrr', r'Method & PPL & HF PPL & PPL/HF & Descriptive 95\% interval & KV MiB & KV reduction'),
        'causal_rows.tex': ('lrr', r'Mode & Ratio & 95\% interval'),
        'split_rows.tex': ('lrr', r'Mode & Ratio & 95\% interval'),
        'profile_rows.tex': ('lrrrrr', r'Method & Projections/LM & INT8 history & FP16 attention & Merge & Total'),
        'qwen_rows.tex': ('lrrrrrr', r'Method & PPL & HF PPL & PPL/HF & Descriptive 95\% interval & KV MiB & KV reduction'),
        'frontier_rows.tex': ('lrrrrr', r'Method & Step ms & Repeat range ms & Aggregate tokens/s & KV GiB & Peak alloc. GiB'),
        'book_rows.tex': ('lrr', r'Method & PPL & PPL/HF [95\% interval]'),
        'graph_rows.tex': ('lrr', r'Method & Step ms & Repeat range ms'),
        'capacity_rows.tex': ('lccc', r'Method & B4 & B8 & B16'),
        'native_cost_rows.tex': ('llrrrrr', r'Model & Method & Prefill s & Step ms [range] & KV MiB & KV reduction & Peak GiB'),
        'regional_rows.tex': ('lrrrrr', r'Policy & PPL/HF & PPL/reference policy [95\% interval] & Top1/HF \% & KV MiB & KV reduction'),
    }
    for name, (columns, header) in layouts.items():
        tables[name] = (r'\begin{tabular}{'+columns+'}\n'+r'\toprule'+'\n'+header+r'\\\midrule'+'\n'
                        +tables[name]+r'\bottomrule'+'\n'+r'\end{tabular}'+'\n')
    out = HERE/'generated'
    out.mkdir(exist_ok=True)
    for name, text in tables.items():
        (out/name).write_text('% Generated development evidence; do not edit by hand.\n'+text)
    manifest = {'status': 'development_evidence_only', 'input_sha256': inputs,
                'builder_sha256': sha(__file__),
                'output_sha256': {name: sha(out/name) for name in tables},
                'scope': 'Exposed TRAIN data and instrumented profiles. No new final TEST claim.'}
    (out/'evidence_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
    print(f'Generated {len(tables)} LaTeX tables; verified {len(inputs)} evidence files.')


if __name__ == '__main__':
    main()
