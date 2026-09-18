"""Read-only revalidation of completed development shape-grid evidence."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
from kivi_quality import verify, base
from shape_contract import schedule, validate_result


def audit(out):
    read = lambda p: json.loads(p.read_text())
    m = read(out/'manifest.json')
    verify(m)
    a = read(out/'analysis.json')
    assert a['execution_complete'] is True
    assert a['rows'] == read(out/'progress.json')['rows']
    config = read(Path(m['model'])/'config.json')
    assert m['cells'] == schedule(m['parameter_header_accounting']['fp16_parameter_bytes'], config['max_position_embeddings'])
    assert len(a['rows']) == len(m['cells']) == 60
    for fixture in m['fixtures'].values():
        assert base.sha256_file(Path(fixture['path'])) == fixture['sha256']
    for cell, row in zip(m['cells'], a['rows']):
        assert all(row[k] == v for k, v in cell.items())
        if cell['action'] != 'run':
            assert row['outcome'] == 'analytically_infeasible'
            continue
        target = out/f"cell_{cell['index']}"
        cm = read(target/'manifest.json')
        verify(cm)
        assert all(cm[k] == v for k, v in cell.items())
        assert base.sha256_file(target/'tokens.json') == cm['tokens_sha256']
        fixture = read(Path(m['fixtures'][str(cell['context'])]['path']))['ids']
        assert read(target/'tokens.json')['ids'] == fixture[:cell['batch']]
        assert base.sha256_file(target/'completion.json') == row['completion_sha256']
        c = read(target/'completion.json')
        assert c['sampled_exclusivity_passed'] and c['own_pid_seen']
        assert c['monitor_contract'] == m['process_monitor']
        assert base.sha256_file(target/'block_0.log') == c['log_sha256']
        assert base.sha256_file(target/'block_0_telemetry.json') == c['telemetry_sha256']
        if c['return_code']:
            failure = read(target/'failure.json')
            assert not cell['validate'] and row['outcome'] == 'measured_cuda_oom'
            assert failure == row['failure'] and failure['type'] == 'OutOfMemoryError'
            assert 'CUDA out of memory' in failure['error']
            continue
        assert base.sha256_file(target/'analysis.json') == row['analysis_sha256']
        r = read(target/'analysis.json')
        validate_result(r, cm)
        assert row['outcome'] == ('validated' if cell['validate'] else 'verified_feasible')
        if not cell['validate']:
            times = [v['wall_ms_per_step'] for v in r['rows']]
            assert row['wall_ms_per_step'] == statistics.median(times)
            assert row['range_ms_per_step'] == [min(times), max(times)]
            assert row['served_cache_bytes'] == r['rows'][-1]['cache']['unique_storage_bytes']
            assert row['peak_allocated_bytes'] == max(v['final_memory']['peak_allocated_bytes'] for v in r['rows'])
    return {'audit_passed': True, 'counts': dict(Counter(r['outcome'] for r in a['rows'])),
            'analysis_sha256': base.sha256_file(out/'analysis.json'),
            'scope': 'Revalidated frozen sources, input metadata, actual tokens, outcome classification, result/log/telemetry hashes and timing reduction. No new GPU run or confidence interval.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    print(json.dumps(audit(p.parse_args().directory.resolve()), indent=2))
