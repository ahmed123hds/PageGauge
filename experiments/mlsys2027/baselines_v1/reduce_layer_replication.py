"""Audit frozen replication and apply fixture-then-pair hierarchical bootstrap."""
import argparse
import json
from pathlib import Path
import statistics
import numpy as np
from kivi_quality import base, verify
import attention_graph_dispatch as attention
import layer_segment_dispatch as segments


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    out = p.parse_args().directory.resolve()
    m = json.loads((out/'manifest.json').read_text())
    e = json.loads((out/'execution_manifest.json').read_text())
    a = json.loads((out/'timing_complete.json').read_text())
    verify(m)
    assert base.sha256_file(out/'manifest.json') == e['parent_sha256']
    for name, digest in e['validation_sha256'].items():
        assert base.sha256_file(Path(name)) == digest
    assert a['execution_complete'] and len(a['rows']) == 8 and e['order'] == m['order']
    cache, tokens = {}, {}
    for i, row in enumerate(a['rows']):
        d = out/f'block_{i}'
        cm = json.loads((d/'manifest.json').read_text())
        c = json.loads((d/'completion.json').read_text())
        r = json.loads((d/'analysis.json').read_text())
        verify(cm)
        assert row['index'] == i and row['backend'] == m['order'][i]['backend'] and row['fixture'] == m['order'][i]['fixture']
        assert cm['backend'] == row['backend'] and (cm['batch'], cm['context'], cm['decode_steps']) == (4, 20480, 1536)
        assert c['return_code'] == 0 and c['sampled_exclusivity_passed'] and c['own_pid_seen']
        for name, key in [('block_0.log', 'log_sha256'), ('block_0_telemetry.json', 'telemetry_sha256')]:
            assert base.sha256_file(d/name) == c[key]
        assert base.sha256_file(d/'analysis.json') == row['analysis_sha256']
        assert base.sha256_file(d/'tokens.json') == cm['tokens_sha256']
        tokens.setdefault(row['fixture'], set()).add(cm['tokens_sha256'])
        assert len(r['rows']) == 3 and len(r['layer_segment_timing']) == 4 and not r['validation_only']
        for stats in r['layer_segment_timing']:
            attention.verify_stats(stats['attention'], 1536, 32, False)
            segments.verify(stats['segments'], 1536, False)
        for record in [r['warmup']]+r['rows']:
            assert record['steps'] == 1536 and record['request_tokens'] == 6144
            cache.setdefault(row['backend'], set()).add(record['cache']['unique_storage_bytes'])
        assert statistics.median(v['wall_ms_per_step'] for v in r['rows']) == row['median_ms']
    assert all(len(v) == 1 for v in tokens.values()) and tokens[0] != tokens[1]
    assert all(len(v) == 1 for v in cache.values())
    logs = []
    for i in range(0, 8, 2):
        pair = {r['backend']: r['median_ms'] for r in a['rows'][i:i+2]}
        logs.append(np.log(pair['flashinfer_fp16']/pair['page_gauge']))
    logs = np.array(logs).reshape(2, 2)
    rng = np.random.default_rng(2026090915)
    fixtures = rng.integers(0, 2, size=(50000, 2))
    pairs = rng.integers(0, 2, size=(50000, 2, 2))
    samples = np.exp(logs[fixtures[:, :, None], pairs].mean(axis=(1, 2)))
    point = float(np.exp(logs.mean()))
    low, high = map(float, np.quantile(samples, [.025, .975]))
    result = {'execution_audit_passed': True, 'fresh_processes': 8, 'fixture_clusters': 2,
        'flashinfer_over_page_gauge': point, 'hierarchical_95_ci': [low, high],
        'ci_lower_gt_1_1': low > 1.1, 'pair_ratios': np.exp(logs).tolist(),
        'served_cache_bytes': {k: next(iter(v)) for k, v in cache.items()},
        'bootstrap_draws': 50000, 'bootstrap_seed': 2026090915,
        'timing_complete_sha256': base.sha256_file(out/'timing_complete.json'),
        'reducer_sha256': base.sha256_file(Path(__file__)),
        'scope': 'Development B4/C20480/D1536 reference-policy full teacher-forced decode. Two fixture clusters limit generality. Capture/prefill excluded; not final TEST, online serving, or universal speed claim.'}
    if (out/'replication_analysis.json').exists():
        assert json.loads((out/'replication_analysis.json').read_text()) == result
    else:
        base.atomic_json(out/'replication_analysis.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
