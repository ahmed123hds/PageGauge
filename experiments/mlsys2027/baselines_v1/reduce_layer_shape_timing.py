"""Verified layer-segment point reduction, no confidence interval."""
import argparse
import json
from pathlib import Path
import statistics
from kivi_quality import base, verify
import attention_graph_dispatch as attention
import layer_segment_dispatch as segments


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directories', nargs=2, type=Path)
    args = p.parse_args()
    rows, signatures = {}, []
    for d in args.directories:
        m = json.loads((d/'manifest.json').read_text())
        c = json.loads((d/'completion.json').read_text())
        r = json.loads((d/'analysis.json').read_text())
        verify(m)
        assert c['return_code'] == 0 and c['sampled_exclusivity_passed'] and c['own_pid_seen']
        for name, key in [('block_0.log', 'log_sha256'), ('block_0_telemetry.json', 'telemetry_sha256')]:
            assert base.sha256_file(d/name) == c[key]
        assert base.sha256_file(d/'tokens.json') == m['tokens_sha256']
        signatures.append(tuple(m[k] for k in ('tokens_sha256', 'batch', 'context', 'decode_steps', 'model', 'allocator_budget_bytes')))
        assert not r['validation_only'] and len(r['rows']) == 3 and len(r['layer_segment_timing']) == 4
        for stats in r['layer_segment_timing']:
            attention.verify_stats(stats['attention'], 1536, 32, False)
            segments.verify(stats['segments'], 1536, False)
        assert all(v['steps'] == 1536 and v['request_tokens'] == m['batch']*1536 for v in [r['warmup']]+r['rows'])
        times = [v['wall_ms_per_step'] for v in r['rows']]
        rows[m['backend']] = {'median_ms': statistics.median(times), 'range_ms': [min(times), max(times)],
            'served_cache_bytes': r['rows'][-1]['cache']['unique_storage_bytes'],
            'analysis_sha256': base.sha256_file(d/'analysis.json')}
    assert signatures[0] == signatures[1] and set(rows) == {'flashinfer_fp16', 'page_gauge'}
    print(json.dumps({'audit_passed': True, 'rows': rows,
        'flashinfer_over_page_gauge': rows['flashinfer_fp16']['median_ms']/rows['page_gauge']['median_ms'],
        'scope': __doc__}, indent=2))


if __name__ == '__main__':
    main()
