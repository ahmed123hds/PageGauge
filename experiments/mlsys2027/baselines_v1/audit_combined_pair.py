"""Revalidate combined graph pilot; no new experiment or confidence interval."""
import argparse
import json
from pathlib import Path
import statistics
from kivi_quality import base, verify
from attention_graph_dispatch import verify_stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pair', type=Path)
    out = parser.parse_args().pair.resolve()
    a = json.loads((out/'analysis.json').read_text())
    parent = json.loads((out/'manifest.json').read_text())
    for name, digest in parent['validation_sha256'].items():
        assert base.sha256_file(Path(name)) == digest
    tokens = set()
    for backend in ('flashinfer_fp16', 'page_gauge'):
        d = out/backend
        m = json.loads((d/'manifest.json').read_text())
        verify(m)
        assert base.sha256_file(d/'tokens.json') == m['tokens_sha256']
        tokens.add(m['tokens_sha256'])
        c = json.loads((d/'completion.json').read_text())
        assert c['return_code'] == 0 and c['own_pid_seen'] and c['sampled_exclusivity_passed']
        for name, field in [('block_0.log', 'log_sha256'), ('block_0_telemetry.json', 'telemetry_sha256')]:
            assert base.sha256_file(d/name) == c[field]
        assert base.sha256_file(d/'analysis.json') == a['rows'][backend]['analysis_sha256']
        r = json.loads((d/'analysis.json').read_text())
        assert r['dense_graph_calls'] == [[1536]*32]*4
        assert len(r['attention_graph_dispatch']) == 4
        for stats in r['attention_graph_dispatch']:
            verify_stats(stats, 1536, 32, False)
        assert len(r['rows']) == 3 and all(v['steps'] == 1536 for v in [r['warmup']]+r['rows'])
        times = [v['wall_ms_per_step'] for v in r['rows']]
        assert a['rows'][backend]['median_ms'] == statistics.median(times)
        assert a['rows'][backend]['range_ms'] == [min(times), max(times)]
    assert len(tokens) == 1
    assert a['flashinfer_over_page_gauge'] == a['rows']['flashinfer_fp16']['median_ms']/a['rows']['page_gauge']['median_ms']
    print(json.dumps({'audit_passed': True, **a}, indent=2))


if __name__ == '__main__':
    main()
