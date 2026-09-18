"""Verify and reduce actual dense-dispatch profiles without new GPU work."""
import argparse
import json
from pathlib import Path
from frontier_profile_pair import reduce_run
from kivi_quality import verify, base


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directories', nargs=2, type=Path)
    args = p.parse_args()
    rows = {}
    for directory in args.directories:
        m = json.loads((directory/'manifest.json').read_text())
        c = json.loads((directory/'completion.json').read_text())
        verify(m)
        if not c['own_pid_seen']:
            raise ValueError('Missing ownership observation')
        for name, key in [('block_0.log', 'log_sha256'), ('block_0_telemetry.json', 'telemetry_sha256')]:
            if base.sha256_file(directory/name) != c[key]:
                raise ValueError('Changed monitor record')
        if base.sha256_file(directory/'tokens.json') != m['tokens_sha256']:
            raise ValueError('Changed tokens')
        r = json.loads((directory/'analysis.json').read_text())
        if r['dense_graph_calls'] != [[1536]*32]:
            raise ValueError('Incomplete dense dispatch')
        rows[m['backend']] = reduce_run(directory)
    if set(rows) != {'flashinfer_fp16', 'page_gauge'} or len({v['tokens_sha256'] for v in rows.values()}) != 1:
        raise ValueError('Unmatched profiles')
    result = {'rows': rows, 'scope': 'Instrumented kernel self-time and CPU launch attribution, not end-to-end speed or additive wall-time decomposition.'}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
