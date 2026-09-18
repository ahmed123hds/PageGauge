"""Reduce three completed native arms plus a resumed fourth; never hide interruption."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'diagnostics'))
import mlsys_rtx5090_entry as base
from kivi_quality import verify
from native_serving_suite import summarize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', type=Path, required=True)
    p.add_argument('--replacement', type=Path, required=True)
    p.add_argument('--interrupted', type=Path, required=True)
    args = p.parse_args()
    suite, replacement, interrupted = (path.resolve() for path in (args.suite, args.replacement, args.interrupted))
    parent = json.loads((suite/'manifest.json').read_text())
    progress = json.loads((suite/'progress.json').read_text())
    if len(parent['jobs']) != 4 or len(progress['rows']) != 3 or (suite/'analysis.json').exists():
        raise ValueError('Expected exactly three completed arms in an unfinished four-arm suite')
    if (interrupted/'completion.json').exists() or (interrupted/'analysis.json').exists():
        raise ValueError('Interrupted arm now has a terminal result; inspect rather than replace')
    evidence = {}
    def read(path, expected=None):
        digest = base.sha256_file(path)
        if expected is not None and digest != expected:
            raise ValueError('Changed retained evidence: '+str(path))
        evidence[str(path)] = digest
        return json.loads(path.read_text())
    read(suite/'manifest.json')
    read(suite/'progress.json')
    im = read(interrupted/'manifest.json')
    for name, digest in parent['input_sha256'].items():
        if base.sha256_file(Path(name)) != digest:
            raise ValueError('Original suite source/input drift: '+name)
        evidence[name] = digest
    rows = []
    directories = [Path(row['run']) for row in progress['rows']]+[replacement]
    for index, (job, directory) in enumerate(zip(parent['jobs'], directories)):
        m, c, raw = (read(directory/name) for name in ('manifest.json', 'completion.json', 'analysis.json'))
        verify(m)
        read(directory/'tokens.json', m['tokens_sha256'])
        if c['return_code'] or not c['sampled_exclusivity_passed']:
            raise ValueError('Incomplete replacement/retained worker')
        if (m['family'], m['backend'], m['quality_fixture']) != (job['family'], job['backend'], job['fixture']):
            raise ValueError('Changed scheduled arm/quality fixture')
        summary = summarize(raw)
        if index < 3:
            prior = progress['rows'][index]
            if prior['analysis_sha256'] != base.sha256_file(directory/'analysis.json') or summary != prior['summary']:
                raise ValueError('Retained arm reduction changed')
        else:
            for field in ('family', 'backend', 'source_sha256', 'tokens_sha256', 'model', 'context',
                          'decode_steps', 'repeats', 'allocator_configuration', 'allocator_budget_bytes'):
                if m[field] != im[field]:
                    raise ValueError('Replacement changed interrupted configuration: '+field)
        rows.append(dict(job, run=str(directory), analysis_sha256=base.sha256_file(directory/'analysis.json'), summary=summary))
    out = suite.parent/('native_serving_recovery_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True)
    base.atomic_json(out/'analysis.json', dict(rows=rows, input_sha256=evidence,
        original_suite=str(suite), interrupted_run=str(interrupted), replacement_run=str(replacement),
        reducer_sha256=base.sha256_file(Path(__file__)),
        scope='Completed native request-cost pilots, retaining three original completed arms and rerunning only the interrupted fourth after runtime restart. Original interrupted directory unchanged; no numerical/kernel failure inferred. Kitty arms are separated by a host interruption/time gap: raw per-arm pilot costs, not an adjacent matched speed ratio, balanced comparison, CI, PageGauge contrast or final TEST.'))
    print('Recovered native serving evidence: '+str(out), flush=True)
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == '__main__':
    main()
