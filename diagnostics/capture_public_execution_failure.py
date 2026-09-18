"""Separate one-input failure replay; never edits the frozen evaluation."""
import argparse
import hashlib
import json
import linecache
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--case-id', required=True)
    p.add_argument('--candidate', action='store_true')
    args = p.parse_args()
    source, out = args.source.resolve(), args.output.resolve()
    if out.exists():
        raise ValueError('Diagnostic output must be new')
    sys.path.insert(0, str(ROOT/'experiments/mlsys2027/tasks_v1'))
    import task_generation_worker as worker
    manifest = json.loads((source/'manifest.json').read_text())
    worker.verify(manifest)
    if args.candidate:
        import benchmark_pg19_external_quality as quality
        from install_merge_center_candidate import install
        install(quality.PG.TransformerDecoder)
    fixtures_path = source/'fixtures.json'
    if worker.base.sha256_file(fixtures_path) != manifest['fixtures_sha256']:
        raise ValueError('Original fixture hash changed')
    fixtures = json.loads(fixtures_path.read_text())
    selected = [c for c in fixtures['cases'] if c['case_id'] == args.case_id]
    if len(selected) != 1:
        raise ValueError('Expected exactly one known failed input')
    out.mkdir(parents=True)
    fixtures['cases'] = selected
    worker.base.atomic_json(out/'fixtures.json', fixtures)
    manifest['fixtures_sha256'] = worker.base.sha256_file(out/'fixtures.json')
    manifest['diagnostic_only'] = True
    manifest['merge_center_candidate'] = args.candidate
    manifest['original_failed_job'] = str(source)
    worker.base.atomic_json(out/'manifest.json', manifest)
    worker.base.atomic_json(out/'diagnostic_provenance.json', {
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'original_manifest_sha256': worker.base.sha256_file(source/'manifest.json'),
        'original_failure_sha256': (worker.base.sha256_file(source/'failure.json')
                                   if (source/'failure.json').exists() else None),
        'scope': 'One exposed failed input, unchanged assertion; not final evaluation. First-case HF validation additionally runs.'})
    captured = False
    region_capture = {}

    def trace(frame, event, arg):
        nonlocal captured
        if (event == 'line' and frame.f_code.co_name == 'eager_attention'
                and Path(frame.f_code.co_filename).name == 'benchmark_page_gauge_transformer.py'
                and frame.f_locals.get('layer') == 31):
            source_line = linecache.getline(frame.f_code.co_filename, frame.f_lineno).strip()
            driver = frame.f_locals['self']
            if source_line == 'flashinfer.merge_state_in_place(':
                region_capture.clear()
                for name, value in (
                    ('history_output', driver.attention_output[31, 0, :4]),
                    ('history_lse', driver.old_lse[31, 0, :4]),
                    ('exact_output', driver.exact_output[31, 0, :4]),
                    ('exact_lse', driver.exact_lse[31, 0, :4])):
                    region_capture[name] = value.detach().cpu().clone()
            elif source_line == 'self.attention_output[layer].add_(self.cache.output_center[layer])':
                region_capture['merged_centered_output'] = driver.attention_output[31, 0, :4].detach().cpu().clone()
        if (not captured and event == 'exception'
                and frame.f_code.co_name == 'attention'
                and Path(frame.f_code.co_filename).name == 'regional_quality.py'
                and 'Selected reconstructed-cache execution check failed' in str(arg[1])):
            import torch
            names = ('keys', 'values', 'query', 'expected', 'observed',
                     'old_k', 'old_v', 'exact_k', 'exact_v', 'weights')
            payload = {n: frame.f_locals[n].detach().cpu().clone() for n in names}
            driver = frame.f_locals['driver']
            layer, head = frame.f_locals['layer'], frame.f_locals['head']
            payload['center'] = driver.cache.output_center[layer, 0, head*4:(head+1)*4].detach().cpu().clone()
            payload['row'] = frame.f_locals['row']
            payload.update(region_capture)
            torch.save(payload, out/'failure_tensors.pt')
            captured = True
        return trace

    sys.settrace(trace)
    try:
        worker.worker(out)
    finally:
        sys.settrace(None)
        worker.base.atomic_json(out/'capture_status.json', {'captured': captured,
            'scope': 'Diagnostic replay only; original failure and thresholds retained'})


if __name__ == '__main__':
    main()
