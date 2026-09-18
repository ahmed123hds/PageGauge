"""Isolated experiment adapter; production files and default behavior unchanged."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]


def main():
    arm = os.environ.get('PAGEGAUGE_FACTORIZATION_ARM')
    if arm not in ('factorized', 'register_control'):
        raise ValueError('Explicit experiment arm required')
    if os.environ.get('PAGEGAUGE_VALUE_CONDITIONING', 'none') != 'none':
        raise ValueError('This contrast must not condition the representation')
    path = ROOT/'diagnostics/benchmark_sustained_dynamic_graphs.py'
    spec = importlib.util.spec_from_file_location('e1_sustained_worker', path)
    worker = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = worker
    spec.loader.exec_module(worker)
    args = worker.parse_args()
    if args.backend != 'page_gauge' or args.old_value_scale_placement != 'probability':
        raise ValueError('Expected unmodified probability-placement PageGauge interface')
    if args.tail_attention != 'flashinfer_merge':
        raise ValueError('Only the audited segmented control is supported')
    # Patch only this process's factory, before constructing any decoder. The
    # register-control header disables fragment corrections and scales K/V
    # registers instead; its actual module/hash are emitted by the wrapper.
    if arm == 'register_control':
        from control import make_wrapper
        worker.PG.PAGE_KERNEL.make_page_gauge_wrapper = make_wrapper
    result = worker.run(args)
    implementation = result['attention_implementation']
    uri = implementation['custom_module_uri']
    if ('centered_register_control' in uri) != (arm == 'register_control'):
        raise RuntimeError('Requested and observed implementations differ')
    implementation['old_int8_value_scale_placement'] = (
        'converted_value_register' if arm == 'register_control' else 'probability')
    implementation['old_int8_key_scale_placement'] = (
        'converted_key_register' if arm == 'register_control' else 'score_fragment')
    result['factorization_experiment'] = {
        'arm': arm,
        'adapter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'production_sources_modified': False,
        'common_center_factoring_retained_in_both_arms': True,
        'configuration_note': 'Worker CLI selects interface; actual scale placement is in attention_implementation.',
    }
    args.output.write_text(json.dumps(result, indent=2, default=str)+'\n')
    print('Wrote E1 '+arm+' '+str(args.output), flush=True)
    if not result['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
