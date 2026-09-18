"""Explicit history160/exact32 launch policy for all regional comparisons.

The history split is an analytical capacity repair, not a quantizer/kernel edit
or timing-selected sweep. Original history128 experiments remain unchanged.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]


def main():
    path = ROOT/'diagnostics/benchmark_sustained_dynamic_graphs.py'
    spec = importlib.util.spec_from_file_location('regional_sustained', path)
    worker = importlib.util.module_from_spec(spec); sys.modules[spec.name] = worker; spec.loader.exec_module(worker)
    args = worker.parse_args()
    if (args.backend, args.tail_attention, args.old_value_scale_placement, args.candidate_split_pages) != ('page_gauge','flashinfer_merge','probability',160):
        raise ValueError('Only explicit unconditioned history160/exact32 is supported')
    observations = {'decoder_instances': 0, 'exact_plan_calls': 0, 'old_plan_calls': 0,
        'requested_exact_splits': set(), 'effective_exact_splits': set(), 'old_splits': set()}
    original = worker.PG.TransformerDecoder.__init__
    def initialize(self, *positional, **keywords):
        original(self, *positional, **keywords)
        if self.backend != 'page_gauge': raise ValueError('Unexpected backend')
        observations['decoder_instances'] += 1
        exact, old = self.exact_wrapper.plan, self.old_wrapper.plan
        def exact_plan(indices, length, last, split, page_table_epoch=None):
            if split != 160: raise ValueError('Unexpected requested split')
            observations['exact_plan_calls'] += 1
            observations['requested_exact_splits'].add(split)
            observations['effective_exact_splits'].add(32)
            return exact(indices, length, last, 32, page_table_epoch)
        def old_plan(indices, length, last, split, page_table_epoch=None):
            if split != 160: raise ValueError('History split changed')
            observations['old_plan_calls'] += 1; observations['old_splits'].add(split)
            return old(indices, length, last, split, page_table_epoch)
        self.exact_wrapper.plan, self.old_wrapper.plan = exact_plan, old_plan
    worker.PG.TransformerDecoder.__init__ = initialize
    capacity_report = worker.scheduler_capacity_report
    def capacity(**kwargs):
        report = capacity_report(**kwargs)
        count = report['wrappers']['exact_fp16']['maximum_active_pages_per_request']
        report['wrappers']['exact_fp16'] = worker.scheduler_capacity(num_sms=kwargs['num_sms'],
            batch_size=kwargs['batch_size'], hkv=kwargs['hkv'], maximum_active_pages=count, fixed_split_pages=32)
        report['all_wrappers_analytically_within_capacity'] = all(r['analytically_within_capacity'] for r in report['wrappers'].values())
        report['exact_split_override'] = 32
        return report
    worker.scheduler_capacity_report = capacity
    result = worker.run(args)
    for key in ('requested_exact_splits','effective_exact_splits','old_splits'):
        observations[key] = sorted(observations[key])
    result['exact_split_experiment'] = {'exact_split_pages': 32, 'history_split_pages': 160,
        'observed_planning': observations, 'adapter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'production_sources_modified': False, 'quantization_or_kernel_changed': False,
        'scope': __doc__}
    args.output.write_text(json.dumps(result, indent=2, default=str)+'\n')
    print('Regional history160/exact32 complete: '+str(args.output), flush=True)
    if not result['passed']: raise SystemExit(2)


if __name__ == '__main__': main()
