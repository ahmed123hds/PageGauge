"""Per-process PG exact-wrapper launch partitioning; no kernel/math changes."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]


def main():
    split=int(os.environ['PAGEGAUGE_EXPERIMENT_EXACT_SPLIT'])
    if split not in (32,128) or os.environ.get('PAGEGAUGE_VALUE_CONDITIONING','none')!='none':
        raise ValueError('Only unconditioned exact128/32 contrast allowed')
    path=ROOT/'diagnostics/benchmark_sustained_dynamic_graphs.py'
    spec=importlib.util.spec_from_file_location('optimized_split_sustained',path)
    worker=importlib.util.module_from_spec(spec);sys.modules[spec.name]=worker;spec.loader.exec_module(worker)
    args=worker.parse_args()
    if args.backend!='page_gauge' or args.tail_attention!='flashinfer_merge' or args.old_value_scale_placement!='probability':
        raise ValueError('Wrong backend/cache contract')
    observations={'decoder_instances':0,'exact_plan_calls':0,'old_plan_calls':0,
                  'requested_exact_splits':set(),'effective_exact_splits':set(),'old_splits':set()}
    original_init=worker.PG.TransformerDecoder.__init__
    def initialize(self,*positional,**keywords):
        original_init(self,*positional,**keywords)
        if self.backend!='page_gauge':raise RuntimeError('Unexpected opposite-backend decoder')
        observations['decoder_instances']+=1
        exact=self.exact_wrapper.plan;old=self.old_wrapper.plan
        def plan_exact(physical_indices,logical_tokens,last_page_len,split_pages,page_table_epoch=None):
            observations['exact_plan_calls']+=1
            observations['requested_exact_splits'].add(split_pages)
            observations['effective_exact_splits'].add(split)
            return exact(physical_indices,logical_tokens,last_page_len,split,page_table_epoch)
        def plan_old(physical_indices,logical_tokens,last_page_len,split_pages,page_table_epoch=None):
            if split_pages!=128:raise RuntimeError('INT8 history split changed')
            observations['old_plan_calls']+=1;observations['old_splits'].add(split_pages)
            return old(physical_indices,logical_tokens,last_page_len,split_pages,page_table_epoch)
        self.exact_wrapper.plan=plan_exact;self.old_wrapper.plan=plan_old
    worker.PG.TransformerDecoder.__init__=initialize
    capacity_report=worker.scheduler_capacity_report
    def actual_capacity(**kwargs):
        report=capacity_report(**kwargs)
        if kwargs['backend']!='page_gauge':raise RuntimeError('Unexpected capacity backend')
        count=report['wrappers']['exact_fp16']['maximum_active_pages_per_request']
        report['wrappers']['exact_fp16']=worker.scheduler_capacity(num_sms=kwargs['num_sms'],
            batch_size=kwargs['batch_size'],hkv=kwargs['hkv'],maximum_active_pages=count,fixed_split_pages=split)
        report['all_wrappers_analytically_within_capacity']=all(
            r['analytically_within_capacity'] for r in report['wrappers'].values())
        report['exact_split_override']=split
        return report
    worker.scheduler_capacity_report=actual_capacity
    result=worker.run(args)
    for name in ('requested_exact_splits','effective_exact_splits','old_splits'):
        observations[name]=sorted(observations[name])
    result['exact_split_experiment']={'exact_split_pages':split,'observed_planning':observations,
        'adapter_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'production_sources_modified':False,'quantization_or_kernel_changed':False,
        'scope':'Exact-wrapper partitioning only; candidate_fixed_split_pages still describes INT8 history. Actual exact split and analytic capacity reported separately; exhaustive graph preflight uses this override.'}
    args.output.write_text(json.dumps(result,indent=2,default=str)+'\n')
    print('Optimized exact split '+str(split)+' complete: '+str(args.output),flush=True)
    if not result['passed']:raise SystemExit(2)


if __name__=='__main__':main()
