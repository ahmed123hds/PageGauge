import copy
import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('optimized_split_test_module',ROOT/'experiments/mlsys2027/baselines_v1/optimized_split.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class OptimizedSplitTests(unittest.TestCase):
    def test_capacity_report_must_match_observed_split(self):
        block={'exact_split_pages':32}
        manifest={'source_sha256':{'experiments/mlsys2027/baselines_v1/optimized_split_worker.py':'digest'}}
        p={'exact_split_experiment':{'exact_split_pages':32,'adapter_sha256':'digest',
            'production_sources_modified':False,'quantization_or_kernel_changed':False,
            'observed_planning':{'decoder_instances':1,'exact_plan_calls':1536,'old_plan_calls':1536,
                'requested_exact_splits':[128],'effective_exact_splits':[32],'old_splits':[128]}},
            'scheduler_capacity':{'wrappers':{'exact_fp16':{'fixed_split_pages':32},'old_int8':{'fixed_split_pages':128}}}}
        m.validate_experiment(p,block,manifest)
        p['scheduler_capacity']['wrappers']['exact_fp16']['fixed_split_pages']=128
        with self.assertRaises(RuntimeError):m.validate_experiment(p,block,manifest)

    def test_ratio_direction_and_missing_blocks(self):
        blocks=[dict(b,backend='page_gauge',exact_split_pages=128 if b['backend']=='flashinfer_fp16' else 32) for b in m.previous.schedule()]
        payloads=[]
        for b in blocks:
            latency=12 if b['exact_split_pages']==128 else 10
            payloads.append({'pairing':{'configuration':{k:str(b['seed']) for k in m.previous.MATCHED_FIELDS}},
                'timed_work':{'steps':1536},'timing_modes':{mode:{'raw_samples':[{'wall_ms':latency*1536,'cuda_ms':latency*1536}]*3} for mode in m.previous.MODES}})
        result=m.reduce_results(payloads,blocks)
        self.assertAlmostEqual(result['endpoints']['cache_neutral.wall_ms']['exact128_over_exact32'],1.2)
        with self.assertRaises(ValueError):m.reduce_results(payloads[:-1],blocks)
        changed=copy.deepcopy(payloads);changed[1]['pairing']['configuration']['teacher_inputs_sha256']='different'
        with self.assertRaises(RuntimeError):m.reduce_results(changed,blocks)


if __name__=='__main__':unittest.main()
