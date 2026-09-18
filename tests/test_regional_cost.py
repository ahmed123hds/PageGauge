import copy
import json
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'experiments/mlsys2027/ablation_v1'))
from regional_cost import POLICIES, accounting, configuration, validate_runtime, assess, previous


class RegionalCost(unittest.TestCase):
    def block(self, policy):
        return dict(previous.schedule()[0], backend='page_gauge', policy=policy, exact_split_pages=32)

    def test_exact_storage_and_new_history_counts(self):
        for policy, single_bytes in zip(POLICIES, (1822130176, 1813741568, 1553694720, 1723564032)):
            config = configuration(self.block(policy[0]), '/model')
            self.assertEqual(accounting(config)['served_bytes'], 4*single_bytes)
            self.assertEqual(config['exact_prefix_pages'], policy[1])

    def test_policy_planner_counts(self):
        for policy, rebuilds in zip(POLICIES, (48, 48, 96, 95)):
            config = configuration(self.block(policy[0]), '/model')
            gate = {'passed': True, 'observed_dispatch': {'graph_replays': 49152,
                'eager_calls': 0, 'graph_bank_misses': 0},
                'observed_nested_attention_dispatch': {'total_calls': 0},
                'observed_operations': {'decoder_plan_calls': 1536, 'device_position_fills': 1536,
                    'heterogeneous_page_table_updates': 0, 'exact_page_table_updates': 96,
                    'wrappers': {n: {'plan_invocations': 1536, 'plan_rebuilds': r,
                        'last_page_len_device_fills': 1536} for n, r in (('old_int8', rebuilds), ('exact_fp16', 96))}}}
            validate_runtime(gate, config)
            invalid = copy.deepcopy(gate)
            invalid['observed_operations']['wrappers']['old_int8']['plan_rebuilds'] += 1
            with self.assertRaises(RuntimeError):
                validate_runtime(invalid, config)

    def test_a0_requires_larger_history_split_not_exact_split(self):
        maximum_chunks = ((2*170)//8)//4
        self.assertEqual(maximum_chunks, 10)
        history_pages = 1376-4-48
        self.assertEqual(history_pages, 1324)
        self.assertGreater(math.ceil(history_pages/128), maximum_chunks)
        self.assertEqual(math.ceil(history_pages/maximum_chunks), 133)
        for _, prefix, suffix, tail in POLICIES:
            history = 1376-prefix-suffix-tail//16
            exact = prefix+suffix+tail//16
            self.assertLessEqual(math.ceil(history/160), maximum_chunks)
            self.assertLessEqual(math.ceil(exact/32), maximum_chunks)

    def test_completed_reference_policy_evidence(self):
        out = ROOT/'results/mlsys2027_baselines_v1/optimized_split_20260908T120901Z_3d8fd90a'
        manifest = json.loads((out/'manifest.json').read_text())
        block = next(b for b in manifest['schedule'] if b['exact_split_pages'] == 32)
        payload = json.loads((out/f"block_{block['index']}.json").read_text())
        completion = json.loads((out/f"block_{block['index']}_completion.json").read_text())
        manifest['kernel_header_sha256'] = payload['attention_implementation']['custom_module_source_hashes']['header_sha256']
        result = assess(payload, dict(block, policy='reference_policy'), manifest, completion)
        self.assertTrue(result['execution_passed'])
        # A reference result cannot be relabeled as the no-suffix policy.
        with self.assertRaises(RuntimeError):
            assess(payload, dict(block, policy='without_static_suffix'), manifest, completion)


if __name__ == '__main__':
    unittest.main()
