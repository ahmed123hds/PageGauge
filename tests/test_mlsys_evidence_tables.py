"""CPU checks for paper-table direction, units, and cohort consistency."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('paper_tables', ROOT/'paper/mlsys2027/build_evidence_tables.py')
TABLES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TABLES)


class EvidenceTableTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads(TABLES.QUALITY.read_text())

    def test_units_and_ratios(self):
        text = TABLES.quality_rows(self.data)
        self.assertIn('2752.00 & 0.00\\%', text)
        self.assertIn('1737.72 & 36.86\\%', text)
        self.assertIn('1.02858', text)
        self.assertEqual(len(text.splitlines()), 5)

    def test_inconsistent_evidence_rejected(self):
        for key, value in [('tokens', 1), ('ppl_ratio_to_native_hf', 0.9),
                           ('kv_reduction_fraction', 0.9)]:
            data = copy.deepcopy(self.data)
            data['rows']['pagegauge_int8'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                TABLES.quality_rows(data)
        self.data['matched_actual_token_ids'] = False
        with self.assertRaises(ValueError):
            TABLES.quality_rows(self.data)

    def test_nsn_matched_cohort_and_accounting(self):
        nsn = json.loads(TABLES.NSN.read_text())
        text = TABLES.quality_rows(self.data, nsn)
        self.assertEqual(len(text.splitlines()), 7)
        self.assertIn('385.17 & 86.00\\%', text)
        self.assertIn('1.00897', text)
        nsn['matched_actual_token_ids_to_pagegauge'] = False
        with self.assertRaises(ValueError):
            TABLES.quality_rows(self.data, nsn)

    def test_qwen_kitty_references_and_reductions(self):
        qwen = json.loads(TABLES.QWEN.read_text())
        kitty = json.loads(TABLES.KITTY.read_text())
        text = TABLES.qwen_rows(qwen, kitty)
        self.assertEqual(len(text.splitlines()), 3)
        self.assertIn('1.01716', text)
        self.assertIn('521.39 & 83.16\\%', text)
        self.assertIn('7.46873', text)
        kitty['matched_actual_token_ids_to_qwen_pagegauge'] = False
        with self.assertRaises(ValueError):
            TABLES.qwen_rows(qwen, kitty)

    def test_common_engine_units_and_complete_cohort(self):
        result = json.loads(TABLES.FRONTIER.read_text())
        text = TABLES.frontier_rows(result)
        self.assertEqual(len(text.splitlines()), 5)
        self.assertIn('22.11 & [20.78, 22.67] & 180.95', text)
        self.assertIn('23.73 & [21.71, 24.94] & 168.56', text)
        result['rows']['page_gauge']['aggregate_tokens_per_second_median'] *= 4
        with self.assertRaises(ValueError):
            TABLES.frontier_rows(result)

    def test_pg19_development_not_final_test(self):
        result = json.loads(TABLES.BOOKS.read_text())
        text = TABLES.book_rows(result)
        self.assertEqual(len(text.splitlines()), 2)
        self.assertIn('6.42155', text)
        self.assertIn('[0.99993, 1.00007]', text)
        result['book_objects'] = result['book_objects'][:7]
        with self.assertRaises(ValueError):
            TABLES.book_rows(result)

    def test_capacity_ooms_not_silently_dropped(self):
        result = json.loads(TABLES.CAPACITY.read_text())
        text = TABLES.capacity_rows(result)
        self.assertIn('FI FP16 & OK & Bound & Bound', text)
        self.assertIn('PG INT8 & OK & OOM & Bound', text)
        self.assertIn('BitDecoding INT4 & OK & OK & OOM', text)
        result['rows'] = result['rows'][:-1]
        with self.assertRaises(ValueError):
            TABLES.capacity_rows(result)

    def test_qwen_pg19_complete_cohort(self):
        result = json.loads(TABLES.QWEN_BOOKS.read_text())
        text = TABLES.book_rows(result, 'Q8 ')
        self.assertIn('Q8 PG INT8 & 12.09432', text)
        self.assertIn('[0.99991, 1.00101]', text)
        self.assertEqual(len(text.splitlines()), 2)
        result['rows']['page_gauge']['tokens'] = 1536
        with self.assertRaises(ValueError):
            TABLES.book_rows(result)

    def test_regional_cohort_retains_four_policies_and_denominators(self):
        result = json.loads(TABLES.REGIONAL.read_text())
        text = TABLES.regional_rows(result)
        self.assertEqual(len(text.splitlines()), 4)
        self.assertIn('S4/A0/T768 & 0.999910 & 0.999915', text)
        self.assertIn('99.805', text)
        result['rows']['without_static_suffix']['tokens'] = 1536
        with self.assertRaises(ValueError):
            TABLES.regional_rows(result)

    def test_native_cost_keeps_all_arms_and_interruption(self):
        result = json.loads(TABLES.NATIVE_COST.read_text())
        text = TABLES.native_cost_rows(result)
        self.assertEqual(len(text.splitlines()), 4)
        self.assertIn('NSN INT2 & 2.216 & 31.17', text)
        self.assertIn('Kitty-Pro & 2.830 & 49.77', text)
        self.assertIn('521.39 & 83.16\\%', text)
        del result['interrupted_run']
        with self.assertRaises(ValueError):
            TABLES.native_cost_rows(result)


if __name__ == '__main__':
    unittest.main()
