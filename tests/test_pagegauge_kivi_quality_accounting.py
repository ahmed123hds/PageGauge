import importlib.util
from pathlib import Path
import unittest
import torch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('kivi_quality_accounting',ROOT/'experiments/mlsys2027/baselines_v1/kivi_quality.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class KiviAccountingTests(unittest.TestCase):
    def test_counts_shared_storage_once_and_keeps_view_backing_bytes(self):
        tensor=torch.zeros(16,dtype=torch.float16)
        result=m.cache_accounting(((tensor[:4],tensor[4:8],None,32),))
        self.assertEqual(result['logical_tensor_bytes'],16)
        self.assertEqual(result['unique_storage_bytes'],32)

    def test_distinct_storage_and_metadata(self):
        result=m.cache_accounting(((torch.zeros(4,dtype=torch.int32),torch.zeros(3,dtype=torch.float16),None,128),))
        self.assertEqual(result['unique_storage_bytes'],22)
        self.assertEqual(result['tensor_count'],2)


if __name__=='__main__':unittest.main()
