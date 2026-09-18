import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/mlsys2027/baselines_v1'))
spec=importlib.util.spec_from_file_location('common_engine_checks',ROOT/'experiments/mlsys2027/baselines_v1/common_engine.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class CommonEngineChecks(unittest.TestCase):
    def test_rejects_missing_layer_updates(self):
        with self.assertRaises(RuntimeError):
            m.check_cache(tuple((None,22016) for _ in range(31)),'kivi_int4',20480,1536)

    def test_bitdecode_requires_new_history_consumption(self):
        cache=SimpleNamespace(finalized_blocks=12,consumed_finalized_blocks=11)
        past=tuple((cache,22016) for _ in range(32))
        m.check_cache(past,'bitdecode_int4',20480,1536)
        cache.consumed_finalized_blocks=10
        with self.assertRaises(RuntimeError):m.check_cache(past,'bitdecode_int4',20480,1536)

    def test_paged_requires_each_plan_and_layer(self):
        cache=SimpleNamespace(logical_lengths=[22016]*32,decoder_plan_calls=1536)
        past=tuple((cache,22016) for _ in range(32))
        m.check_cache(past,'page_gauge',20480,1536)
        cache.logical_lengths[-1]-=1
        with self.assertRaises(RuntimeError):m.check_cache(past,'page_gauge',20480,1536)


if __name__=='__main__':unittest.main()
