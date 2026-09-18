import ast
from pathlib import Path
import sys
import unittest
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'experiments/mlsys2027/serving_v1'))
from pagegauge_vllm.partition import Policy, partition, exact_slot
from pagegauge_vllm.request_slots import RequestSlots
from pagegauge_vllm.cache_layout import Layout


class ServingCPU(unittest.TestCase):
    def test_partition_matches_production_including_partial_pages(self):
        source = ast.parse((ROOT/'scripts/benchmark_page_gauge_transformer.py').read_text())
        selected = {'validate_exact_prefix_pages','validate_exact_static_suffix_pages','page_gauge_logical_partition'}
        functions = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in selected]
        self.assertEqual(len(functions), 3)
        namespace = {'PAGE': 16, 'Any': Any}
        exec(compile(ast.Module(body=functions, type_ignores=[]), '<unchanged-production-partition>', 'exec'), namespace)
        for suffix in (0,128):
            policy = Policy(4,suffix,48)
            for prompt in (8192,8193,20480):
                for delta in (0,1,15,16,767,768,769,1536):
                    length = prompt+delta
                    ours = partition(prompt,length,policy)
                    old = namespace['page_gauge_logical_partition']((length+15)//16,48,4,(length-1)%16+1,suffix,prompt//16)
                    self.assertEqual(ours['history'], old['old_logical_pages'])
                    self.assertEqual(ours['exact'], old['exact_logical_pages'])
                    self.assertEqual(ours['history_tokens']+ours['exact_tokens'], length)
                    slots = [exact_slot(p,prompt,2,policy) for p in ours['exact']]
                    self.assertEqual(len(set(slots)), len(slots))
                    self.assertTrue(all(2*policy.storage_pages <= p < 3*policy.storage_pages for p in slots))
        self.assertEqual(partition(8192,8193)['tail_tokens'],753)
        self.assertEqual(partition(8192,8208)['tail_tokens'],768)

    def test_reorder_release_and_lease_generation(self):
        manager = RequestSlots(2)
        a = manager.admit('a',8192,range(512))
        b = manager.admit('b',8200,range(1000,1513))
        self.assertEqual(manager.batch(['b','a']), (b,a))
        self.assertEqual(manager.batch(['a']), (a,))
        self.assertEqual(manager.state(b).current_tokens,8200)  # Not scheduled != finished.
        with self.assertRaises(MemoryError): manager.admit('c',8192,range(2000,2512))
        manager.release(a)
        c = manager.admit('c',8192,range(512))
        self.assertEqual(c.slot,a.slot)
        self.assertGreater(c.generation,a.generation)
        with self.assertRaises(ValueError): manager.state(a)
        with self.assertRaises(ValueError): manager.release(a)
        with self.assertRaises(ValueError): manager.append(b,8192,range(1000,1513))
        with self.assertRaises(ValueError): manager.batch(['c','c'])

    def test_append_page_closure_and_cross_request_alias_rejection(self):
        manager = RequestSlots(2)
        a = manager.admit('a',8192,range(512))
        b = manager.admit('b',8192,range(2000,2512))
        with self.assertRaises(ValueError): manager.append(a,8192,list(range(512))+[2000])
        self.assertEqual(manager.state(a).current_tokens,8192)
        closed, aged = [], []
        for position in range(8192,8192+800):
            event = manager.append(a,position,range((position+16)//16))
            if event['closed_page'] is not None: closed.append(event['closed_page'])
            aged.extend(event['new_history_pages'])
            self.assertEqual(manager.batch(['b','a']), (b,a))
        self.assertEqual(closed,list(range(512,562)))
        self.assertTrue(all(p < 562 for p in aged))
        self.assertIn(512,aged)  # Newly generated page really ages beyond the tail.
        with self.assertRaises(ValueError): manager.append(a,8992,list(range(1,563)))

    def test_accounting_reserves_empty_sidecar_slots(self):
        reference = Layout(32,4*1376,4)
        self.assertEqual(reference.accounting()['total_served_capacity_bytes'],7288520704)
        candidate = Layout(32,4*1376,4,policy=Policy(4,0,48))
        self.assertEqual(candidate.accounting()['total_served_capacity_bytes'],6214778880)
        accounting = reference.accounting()
        self.assertEqual(reference.admissible_pages(accounting['total_served_capacity_bytes']),4*1376)
        with self.assertRaises(MemoryError): reference.admissible_pages(accounting['reserved_sidecar_bytes']-1)
        with self.assertRaises(ValueError): RequestSlots(1).admit('short',128,range(8))


if __name__ == '__main__': unittest.main()
