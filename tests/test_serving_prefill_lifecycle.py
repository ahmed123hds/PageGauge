"""Fake events test bookkeeping only, not CUDA stream safety."""
import unittest
from experiments.mlsys2027.serving_v1.pagegauge_vllm.prefill_lifecycle import PrefillLifecycle


class Event:
    done = False

    def query(self):
        return self.done


class PrefillLifecycleTests(unittest.TestCase):
    def test_all_layers_and_completion_before_decode(self):
        manager = PrefillLifecycle(1, 2)
        lease = manager.begin('a', 4096, [range(256)])
        with self.assertRaises(RuntimeError):
            manager.decode_batch(['a'])
        for layers in ([0], [0, 0], [0, 1, 2]):
            with self.assertRaises(ValueError):
                manager.submit_packing(lease, layers, Event())
        event = Event()
        manager.submit_packing(lease, [1, 0], event)
        self.assertFalse(manager.ready(lease))
        event.done = True
        self.assertEqual(manager.decode_batch(['a'])[0].lease, lease)

    def test_finish_waits_before_same_id_and_blocks_reused(self):
        manager = PrefillLifecycle(1, 2)
        lease = manager.begin('a', 4096, [range(256)])
        event = Event()
        manager.retire(lease, event)  # An aborted prefill also needs a fence.
        self.assertFalse(manager.reclaim(lease))
        with self.assertRaises(ValueError):
            manager.begin('a', 4096, [range(256)])
        event.done = True
        self.assertTrue(manager.reclaim(lease))
        new = manager.begin('a', 4096, [range(256)])
        self.assertEqual(new.slot, lease.slot)
        self.assertGreater(new.generation, lease.generation)
        with self.assertRaises(ValueError):
            manager.ready(lease)

    def test_unscheduled_requests_keep_identity(self):
        manager = PrefillLifecycle(2, 1)
        leases = [manager.begin(name, 4096, [range(start, start+256)])
                  for name, start in [('a', 0), ('b', 256)]]
        event = Event()
        event.done = True
        for lease in leases:
            manager.submit_packing(lease, [0], event)
        manager.decode_batch(['b'])
        self.assertEqual([s.lease for s in manager.decode_batch(['b', 'a'])], leases[::-1])

    def test_unsupported_admission_does_not_consume_slot(self):
        manager = PrefillLifecycle(1, 1)
        with self.assertRaises(ValueError):
            manager.begin('a', 4096, [range(256)], num_computed_tokens=16)
        with self.assertRaises(ValueError):
            manager.begin('a', 4096, [range(256), range(256)])
        self.assertEqual(manager.begin('a', 4096, [range(256)]).generation, 1)

    def test_decode_advances_once_after_completion(self):
        manager = PrefillLifecycle(1, 1)
        lease = manager.begin('a', 4096, [range(256)])
        packed = Event()
        packed.done = True
        manager.submit_packing(lease, [0], packed)
        batch = manager.prepare_decode([(lease, 4096, range(257))])
        decoded = Event()
        self.assertIsNone(manager.commit_decode(batch, decoded))
        self.assertEqual(manager.decode_batch(['a'])[0].current_tokens, 4096)
        decoded.done = True
        manager.commit_decode(batch, decoded)
        self.assertEqual(manager.decode_batch(['a'])[0].current_tokens, 4097)
        with self.assertRaises(ValueError):
            manager.commit_decode(batch, decoded)

    def test_retired_batch_cannot_partially_commit(self):
        manager = PrefillLifecycle(2, 1)
        a = manager.begin('a', 4096, [range(256)])
        b = manager.begin('b', 4096, [range(1000, 1256)])
        done = Event()
        done.done = True
        for lease in (a, b):
            manager.submit_packing(lease, [0], done)
        batch = manager.prepare_decode([(a, 4096, range(257)),
                                        (b, 4096, range(1000, 1257))])
        manager.retire(b, done)
        with self.assertRaises(RuntimeError):
            manager.commit_decode(batch, done)
        self.assertEqual(manager.decode_batch(['a'])[0].current_tokens, 4096)


if __name__ == '__main__':
    unittest.main()
