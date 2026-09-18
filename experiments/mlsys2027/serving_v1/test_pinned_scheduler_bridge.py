"""Real pinned scheduler messages; fake events prove CPU bookkeeping only."""
import unittest

import torch
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput

from experiments.mlsys2027.serving_v1.pagegauge_vllm.scheduler_bridge import (
    PendingDeviceWork,
    SchedulerBridge,
)


class Event:
    def __init__(self, done=True):
        self.done = done

    def query(self):
        return self.done


def new_request(name, start=0):
    return NewRequestData(
        req_id=name, prompt_token_ids=[7] * 4096, mm_features=[],
        sampling_params=None, pooling_params=None,
        block_ids=(list(range(start, start + 256)),),
        num_computed_tokens=0, lora_request=None,
    )


def output(new=(), cached=(), finished=()):
    result = SchedulerOutput.make_empty()
    result.scheduled_new_reqs = list(new)
    result.finished_req_ids = set(finished)
    result.num_scheduled_tokens = {r.req_id: len(r.prompt_token_ids) for r in new}
    data = result.scheduled_cached_reqs
    for name, position, blocks in cached:
        data.req_ids.append(name)
        data.num_computed_tokens.append(position)
        data.num_output_tokens.append(position - 4096 + 1)
        data.new_block_ids.append(None if blocks is None else (list(blocks),))
        result.num_scheduled_tokens[name] = 1
    result.total_num_scheduled_tokens = sum(result.num_scheduled_tokens.values())
    return result


class SchedulerBridgeTests(unittest.TestCase):
    def setUp(self):
        self.bridge = SchedulerBridge(2, 2, 2048)

    def admit(self, *requests):
        step = self.bridge.apply(output(new=requests))
        layers = {s.lease: [0, 1] for s in step.prefills}
        self.assertTrue(self.bridge.complete(step, layers, Event()))
        return step

    def test_mixed_prefill_decode_waits_for_all_layers_and_event(self):
        self.admit(new_request('a'))
        message = output(new=[new_request('b', 256)], cached=[('a', 4096, [512])])
        step = self.bridge.apply(message, request_order=['a', 'b'])
        self.assertEqual(step.request_ids, ('a', 'b'))
        self.assertEqual(step.decode.rows[0].physical_token_slot, 512 * 16)
        event = Event(False)
        layers = {step.prefills[0].lease: [1, 0]}
        self.assertFalse(self.bridge.complete(step, layers, event))
        self.assertEqual(self.bridge.state('a').current_tokens, 4096)
        with self.assertRaises(RuntimeError):
            self.bridge.apply(SchedulerOutput.make_empty())
        event.done = True
        with self.assertRaises(ValueError):
            self.bridge.complete(step, {step.prefills[0].lease: [0]}, event)
        self.assertEqual(self.bridge.state('a').current_tokens, 4096)
        self.assertTrue(self.bridge.complete(step, layers, event))
        self.assertEqual(self.bridge.state('a').current_tokens, 4097)
        self.assertEqual(self.bridge.state('b').current_tokens, 4096)
        with self.assertRaises(ValueError):
            self.bridge.complete(step, layers, event)

    def test_unscheduled_rows_keep_leases_and_reordered_block_tables(self):
        self.admit(new_request('a'), new_request('b', 256))
        a, b = self.bridge.state('a').lease, self.bridge.state('b').lease
        step = self.bridge.apply(output(cached=[('b', 4096, [512])]))
        self.bridge.complete(step, {}, Event())
        self.assertEqual(self.bridge.state('a').lease, a)
        step = self.bridge.apply(output(cached=[('a', 4096, [513]),
                                                 ('b', 4097, None)]),
                                 request_order=['b', 'a'])
        self.assertEqual([r.lease for r in step.decode.rows], [b, a])
        self.assertEqual(step.decode.rows[0].physical_pages[-1], 512)
        self.bridge.complete(step, {}, Event())
        self.assertEqual(self.bridge.state('b').current_tokens, 4098)

    def test_same_id_reuse_requires_old_event_and_new_generation(self):
        self.admit(new_request('a'))
        old = self.bridge.state('a').lease
        message = output(new=[new_request('a')], finished=['a'])
        event = Event(False)
        with self.assertRaises(PendingDeviceWork):
            self.bridge.apply(message, last_use_events={old: event})
        self.assertEqual(self.bridge.state('a').lease, old)
        event.done = True
        step = self.bridge.apply(message, last_use_events={old: event})
        replacement = step.prefills[0].lease
        self.assertEqual(replacement.slot, old.slot)
        self.assertEqual(replacement.generation, old.generation + 1)

    def test_rejected_batch_does_not_partially_admit_or_release(self):
        self.admit(new_request('a'))
        old = self.bridge.state('a').lease
        # The first replacement is valid; the second aliases its physical pages.
        message = output(new=[new_request('a'), new_request('b')], finished=['a'])
        with self.assertRaises(ValueError):
            self.bridge.apply(message, last_use_events={old: Event()})
        self.assertEqual(self.bridge.state('a').lease, old)
        with self.assertRaises(ValueError):
            self.bridge.state('b')
        step = self.bridge.apply(output(new=[new_request('b', 256)]))
        self.assertEqual(step.prefills[0].lease.generation, 1)

    def test_unsupported_scheduler_features_rejected_before_admission(self):
        for field, value in [('kv_cache_block_copies', [object()]),
                             ('preempted_req_ids', {'a'}),
                             ('scheduled_spec_decode_tokens', {'a': [1]}),
                             ('num_common_prefix_blocks', [1]),
                             ('kv_connector_metadata', object())]:
            with self.subTest(field=field):
                message = output(new=[new_request('a')])
                setattr(message, field, value)
                with self.assertRaises(ValueError):
                    self.bridge.apply(message)
        for mode in ('chunk', 'reuse', 'groups', 'resumed', 'order'):
            with self.subTest(mode=mode):
                request = new_request('a')
                message = output(new=[request])
                kwargs = {}
                if mode == 'chunk':
                    message.num_scheduled_tokens['a'] -= 1
                    message.total_num_scheduled_tokens -= 1
                elif mode == 'reuse':
                    request.num_computed_tokens = 16
                elif mode == 'groups':
                    request.block_ids += (list(range(256)),)
                elif mode == 'resumed':
                    message.scheduled_cached_reqs.resumed_req_ids = {'a'}
                else:
                    kwargs['request_order'] = ['unknown']
                with self.assertRaises(ValueError):
                    self.bridge.apply(message, **kwargs)
        self.assertEqual(self.admit(new_request('a')).prefills[0].lease.generation, 1)

    def test_zero_pages_cannot_destroy_live_cache(self):
        self.admit(new_request('a'))
        message = output(cached=[('a', 4096, [256])])
        for pages in ([0], [256, 256], [-1], [2048]):
            message.new_block_ids_to_zero = pages
            with self.subTest(pages=pages), self.assertRaises(ValueError):
                self.bridge.apply(message)
        message.new_block_ids_to_zero = [256]
        step = self.bridge.apply(message)
        self.assertEqual(step.zero_pages, (256,))

    def test_abort_reclaims_only_scheduled_leases_after_fence(self):
        self.admit(new_request('a'), new_request('b', 256))
        b = self.bridge.state('b').lease
        step = self.bridge.apply(output(cached=[('a', 4096, [512])]))
        event = Event(False)
        self.assertFalse(self.bridge.abort(step, event))
        self.assertEqual(self.bridge.state('a').current_tokens, 4096)
        event.done = True
        self.assertTrue(self.bridge.abort(step, event))
        with self.assertRaises(ValueError):
            self.bridge.state('a')
        self.assertEqual(self.bridge.state('b').lease, b)

    def test_no_cuda_initialization(self):
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
