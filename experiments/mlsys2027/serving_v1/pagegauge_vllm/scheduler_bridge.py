"""Pinned vLLM scheduler messages to PageGauge request/packing/decode state.

CPU bridge only: the engine must call this BEFORE generic block zero/copy,
execute the returned work, and supply events covering every participating
stream. No hooks or GPU kernels are installed by importing this module.
"""
from copy import copy
from dataclasses import dataclass

from .prefill_lifecycle import PrefillLifecycle


class PendingDeviceWork(RuntimeError):
    """The caller must wait for an old lease's last use, then retry unchanged."""


@dataclass(frozen=True)
class SchedulerStep:
    request_ids: tuple
    prefills: tuple
    decode: object
    zero_pages: tuple


def _fork(manager):
    """Copy CPU bookkeeping, sharing only immutable states and event handles."""
    trial = copy(manager)
    trial._slots = copy(manager._slots)
    for name in ('free', 'generations', 'states', 'page_owner'):
        setattr(trial._slots, name, getattr(manager._slots, name).copy())
    trial._phase = manager._phase.copy()
    trial._events = manager._events.copy()
    return trial


class SchedulerBridge:
    def __init__(self, maximum_requests, num_layers, num_physical_pages,
                 **slot_options):
        if type(num_physical_pages) is not int or num_physical_pages < 1:
            raise ValueError('Positive physical pool capacity required')
        self.num_physical_pages = num_physical_pages
        self._lifecycle = PrefillLifecycle(maximum_requests, num_layers,
                                          **slot_options)
        self._leases = {}
        self._pending = None

    def state(self, request_id):
        try:
            lease = self._leases[request_id]
        except KeyError as error:
            raise ValueError('Unknown request') from error
        return self._lifecycle._slots.state(lease)

    def _pages(self, pages):
        pages = tuple(pages)
        if (any(type(p) is not int or not 0 <= p < self.num_physical_pages
                for p in pages) or len(set(pages)) != len(pages)):
            raise ValueError('Invalid/duplicate physical pool page')
        return pages

    def _groups(self, groups):
        groups = tuple(groups)
        if len(groups) != 1:
            raise ValueError('Exactly one cache group required')
        return self._pages(groups[0])

    @staticmethod
    def _envelope(output):
        unsupported = (
            'scheduled_spec_decode_tokens', 'scheduled_encoder_inputs',
            'free_encoder_mm_hashes', 'preempted_req_ids',
            'num_invalid_spec_tokens', 'kv_connector_metadata',
            'has_sync_kv_loads', 'ec_connector_metadata', 'ec_manager_metadata',
            'kv_cache_block_copies', 'kv_connector_block_state',
            'num_spec_tokens_to_schedule',
        )
        for field in unsupported:
            if getattr(output, field):
                raise ValueError(f'Unsupported scheduler feature: {field}')
        if any(output.num_common_prefix_blocks):
            raise ValueError('Prefix sharing is unsupported')
        cached = output.scheduled_cached_reqs
        if cached.resumed_req_ids or cached.new_token_ids:
            raise ValueError('Preemption/resume and pipeline parallelism unsupported')
        count = len(cached.req_ids)
        if any(len(values) != count for values in (
                cached.new_block_ids, cached.num_computed_tokens,
                cached.num_output_tokens)):
            raise ValueError('Mismatched cached-request columns')
        new_ids = [r.req_id for r in output.scheduled_new_reqs]
        ids = new_ids + list(cached.req_ids)
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate/new-and-cached scheduled request')
        counts = output.num_scheduled_tokens
        if (set(counts) != set(ids)
                or any(type(n) is not int or n < 1 for n in counts.values())
                or sum(counts.values()) != output.total_num_scheduled_tokens):
            raise ValueError('Inconsistent scheduled token counts')
        return ids

    def apply(self, output, *, request_order=None, last_use_events=None):
        """Validate a complete scheduler transaction before changing any state.

        request_order must be the engine's actual request-row order, not an
        assumed match to dictionary order. last_use_events is keyed by Lease,
        not string ID, to distinguish a finished request from its replacement.
        The caller must not execute another step until complete/abort succeeds.
        """
        if self._pending is not None:
            raise RuntimeError('Previous scheduler step is still in flight')
        ids = self._envelope(output)
        order = tuple(ids if request_order is None else request_order)
        if len(order) != len(ids) or set(order) != set(ids):
            raise ValueError('Actual request-row order must cover each request once')
        trial = _fork(self._lifecycle)
        leases = self._leases.copy()
        events = {} if last_use_events is None else dict(last_use_events)
        retiring = {leases[name] for name in output.finished_req_ids if name in leases}
        if set(events) != retiring:
            raise ValueError('Supply one last-use event per known finished lease')
        for lease in sorted(retiring, key=lambda value: value.slot):
            trial.retire(lease, events[lease])
            if not trial.reclaim(lease):
                raise PendingDeviceWork('Finished lease still has outstanding device work')
            del leases[lease.request_id]

        zero_pages = self._pages(output.new_block_ids_to_zero or ())
        if any(page in trial._slots.page_owner for page in zero_pages):
            raise ValueError('Zeroing would destroy a live request cache')

        prefills = {}
        for request in output.scheduled_new_reqs:
            if (request.prompt_token_ids is None or request.prompt_embeds is not None
                    or request.mm_features or request.lora_request is not None
                    or request.pooling_params is not None
                    or request.prefill_token_ids is not None
                    or (request.prompt_is_token_ids is not None
                        and not all(request.prompt_is_token_ids))):
                raise ValueError('Only text-token, non-LoRA generation is supported')
            length = len(request.prompt_token_ids)
            if output.num_scheduled_tokens[request.req_id] != length:
                raise ValueError('One-shot complete prefill required; no chunking')
            pages = self._groups(request.block_ids)
            lease = trial.begin(request.req_id, length, (pages,),
                                request.num_computed_tokens)
            leases[request.req_id] = lease
            prefills[request.req_id] = trial._slots.state(lease)

        cached = output.scheduled_cached_reqs
        scheduled = {}
        for index, name in enumerate(cached.req_ids):
            if name not in leases or name in output.finished_req_ids:
                raise ValueError('Unknown or finished cached request')
            if output.num_scheduled_tokens[name] != 1:
                raise ValueError('Exactly one decode token per cached request required')
            lease = leases[name]
            current = trial._slots.state(lease)
            increment = cached.new_block_ids[index]
            pages = current.physical_pages
            if increment is not None:
                pages += self._groups(increment)
            self._pages(pages)
            scheduled[name] = (lease, cached.num_computed_tokens[index], pages)
        batch = trial.prepare_decode([scheduled[name] for name in order
                                      if name in scheduled])
        step = SchedulerStep(order, tuple(prefills[name] for name in order
                                          if name in prefills), batch, zero_pages)
        self._lifecycle, self._leases, self._pending = trial, leases, step
        return step

    def _check_pending(self, step, completion_event):
        if step is not self._pending:
            raise ValueError('Unknown, stale or already completed scheduler step')
        self._lifecycle._validate_event(completion_event)
        return bool(completion_event.query())

    def complete(self, step, packed_layers, completion_event):
        """Commit all prefills/decodes after successful, event-covered work.

        False means the event is pending and no state was advanced. The caller
        owns event-to-step association and propagation of asynchronous errors.
        """
        if not self._check_pending(step, completion_event):
            return False
        expected = {state.lease for state in step.prefills}
        if set(packed_layers) != expected:
            raise ValueError('Packing coverage must match exactly the new leases')
        trial = _fork(self._lifecycle)
        for state in step.prefills:
            trial.submit_packing(state.lease, packed_layers[state.lease],
                                 completion_event)
            if not trial.ready(state.lease):
                return False
        if trial.commit_decode(step.decode, completion_event) is None:
            return False
        self._lifecycle, self._pending = trial, None
        return True

    def abort(self, step, last_use_event):
        """Reclaim failed-step requests only after all their device uses finish.

        This does not turn a failed computation into a result. The engine must
        also abort these requests and release their own physical allocations.
        """
        if not self._check_pending(step, last_use_event):
            return False
        trial = _fork(self._lifecycle)
        leases = self._leases.copy()
        for name in step.request_ids:
            lease = leases.pop(name)
            trial.retire(lease, last_use_event)
            if not trial.reclaim(lease):
                return False
        self._lifecycle, self._leases, self._pending = trial, leases, None
        return True
