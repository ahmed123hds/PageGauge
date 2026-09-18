"""CPU lifecycle bridge. Event recording/stream coverage is the caller's duty.

This module does not import CUDA or install an engine hook. A queryable event
must be recorded after all packing work (or all uses when retiring a lease).
"""
from .request_slots import RequestSlots
from .decode_metadata import prepare_decode, commit_decode


class PrefillLifecycle:
    def __init__(self, maximum_requests, num_layers, **slot_options):
        if type(num_layers) is not int or num_layers < 1:
            raise ValueError('Positive layer count required')
        self.num_layers = num_layers
        self._slots = RequestSlots(maximum_requests, **slot_options)
        self._phase = {}
        self._events = {}

    def begin(self, request_id, prompt_tokens, block_groups, num_computed_tokens=0):
        if num_computed_tokens != 0:
            raise ValueError('Prefix reuse/chunked prefill is unsupported')
        groups = tuple(block_groups)
        if len(groups) != 1:
            raise ValueError('Exactly one cache group required')
        lease = self._slots.admit(request_id, prompt_tokens, groups[0])
        # Reservation only: callers must not use this lease for decode yet.
        self._phase[lease] = 'prefill'
        return lease

    @staticmethod
    def _validate_event(event):
        if not callable(getattr(event, 'query', None)):
            raise ValueError('A recorded completion event is required')

    def submit_packing(self, lease, packed_layers, completion_event):
        self._slots.state(lease)
        if self._phase[lease] != 'prefill':
            raise ValueError('Packing already submitted or lease retiring')
        layers = tuple(packed_layers)
        if (any(type(i) is not int for i in layers)
                or sorted(layers) != list(range(self.num_layers))):
            raise ValueError('Every layer must be packed exactly once')
        self._validate_event(completion_event)
        self._events[lease] = completion_event
        self._phase[lease] = 'packing'

    def ready(self, lease):
        self._slots.state(lease)
        phase = self._phase[lease]
        if phase == 'packing' and self._events[lease].query():
            self._phase[lease] = 'ready'
            del self._events[lease]
        return self._phase[lease] == 'ready'

    def decode_batch(self, request_ids):
        leases = self._slots.batch(request_ids)
        if not all(self.ready(lease) for lease in leases):
            raise RuntimeError('Decode requested before packing completion or after finish')
        return tuple(self._slots.state(lease) for lease in leases)

    def retire(self, lease, last_use_event):
        self._slots.state(lease)
        if self._phase[lease] == 'retiring':
            raise ValueError('Lease already retiring')
        self._validate_event(last_use_event)
        self._events[lease] = last_use_event
        self._phase[lease] = 'retiring'

    def prepare_decode(self, scheduled):
        scheduled = tuple(scheduled)
        if not all(self.ready(lease) for lease, _, _ in scheduled):
            raise RuntimeError('All decode leases must be ready')
        return prepare_decode(self._slots, scheduled)

    def commit_decode(self, batch, completion_event):
        """Return None while pending; advance only after caller-recorded work.

        The event must cover successful work for this exact batch. Association
        and asynchronous GPU error handling remain engine responsibilities.
        """
        self._validate_event(completion_event)
        if not all(self.ready(row.lease) for row in batch.rows):
            raise RuntimeError('Cannot commit a non-ready or retiring lease')
        if not completion_event.query():
            return None
        return commit_decode(self._slots, batch)

    def reclaim(self, lease):
        self._slots.state(lease)
        if self._phase[lease] != 'retiring':
            raise ValueError('Explicit finish/abort event required')
        if not self._events[lease].query():
            return False
        self._slots.release(lease)
        del self._events[lease]
        del self._phase[lease]
        return True
