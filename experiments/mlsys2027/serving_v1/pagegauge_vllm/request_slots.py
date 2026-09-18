"""Explicit request lifetimes; row removal alone must never free request state."""
from dataclasses import dataclass
import heapq
from .partition import PAGE, Policy, partition


@dataclass(frozen=True)
class Lease:
    request_id: str
    slot: int
    generation: int


@dataclass(frozen=True)
class State:
    lease: Lease
    prompt_tokens: int
    current_tokens: int
    physical_pages: tuple


class RequestSlots:
    def __init__(self, maximum_requests, policy=Policy()):
        if type(maximum_requests) is not int or maximum_requests < 1:
            raise ValueError('Positive fixed admission capacity required')
        self.policy = policy
        self.free = list(range(maximum_requests))
        self.generations = [0]*maximum_requests
        self.states = {}
        self.page_owner = {}

    def _pages(self, physical_pages, count):
        pages = tuple(physical_pages)
        if len(pages) != count or len(set(pages)) != count or any(type(p) is not int or p < 0 for p in pages):
            raise ValueError('Wrong/duplicate physical block table')
        return pages

    def admit(self, request_id, prompt_tokens, physical_pages):
        if not isinstance(request_id, str) or not request_id or request_id in self.states:
            raise ValueError('Invalid or duplicate live request ID')
        partition(prompt_tokens, prompt_tokens, self.policy)
        pages = self._pages(physical_pages, (prompt_tokens+PAGE-1)//PAGE)
        if any(p in self.page_owner for p in pages):
            raise ValueError('Cross-request page sharing is disabled')
        if not self.free: raise MemoryError('Declared exact-sidecar admission capacity reached')
        slot = heapq.heappop(self.free)
        self.generations[slot] += 1
        lease = Lease(request_id, slot, self.generations[slot])
        self.states[request_id] = State(lease, prompt_tokens, prompt_tokens, pages)
        self.page_owner.update((p, lease) for p in pages)
        return lease

    def state(self, lease):
        state = self.states.get(lease.request_id)
        if state is None or state.lease != lease:
            raise ValueError('Stale request lease/center generation')
        return state

    def batch(self, request_ids):
        ids = tuple(request_ids)
        if len(set(ids)) != len(ids): raise ValueError('Duplicate batch row')
        try: return tuple(self.states[name].lease for name in ids)
        except KeyError as error: raise ValueError('Unknown scheduled request') from error

    def append(self, lease, position, physical_pages):
        state = self.state(lease)
        if position != state.current_tokens:
            raise ValueError('Skipped/replayed token or wrong per-request position')
        length = position+1
        pages = self._pages(physical_pages, (length+PAGE-1)//PAGE)
        if pages[:len(state.physical_pages)] != state.physical_pages:
            raise ValueError('Preemption/remapping is not supported in phase one')
        extra = pages[len(state.physical_pages):]
        if any(p in self.page_owner for p in extra):
            raise ValueError('New physical page belongs to a live request')
        before = partition(state.prompt_tokens, state.current_tokens, self.policy)
        after = partition(state.prompt_tokens, length, self.policy)
        aged = sorted(set(after['history'])-set(before['history']))
        if any(p >= length//PAGE for p in aged):
            raise ValueError('Attempt to consume an unfinished INT8 page')
        self.page_owner.update((p, lease) for p in extra)
        self.states[lease.request_id] = State(lease, state.prompt_tokens, length, pages)
        return {'lease': lease, 'position': position, 'closed_page': position//PAGE if length%PAGE == 0 else None,
            'new_history_pages': tuple(aged), 'partition': after}

    def release(self, lease):
        state = self.state(lease)
        for page in state.physical_pages:
            if self.page_owner.get(page) != lease:
                raise ValueError('Page ownership corrupted')
        for page in state.physical_pages: del self.page_owner[page]
        del self.states[lease.request_id]
        heapq.heappush(self.free, lease.slot)
