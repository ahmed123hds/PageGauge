"""CPU reference ABI for heterogeneous single-token decode batches.

Preparation is non-mutating. This is not a CUDA scatter or stream-safety proof.
GPU integration must encode evicted exact pages BEFORE overwriting ring slots,
check lease generations on device, and fence writes before committing/releasing.
"""
from dataclasses import dataclass
from .partition import PAGE, partition, exact_slot


@dataclass(frozen=True)
class DecodeRow:
    lease: object
    expected_state: object
    position: int
    physical_pages: tuple
    physical_token_slot: int
    exact_token_slot: int
    new_history_logical_pages: tuple
    new_history_physical_pages: tuple
    new_history_exact_slots: tuple


@dataclass(frozen=True)
class DecodeBatch:
    rows: tuple

    def columns(self):
        """Integer columns for a future device ABI, ordered by scheduled row."""
        return {'request_slots':tuple(r.lease.slot for r in self.rows),
                'generations':tuple(r.lease.generation for r in self.rows),
                'positions':tuple(r.position for r in self.rows),
                'physical_token_slots':tuple(r.physical_token_slot for r in self.rows),
                'exact_token_slots':tuple(r.exact_token_slot for r in self.rows)}


def prepare_decode(manager, scheduled):
    """scheduled contains (lease, next position, full physical block table)."""
    rows=[]; ids=set(); allocated=set(manager.page_owner)
    for lease, position, physical_pages in scheduled:
        state=manager.state(lease)
        if lease.request_id in ids: raise ValueError('Duplicate scheduled request')
        ids.add(lease.request_id)
        if type(position) is not int or position != state.current_tokens:
            raise ValueError('Wrong next token position')
        pages=manager._pages(physical_pages,(position+PAGE)//PAGE)
        if pages[:len(state.physical_pages)] != state.physical_pages:
            raise ValueError('Physical remapping is unsupported')
        extra=pages[len(state.physical_pages):]
        if any(p in allocated for p in extra): raise ValueError('Physical page alias across batch')
        allocated.update(extra)
        before=partition(state.prompt_tokens,position,manager.policy)
        after=partition(state.prompt_tokens,position+1,manager.policy)
        aged=tuple(sorted(set(after['history'])-set(before['history'])))
        if any(p >= (position+1)//PAGE for p in aged):
            raise ValueError('Cannot quantize incomplete page')
        rows.append(DecodeRow(lease,state,position,pages,
            pages[position//PAGE]*PAGE+position%PAGE,
            exact_slot(position//PAGE,state.prompt_tokens,lease.slot,manager.policy)*PAGE+position%PAGE,
            aged,tuple(pages[p] for p in aged),
            tuple(exact_slot(p,state.prompt_tokens,lease.slot,manager.policy) for p in aged)))
    return DecodeBatch(tuple(rows))


def commit_decode(manager, batch):
    """CPU bookkeeping only; caller must already have fenced successful GPU work.

No concurrency is supported in this reference manager. Validate the entire
batch before mutation, including stale states and newly introduced page aliases.
"""
    for row in batch.rows:
        if manager.state(row.lease) != row.expected_state:
            raise ValueError('Decode batch is stale')
    fresh=prepare_decode(manager,[(r.lease,r.position,r.physical_pages) for r in batch.rows])
    if fresh != batch: raise ValueError('Decode metadata was altered')
    return tuple(manager.append(r.lease,r.position,r.physical_pages) for r in batch.rows)
