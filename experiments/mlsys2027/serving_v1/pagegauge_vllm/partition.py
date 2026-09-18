"""Page-aligned S/A/T semantics, independent of transient serving batch rows."""
from dataclasses import dataclass

PAGE = 16


@dataclass(frozen=True)
class Policy:
    prefix_pages: int = 4
    suffix_pages: int = 128
    tail_pages: int = 48

    def __post_init__(self):
        if any(type(v) is not int for v in (self.prefix_pages, self.suffix_pages, self.tail_pages)) or min(self.prefix_pages, self.suffix_pages) < 0 or self.tail_pages < 1:
            raise ValueError('Invalid page policy')

    @property
    def storage_pages(self):
        return self.prefix_pages+self.suffix_pages+self.tail_pages


def partition(prompt_tokens, current_tokens, policy=Policy()):
    if type(prompt_tokens) is not int or type(current_tokens) is not int or not 0 < prompt_tokens <= current_tokens:
        raise ValueError('Invalid prompt/current length')
    initial = prompt_tokens//PAGE  # Partial prefill page belongs to the tail.
    pages = (current_tokens+PAGE-1)//PAGE
    if initial < policy.prefix_pages+policy.suffix_pages or pages-policy.tail_pages <= policy.prefix_pages:
        raise ValueError('Prompt cannot support this policy; explicit FP16 fallback required')
    prefix = set(range(policy.prefix_pages))
    suffix = set(range(initial-policy.suffix_pages, initial))
    tail = set(range(pages-policy.tail_pages, pages))
    exact = prefix|suffix|tail
    old = set(range(pages))-exact
    if not old or pages-1 in old:
        raise ValueError('No historical page or partial last page incorrectly quantized')
    return {'prefix': tuple(sorted(prefix)), 'suffix': tuple(sorted(suffix)), 'tail': tuple(sorted(tail)),
        'exact': tuple(sorted(exact)), 'history': tuple(sorted(old)),
        'history_tokens': PAGE*len(old), 'exact_tokens': current_tokens-PAGE*len(old),
        'tail_tokens': (policy.tail_pages-1)*PAGE+(current_tokens-1)%PAGE+1}


def exact_slot(logical_page, prompt_tokens, lease_slot, policy=Policy()):
    initial = prompt_tokens//PAGE
    if min(logical_page, lease_slot) < 0:
        raise ValueError('Negative page/lease slot')
    if logical_page < policy.prefix_pages:
        local = logical_page
    elif initial-policy.suffix_pages <= logical_page < initial:
        local = policy.prefix_pages+logical_page-(initial-policy.suffix_pages)
    else:
        local = policy.prefix_pages+policy.suffix_pages+logical_page%policy.tail_pages
    return lease_slot*policy.storage_pages+local
