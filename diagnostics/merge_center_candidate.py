"""Development-only fused two-state merge and common-center restoration."""
import triton
import triton.language as tl


@triton.jit
def _merge(H, E, LH, LE, C, O, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    a = tl.load(LH+row)
    b = tl.load(LE+row)
    m = tl.maximum(a, b)
    # FlashInfer states use base-2 log-sum-exp. Both empty => center only.
    ah = tl.where(a == -float('inf'), 0., tl.exp2(a-m))
    be = tl.where(b == -float('inf'), 0., tl.exp2(b-m))
    denominator = ah+be
    wh = tl.where(denominator > 0, ah/denominator, 0.)
    we = tl.where(denominator > 0, be/denominator, 0.)
    h = tl.load(H+row*D+col, col < D, 0).to(tl.float32)
    e = tl.load(E+row*D+col, col < D, 0).to(tl.float32)
    c = tl.load(C+row*D+col, col < D, 0).to(tl.float32)
    tl.store(O+row*D+col, h*wh+e*we+c, col < D)


def merge_center(history, exact, history_lse, exact_lse, center, out):
    """Contiguous matching [..., D] arrays; caller provides output storage."""
    import torch
    arrays = (history, exact, center, out)
    if any(x.shape != history.shape or x.dtype != torch.float16 or
           x.device != history.device or not x.is_contiguous() for x in arrays):
        raise ValueError('Matching contiguous FP16 arrays required')
    if not history.is_cuda or history.ndim < 2:
        raise ValueError('CUDA row arrays required')
    for x in (history_lse, exact_lse):
        if (x.shape != history.shape[:-1] or x.dtype != torch.float32 or
                x.device != history.device or not x.is_contiguous()):
            raise ValueError('Matching contiguous FP32 LSE required')
    d = history.shape[-1]
    _merge[(history.numel()//d,)](history, exact, history_lse, exact_lse,
        center, out, d, triton.next_power_of_2(d), num_warps=4)
    return out
