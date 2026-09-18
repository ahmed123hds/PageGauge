"""Isolated matched-storage control: reconstruct centered INT8 values in registers.

Shared-center cancellation/restoration is retained in BOTH arms. This isolates
page-scale placement, not the entire benefit of all PageGauge identities.
Never edits the production include tree or production wrapper.
"""
from pathlib import Path
import hashlib
import sys
import shutil

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT/'build/gauge_affine_flashinfer/include/flashinfer/attention/prefill.cuh'
EXPECTED = 'db0684241566d79bbbf5d48e8219d29dc21f6d5d2fdd57b059d5fbafd49aee54'


def transformed_header(payload):
    if hashlib.sha256(payload).hexdigest() != EXPECTED:
        raise ValueError('Unsupported production header')
    text = payload.decode('utf-8')
    # Two QK/PV fragment correction sites. Disable those corrections and instead
    # multiply each converted half2 K/V register before its FP16 MMA consumes it.
    if text.count('if constexpr (PAGE_GAUGE) {') != 2:
        raise ValueError('Unexpected factorized correction sites')
    text = text.replace('if constexpr (PAGE_GAUGE) {', 'if constexpr (false && PAGE_GAUGE) {')
    lines = text.splitlines(keepends=True)
    output = []
    sites = 0
    for line in lines:
        if line.strip() == 'if constexpr (!PAGE_GAUGE) {':
            indent = line[:len(line)-len(line.lstrip())]
            block = [
                'if constexpr (PAGE_GAUGE) {',
                '  static_assert(std::is_same_v<typename KTraits::DTypeQ, half>);',
                '  const uint32_t metadata_offset =',
                '      static_cast<uint32_t>(affine_page_ids[mma_kv]) * affine_num_kv_heads + affine_kv_head;',
                '  const half2 scale2 = __half2half2(affine_scale[metadata_offset]);',
                '#pragma unroll',
                '  for (uint32_t packed = 0; packed < 4; ++packed) {',
                '    *reinterpret_cast<half2*>(&b_frag[packed]) =',
                '        __hmul2(*reinterpret_cast<half2*>(&b_frag[packed]), scale2);',
                '  }',
                '} else {',
            ]
            output.extend(indent+x+'\n' for x in block)
            sites += 1
        else:
            output.append(line)
    if sites != 2:
        raise ValueError('Unexpected reconstruction sites')
    return ''.join(output)


def prepare():
    text = transformed_header(SOURCE.read_bytes())
    sha = hashlib.sha256(text.encode()).hexdigest()
    include = ROOT/'build/mlsys_factorization_control'/sha/'include'
    target = include/'flashinfer/attention/prefill.cuh'
    # FlashInfer uses relative ../ includes. A lone overriding header cannot
    # resolve those through -I fallback; copy the matching support tree without
    # replacing our generated prefill header. No production files are changed.
    if not (include/'flashinfer/cp_async.cuh').is_file():
        shutil.copytree(SOURCE.parents[2], include, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('prefill.cuh'))
    if target.exists():
        if target.read_text() != text:
            raise ValueError('Generated control header changed')
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('x') as stream:
            stream.write(text)
    return include, sha


def make_wrapper(flashinfer, workspace, **kwargs):
    import torch
    sys.path.insert(0, str(ROOT/'scripts'))
    import benchmark_flashinfer_page_affine_int8 as pg
    include, sha = prepare()
    import flashinfer.decode as decode
    from flashinfer.jit.attention import modules
    modules.dtype_map_kv[torch.int8] = 'int8_t'
    original = decode.gen_customize_batch_prefill_module

    def generator(*args, **kw):
        spec = original(*args, **kw)
        spec.extra_include_dirs = [include, *[p for p in (spec.extra_include_dirs or []) if Path(p) != include]]
        return spec

    decode.gen_customize_batch_prefill_module = generator
    try:
        jit = [f'page_gauge_centered_register_control_{sha[:16]}', torch.float16,
               torch.int8, torch.float16, torch.int32, 128, 128,
               ['k_page_scale','v_page_scale'], ['half','half'], ['sm_scale'], ['double'],
               'PageAffineInt8Attention', pg.AFFINE_VARIANT]
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, 'NHD',
            use_tensor_cores=True, backend='fa2', jit_args=jit, **kwargs)
    finally:
        decode.gen_customize_batch_prefill_module = original
    wrapper.control_header_sha256 = sha
    wrapper.page_gauge_module_uri = jit[0]
    wrapper.page_gauge_source_hashes = {
        'header_sha256': sha, 'expected_header_sha256': sha,
        'header_matches_expected': True, 'base_production_header_sha256': EXPECTED,
        'variant_sha256': hashlib.sha256(pg.AFFINE_VARIANT.encode()).hexdigest(),
        'module_source_sha256': hashlib.sha256(
            transformed_header(SOURCE.read_bytes()).encode() + b'\0' + pg.AFFINE_VARIANT.encode()).hexdigest(),
        'control_scope': 'centered register reconstruction; common-center factoring retained',
    }
    return wrapper
