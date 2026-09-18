"""CPU development prototype: shared power-of-two diagonal channel conditioning.

No production source edits, GPU work, held-out evaluation or speed claims.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone
import uuid
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / 'robustness_v1'))
from affine_reference import attention, exact_mask, round_away, softmax, output_metrics


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)


def fit_gain(initial_residual):
    """No queries, decode tokens or quality scores enter fitting.

    RMS-normalized gains, rounded to powers of two, fixed exponent clamp [-8,8].
    This is a proposed fixed rule, not an optimized or novel learned transform.
    """
    x = np.asarray(initial_residual, dtype=np.float64)
    if x.ndim != 3 or not np.isfinite(x).all() or not len(x):
        raise ValueError('Expected finite prefill [tokens,heads,channels]')
    rms = np.sqrt(np.mean(x * x, axis=0))
    safe = np.maximum(rms, 2.0**-20)
    geometric_mean = np.exp(np.mean(np.log(safe), axis=-1, keepdims=True))
    exponents = np.clip(np.rint(np.log2(safe / geometric_mean)), -8, 8)
    return np.exp2(exponents).astype(np.float16)


def encode(x, center, initial, policy, conditioned):
    x, center = np.asarray(x, dtype=np.float16), np.asarray(center, dtype=np.float16)
    residual = x.astype(np.float32) - center.astype(np.float32)
    gain = fit_gain(residual[:initial]) if conditioned else np.ones(center.shape, dtype=np.float16)
    mask = exact_mask(len(x), initial, policy)
    # Exact regions are identical to the baseline, not newly quantized/transformed storage.
    reconstructed = residual.astype(np.float16).astype(np.float64)
    transformed = reconstructed / gain.astype(np.float64)
    page = policy['page_size']
    scales, clipped = [], 0
    codes = np.zeros(x.shape, dtype=np.int8)
    for start in range(0, len(x), page):
        end = min(start + page, len(x))
        if mask[start:end].all():
            continue
        if end - start != page or mask[start:end].any():
            raise ValueError('Historical page is incomplete or straddles precision regions')
        tile = residual[start:end] / gain.astype(np.float32)
        scale = np.maximum(np.max(np.abs(tile), axis=(0, 2)) / np.float32(127), np.float32(2.0**-20)).astype(np.float16)
        if not np.isfinite(scale).all():
            raise ValueError('Nonfinite stored scale')
        unbounded = round_away(tile / scale.astype(np.float32)[None, :, None])
        clipped += int(np.count_nonzero(np.abs(unbounded) > 127))
        code = np.clip(unbounded, -127, 127).astype(np.int8)
        codes[start:end] = code
        transformed[start:end] = code.astype(np.float64) * scale.astype(np.float64)[None, :, None]
        reconstructed[start:end] = transformed[start:end] * gain.astype(np.float64)
        scales.append(scale)
    return {'real': reconstructed + center.astype(np.float64), 'transformed': transformed,
            'gain': gain, 'center': center, 'mask': mask, 'clipped': clipped,
            'codes': codes, 'scales': np.asarray(scales, dtype=np.float16)}


def factorized(q, key, value, head):
    q = np.asarray(q, dtype=np.float64)
    scores = (q * key['gain'][head].astype(np.float64)) @ key['transformed'][:, head].T / np.sqrt(q.shape[-1])
    return ((softmax(scores) @ value['transformed'][:, head]) * value['gain'][head].astype(np.float64)
            + value['center'][head].astype(np.float64))


def evaluate(path, plan, gain_records):
    with np.load(path, allow_pickle=False) as data:
        q = data['query']
        caches = {}
        for kind in ('key', 'value'):
            x, center = data[kind], data[kind + '_center']
            caches[kind] = [encode(x, center, plan['initial_context_tokens'], plan['policy'], flag) for flag in (False, True)]
        original_k, original_v = data['key'], data['value']
    layer = path.stem.split('_D')[0]
    for kind in ('key', 'value'):
        gain = caches[kind][1]['gain']
        identity = layer + '_' + kind
        if identity in gain_records:
            if not np.array_equal(gain_records[identity], gain):
                raise ValueError('Prefill-only channel gains changed across decode snapshots')
        else:
            gain_records[identity] = gain
    variants = {'baseline': (0, 0), 'key_only': (1, 0), 'value_only': (0, 1), 'key_value': (1, 1)}
    rows = []
    for head in range(8):
        query = q[4*head:4*head+4]
        truth = attention(query, original_k[:, head], original_v[:, head])
        # Attribution controls retain original, not centered/quantized, opposite cache.
        key_loss = attention(query, caches['key'][0]['real'][:, head], original_v[:, head])
        value_loss = attention(query, original_k[:, head], caches['value'][0]['real'][:, head])
        row = {'head': head, 'key_loss_only': output_metrics(key_loss, truth),
               'value_loss_only': output_metrics(value_loss, truth), 'variants': {}}
        for name, (ki, vi) in variants.items():
            key, value = caches['key'][ki], caches['value'][vi]
            out = factorized(query, key, value, head)
            explicit = attention(query, key['real'][:, head], value['real'][:, head])
            mismatch = float(np.max(np.abs(out - explicit)))
            if not np.allclose(out, explicit, rtol=1e-10, atol=1e-11):
                raise ValueError('Generalized identity mismatch')
            metrics = output_metrics(out, truth)
            truth_norm = np.linalg.norm(truth, axis=-1)
            metrics['relative_l2_per_query'] = [float(a / b) if b > 0 else None for a, b in zip(metrics['l2_per_query'], truth_norm)]
            metrics['factorization_max_abs'] = mismatch
            row['variants'][name] = metrics
        rows.append(row)
    return {'capture': path.name, 'heads': rows,
            'quantized_tokens': int((~caches['key'][0]['mask']).sum()),
            'clipped': {kind: [cache['clipped'] for cache in caches[kind]] for kind in caches}}


def summarize(rows):
    heads = [h for row in rows for h in row['heads']]
    baseline = np.array([h['variants']['baseline']['l2_per_query'] for h in heads]).flatten()
    results = {}
    for name in ('baseline', 'key_only', 'value_only', 'key_value'):
        metrics = [h['variants'][name] for h in heads]
        l2 = np.array([m['l2_per_query'] for m in metrics]).flatten()
        cos = [c for m in metrics for c in m['cosine_per_query'] if c is not None]
        relative = [r for m in metrics for r in m['relative_l2_per_query'] if r is not None]
        results[name] = {'mean_l2': float(l2.mean()), 'mean_l2_vs_baseline_ratio': float(l2.mean()/baseline.mean()),
                         'max_l2': float(l2.max()), 'min_cosine': min(cos),
                         'relative_l2_p50': float(np.median(relative)), 'relative_l2_p95': float(np.percentile(relative, 95)),
                         'queries_improved': int((l2 < baseline - 1e-12).sum()),
                         'queries_worsened': int((l2 > baseline + 1e-12).sum()), 'query_count': len(l2),
                         'max_factorization_error': max(m['factorization_max_abs'] for m in metrics)}
    results['attribution'] = {key: float(np.mean([h[key]['l2_per_query'] for h in heads]))
                              for key in ('key_loss_only', 'value_loss_only')}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-run', required=True, type=Path)
    args = parser.parse_args()
    source = args.capture_run.resolve()
    manifest = json.loads((source / 'manifest.json').read_text())
    capture = json.loads((source / 'capture_summary.json').read_text())
    if capture['manifest_sha256'] != manifest['manifest_sha256']:
        raise ValueError('Capture manifest identity mismatch')
    plan = manifest['plan']
    if plan['wikitext_member'] != 'wikitext-2-raw/wiki.train.raw':
        raise ValueError('Development only: TEST fixtures forbidden')
    expected = {(layer, step) for layer in plan['layers'] for step in plan['generated_token_snapshots']}
    if len(capture['records']) != len(expected) or {(r['layer'], r['generated_tokens']) for r in capture['records']} != expected:
        raise ValueError('Incomplete capture cohort')
    evidence = {}
    for record in capture['records']:
        path = source / record['path']
        if path.name != record['path'] or digest(path) != record['sha256']:
            raise ValueError('Capture identity changed')
        evidence[record['path']] = record['sha256']
    output = ROOT / 'results/mlsys2027_representation_v2' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8])
    output.mkdir(parents=True)
    write_json(output / 'manifest.json', {'source_capture_manifest': manifest['manifest_sha256'],
              'capture_files': evidence, 'source_files': {str(p.relative_to(ROOT)): digest(p) for p in (Path(__file__), HERE.parent/'robustness_v1/affine_reference.py')},
              'variants': ['baseline', 'key_only', 'value_only', 'key_value'],
              'rule': 'prefill residual RMS/geometric mean, nearest power of two, exponent clamp [-8,8]; no hyperparameter search',
              'scope': 'development-only CPU FP64; no GPU speed or held-out quality claim; report all variants, no production selection',
              'precision_policy_unchanged': plan['policy']})
    print(f'Representation output: {output}', flush=True)
    rows, gains = [], {}
    for record in capture['records']:
        rows.append(evaluate(source / record['path'], plan, gains))
        print(f'Evaluated {record["path"]}', flush=True)
    np.savez(output / 'prefill_gains.npz', **gains)
    result = {'rows': rows, 'summary': summarize(rows), 'gain_sha256': digest(output/'prefill_gains.npz'),
              'additional_gain_bytes_B4_L32_H8_D128': 2*4*32*8*128*2,
              'cost_not_measured': 'prefill conditioning and per-page finalize division; query scaling, exact-branch alignment and output scaling; no extra FP16 region',
              'production_kernel_modified': False, 'heldout_quality_claim': False, 'speed_claim': False}
    write_json(output / 'analysis.json', result)
    print(json.dumps(result['summary'], indent=2), flush=True)


if __name__ == '__main__':
    main()
