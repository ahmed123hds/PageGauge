"""CPU-only E0 screening: fixed calibration gains, FP16-coordinate cache replay.

Existing exposed TRAIN trajectory only. No new model, GPU timing, production
integration, cross-document generalization, or full-model quality claim.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import uuid
import numpy as np
from prototype import ROOT, HERE, digest, write_json, fit_gain, encode, attention, output_metrics


def calibrated_gain(value, tokens):
    sample = np.asarray(value[:tokens], dtype=np.float16)
    if len(sample) != tokens:
        raise ValueError('Insufficient calibration tokens')
    center = sample.astype(np.float32).mean(0).astype(np.float16)
    return fit_gain(sample.astype(np.float64) - center.astype(np.float64))


def value_cache(value, center, initial, policy, gain):
    if gain is None:
        return encode(value, center, initial, policy, False)['real']
    # Match scaled FP16 coordinate storage, including exact regions. Projection
    # weight rounding is NOT simulated here and requires a full-model experiment.
    scaled = (value.astype(np.float32) / gain.astype(np.float32)).astype(np.float16)
    scaled_center = scaled[:initial].astype(np.float32).mean(0).astype(np.float16)
    if not np.isfinite(scaled).all():
        raise ValueError('Nonfinite scaled cache')
    return encode(scaled, scaled_center, initial, policy, False)['real'] * gain.astype(np.float64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-run', required=True, type=Path)
    args = parser.parse_args()
    source = args.capture_run.resolve()
    manifest = json.loads((source / 'manifest.json').read_text())
    captures = json.loads((source / 'capture_summary.json').read_text())
    plan = manifest['plan']
    if plan['wikitext_member'] != 'wikitext-2-raw/wiki.train.raw':
        raise ValueError('TRAIN only')
    if captures['manifest_sha256'] != manifest['manifest_sha256']:
        raise ValueError('Capture manifest mismatch')
    records = captures['records']
    expected = {(layer, step) for layer in plan['layers'] for step in plan['generated_token_snapshots']}
    if len(records) != len(expected) or {(r['layer'], r['generated_tokens']) for r in records} != expected:
        raise ValueError('Incomplete cohort')
    for record in records:
        if Path(record['path']).name != record['path'] or digest(source / record['path']) != record['sha256']:
            raise ValueError('Changed capture')
    variants = ['baseline', 'fixed_first4096', 'fixed_full_prefill']
    initial = plan['initial_context_tokens']
    first_step = min(plan['generated_token_snapshots'])
    out = ROOT / 'results/mlsys2027_representation_v2' / ('fixed_gain_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8])
    out.mkdir(parents=True)
    sources = [Path(__file__), HERE / 'prototype.py', HERE.parent / 'robustness_v1/affine_reference.py', HERE / 'run.sh']
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    write_json(out / 'manifest.json', {'schema_version': 1, 'stage': 'E0_attention_calibration_budget_pilot',
        'source_sha256': hashes, 'capture_manifest_sha256': digest(source / 'manifest.json'),
        'capture_summary_sha256': digest(source / 'capture_summary.json'), 'capture_records': records,
        'variants': variants, 'calibration_tokens': [4096, initial], 'calibration_snapshot': first_step,
        'evaluation_snapshots': [s for s in plan['generated_token_snapshots'] if s != first_step],
        'scope': 'Existing exposed TRAIN trajectory; fixed-gain temporal replay, not independent-domain transfer',
        'policy': plan['policy'], 'gpu_used': False, 'production_default_changed': False})
    print('Fixed-gain pilot: ' + str(out), flush=True)
    started = time.perf_counter()
    gains = {}
    for record in records:
        if record['generated_tokens'] != first_step:
            continue
        with np.load(source / record['path'], allow_pickle=False) as data:
            gains[record['layer']] = {variants[1]: calibrated_gain(data['value'], 4096),
                                      variants[2]: calibrated_gain(data['value'], initial)}
    np.savez(out / 'gains.npz', **{f'layer{layer}_{name}': gain for layer, entries in gains.items() for name, gain in entries.items()})
    rows = []
    for record in records:
        if record['generated_tokens'] == first_step:
            continue
        with np.load(source / record['path'], allow_pickle=False) as data:
            q, k, v = data['query'], data['key'], data['value']
            keys = encode(k, data['key_center'], initial, plan['policy'], False)['real']
            values = {name: value_cache(v, data['value_center'], initial, plan['policy'],
                      None if name == 'baseline' else gains[record['layer']][name]) for name in variants}
            hkv = k.shape[1]
            if q.shape[0] % hkv:
                raise ValueError('Invalid GQA layout')
            group = q.shape[0] // hkv
            for head in range(hkv):
                query = q[head*group:(head+1)*group]
                truth = attention(query, k[:, head], v[:, head])
                row = {'capture': record['path'], 'layer': record['layer'], 'step': record['generated_tokens'], 'head': head, 'variants': {}}
                for name in variants:
                    row['variants'][name] = output_metrics(attention(query, keys[:, head], values[name][:, head]), truth)
                rows.append(row)
        print('Evaluated ' + record['path'], flush=True)
    baseline = np.array([r['variants']['baseline']['l2_per_query'] for r in rows]).flatten()
    summary = {}
    for name in variants:
        metrics = [r['variants'][name] for r in rows]
        l2 = np.array([m['l2_per_query'] for m in metrics]).flatten()
        summary[name] = {'mean_l2': float(l2.mean()), 'mean_l2_ratio_to_baseline': float(l2.mean()/baseline.mean()),
            'min_cosine': min(c for m in metrics for c in m['cosine_per_query'] if c is not None),
            'queries_improved': int((l2 < baseline-1e-12).sum()), 'queries_worsened': int((l2 > baseline+1e-12).sum()),
            'queries_unchanged': int((abs(l2-baseline) <= 1e-12).sum()), 'query_count': len(l2)}
    for name, sha in hashes.items():
        if digest(ROOT / name) != sha:
            raise ValueError('Source changed during experiment')
    result = {'summary': summary, 'rows': rows, 'gain_sha256': digest(out / 'gains.npz'),
        'elapsed_seconds': time.perf_counter()-started, 'full_model_quality_claim': False,
        'speed_claim': False, 'cross_document_transfer_tested': False, 'production_default_changed': False,
        'next_required': 'Independent development windows; FP16 folded-weight control; full-model quality and complete costs'}
    write_json(out / 'analysis.json', result)
    print(json.dumps({'summary': summary, 'elapsed_seconds': result['elapsed_seconds']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
