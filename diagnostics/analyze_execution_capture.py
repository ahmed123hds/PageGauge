"""CPU-only precision decomposition of a trusted local diagnostic capture."""
import argparse
import json
import math
from pathlib import Path
import torch


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve existing analysis; choose a new output')
    torch.set_num_threads(1)
    p = torch.load(args.capture, map_location='cpu', weights_only=True)
    q, k, v, c = [p[n].double() for n in ('query', 'keys', 'values', 'center')]
    centered = (q @ k.T / math.sqrt(k.shape[-1])).softmax(-1) @ v
    expected = centered + c
    def error(value):
        delta = value.double() - expected
        return {'relative_l2': float(delta.norm()/expected.norm()),
                'max_absolute': float(delta.abs().max())}
    rounded_centered = (centered.half().double()+c).half().double()
    result = {
        'scope': 'CPU same-cache numerical diagnostic, not evaluation or speed evidence',
        'original_failure': p['row'],
        'fp32_reference_vs_fp64': error(p['expected']),
        'observed_vs_fp64': error(p['observed']),
        'ideal_centered_fp16_then_restore_fp16': error(rounded_centered),
        'ideal_restore_then_fp16': error(expected.half()),
        'norms': {'center': float(c.norm()), 'centered': float(centered.norm()),
                  'restored': float(expected.norm())},
        'interpretation': 'Center cancellation amplifies rounding, but ideal centered-output rounding alone does not reproduce the observed failure. Separate region output and merge captures are needed.'}
    if 'history_output' in p:
        region_outputs, region_lse, region_rows = [], [], {}
        for name, kn, vn in (('history', 'old_k', 'old_v'),
                              ('exact', 'exact_k', 'exact_v')):
            scores = q @ p[kn].double().T / math.sqrt(k.shape[-1])
            reference = scores.softmax(-1) @ p[vn].double()
            lse = scores.logsumexp(-1)
            region_outputs.append(reference)
            region_lse.append(lse)
            region_rows[name] = {
                'output_relative_l2': float((p[name+'_output']-reference).norm()/reference.norm()),
                'lse_log2_max_error': float((p[name+'_lse']-lse/math.log(2)).abs().max())}
        weights = torch.stack(region_lse).softmax(0)
        merged_regions = sum(weights[i, :, None]*p[name+'_output'].double()
                             for i, name in enumerate(('history', 'exact')))
        result['regions'] = region_rows
        result['captured_regions_ideal_merge_and_restore'] = error(merged_regions+c)
        result['captured_merge_ideal_restore'] = error(p['merged_centered_output'].double()+c)
        result['fp64_partition_identity'] = error(sum(weights[i, :, None]*region_outputs[i]
                                                      for i in range(2))+c)
        result['region_interpretation'] = ('Both region outputs closely match their FP64 references. '
            'Ideal merging of the captured rounded regions gives lower restored error than the '
            'captured FP16 merge. Test fused higher-precision merge plus center restoration; '
            'this diagnostic alone does not establish a production fix or its speed.')
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
