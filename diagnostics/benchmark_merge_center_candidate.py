"""Synthetic CUDA-graph microbenchmark; not full-model speed evidence."""
import argparse
import json
from pathlib import Path
import statistics
import torch
import flashinfer
from merge_center_candidate import merge_center


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Output must be new')
    torch.manual_seed(20260910)
    rows = []
    for batch in (1, 4, 16):
        h = torch.randn((batch, 32, 128), device='cuda', dtype=torch.float16)
        e, c, out = torch.randn_like(h), torch.randn_like(h), torch.empty_like(h)
        a = torch.randn((batch, 32), device='cuda')
        b = torch.randn_like(a)
        initial_a = a.clone()
        def old():
            flashinfer.merge_state_in_place(out, a, e, b)
            out.add_(c)
        def candidate():
            merge_center(out, e, a, b, c, out)
        graphs = {}
        for name, fn in (('existing', old), ('candidate', candidate)):
            for _ in range(10):
                out.copy_(h)
                a.copy_(initial_a)
                fn()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                # Identical reset in both graphs prevents accumulating centers.
                for _ in range(100):
                    out.copy_(h)
                    a.copy_(initial_a)
                    fn()
            graphs[name] = graph
        samples = {name: [] for name in graphs}
        for _ in range(8):
            for name in ('existing', 'candidate', 'candidate', 'existing'):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                graphs[name].replay()
                end.record()
                end.synchronize()
                samples[name].append(start.elapsed_time(end)/100)
        rows.append({'batch': batch, 'ms_per_operation_including_identical_reset':
                     {name: statistics.median(values) for name, values in samples.items()},
                     'samples_ms': samples})
    result = {'scope': 'Synthetic graph microbenchmark including identical copy reset; not fresh-process or end-to-end speed gate',
              'gpu': torch.cuda.get_device_name(), 'rows': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
