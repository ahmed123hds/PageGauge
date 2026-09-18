"""Versioned Qwen TRAIN replay with fused merge candidate, not final evaluation."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'experiments/mlsys2027/generalization_v1'))
sys.path.insert(0, str(ROOT/'experiments/mlsys2027/ablation_v1'))


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    import qwen_quality as worker
    import benchmark_pg19_external_quality as quality
    from regional_quality import install_probe
    from install_merge_center_candidate import install
    source, out = args.source.resolve(), args.output.resolve()
    m = json.loads((source/'manifest.json').read_text())
    if (m['token_provenance']['split'], m['context'], m['decode_steps']) != ('train', 20480, 1536):
        raise ValueError('Retained TRAIN long-rollout fixture required')
    if worker.base.sha256_file(source/'tokens.json') != m['tokens_sha256']:
        raise ValueError('Retained tokens changed')
    if out.exists():
        raise ValueError('Output must be new')
    changes = {}
    for filename, old in list(m['source_sha256'].items()):
        current = worker.base.sha256_file(Path(filename))
        if current != old:
            changes[filename] = {'original': old, 'current': current}
        m['source_sha256'][filename] = current
    for filename in (Path(__file__), Path(__file__).with_name('merge_center_candidate.py'),
                     Path(__file__).with_name('install_merge_center_candidate.py')):
        m['source_sha256'][str(filename.resolve())] = worker.base.sha256_file(filename)
    worker.verify(m)
    idle = worker.base.idle_preflight(0)
    out.mkdir(parents=True)
    shutil.copyfile(source/'tokens.json', out/'tokens.json')
    m.update(scope=__doc__, source_fixture=str(source), source_changes=changes, idle=idle)
    worker.base.atomic_json(out/'manifest.json', m)
    install(quality.PG.TransformerDecoder)
    original_init = quality.PG.TransformerDecoder.__init__
    probes = []
    def init(self, *a, **kw):
        original_init(self, *a, **kw)
        if a[3] == 'page_gauge':
            install_probe(self, m['context'], m['decode_steps'], probes)
    quality.PG.TransformerDecoder.__init__ = init
    try:
        worker.worker(out)
    finally:
        worker.base.atomic_json(out/'candidate_probes.json', {'probes': probes,
            'scope': 'Selected same-cache checks, unchanged limits; TRAIN only'})


if __name__ == '__main__':
    main()
