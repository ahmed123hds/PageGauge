"""Freeze two TRAIN fixtures and ABBA/BAAB order, qualify second fixture."""
from datetime import datetime, timezone
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import uuid
import frontier_layer_segments as validation
import frontier_layer_timing_v2 as timing
from frontier_layer_segments import ROOT, base, previous, wsl_gpu_monitor, verify, PYTHON


def main():
    import fcntl
    import benchmark_pg19_external_quality as quality
    directory = ROOT/'results/mlsys2027_baselines_v1'
    sources = [directory/'layer_segments_flashinfer_fp16_20260909T190907Z_7b024ad7',
               directory/'layer_segments_page_gauge_20260909T191435Z_c7175f1b']
    manifests = [json.loads((s/'manifest.json').read_text()) for s in sources]
    for m in manifests:
        verify(m)
    args = SimpleNamespace(context=20480, decode_steps=1536, batch_size=4,
        token_source='wikitext2', wikitext_member='wikitext-2-raw/wiki.train.raw',
        wikitext_zip=ROOT/'data/wikitext-2-raw-v1.zip', seed=2026090915,
        token_offset=612000, token_stride=35000, model=manifests[0]['model'])
    ids, provenance = quality.build_token_matrix(args, 32768)
    assert provenance['split'] == 'train'
    starts = provenance['corpus_window_start_offsets']
    assert starts == [612000, 647000, 682000, 717000]
    assert min(starts) >= 577000+22016
    out = directory/('layer_replication_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    out.mkdir()
    base.atomic_json(out/'fixture1_tokens.json', {'ids': ids.tolist()})
    frozen = {'scope': __doc__+' Development only, not held-out TEST. Two fixture clusters, four adjacent pairs; hierarchical bootstrap on fixture then pair, 50000 draws, seed2026090915, ratio of geometric means of per-process median wall times. No token-level resampling.',
        'order': [{'fixture': f, 'backend': b} for f, seq in [(0, ('flashinfer_fp16','page_gauge','page_gauge','flashinfer_fp16')), (1, ('page_gauge','flashinfer_fp16','flashinfer_fp16','page_gauge'))] for b in seq],
        'source_sha256': {**manifests[0]['source_sha256'], **manifests[1]['source_sha256']},
        'input_file_evidence': manifests[0]['input_file_evidence'],
        'fixture0_validations': [str(s) for s in sources],
        'fixture1_tokens_sha256': base.sha256_file(out/'fixture1_tokens.json'),
        'fixture1_provenance': provenance, 'timing_repeats': 3, 'warmup_rounds': 1}
    for path in (Path(__file__), Path(timing.__file__), Path(validation.__file__)):
        frozen['source_sha256'][str(path.resolve())] = base.sha256_file(path)
    base.atomic_json(out/'manifest.json', frozen)
    print('Frozen layer replication: '+str(out), flush=True)
    for m in manifests:
        idle = base.idle_preflight(0); gpu = idle[-1]['uuid']
        with (Path('/tmp')/f'pagegauge-mlsys-{gpu}.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            target = out/('validation1_'+m['backend']); target.mkdir()
            (target/'tokens.json').write_bytes((out/'fixture1_tokens.json').read_bytes())
            cm = copy.deepcopy(m)
            cm.update(token_provenance=provenance, tokens_sha256=base.sha256_file(target/'tokens.json'),
                      source_sha256=frozen['source_sha256'], idle=idle, validate=True, repeats=0)
            base.atomic_json(target/'manifest.json', cm)
            os.environ.pop('PYTORCH_CUDA_ALLOC_CONF', None)
            os.environ['PYTORCH_ALLOC_CONF'] = cm['allocator_configuration']
            command = [PYTHON, '-u', validation.__file__, '--worker', str(target)]
            base.atomic_json(target/'invocation.json', {'command': command})
            c = wsl_gpu_monitor.run_process(previous, command, target, {'index': 0}, gpu)
            base.atomic_json(target/'completion.json', c)
            verify(frozen)
            if c['return_code'] or not c['sampled_exclusivity_passed']:
                raise ValueError('Second fixture validation failed; stop before replication')
    base.atomic_json(out/'qualification.json', {'second_fixture_execution_complete': True})


if __name__ == '__main__':
    main()
