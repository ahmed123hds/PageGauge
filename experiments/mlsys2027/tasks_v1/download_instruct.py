"""Pinned public instruction-model preparation only; no benchmark data/GPU."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import uuid
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[3]
REPO = 'mistralai/Mistral-7B-Instruct-v0.3'
REVISION = 'c170c708c41dac9275d15a8fff4eca08d52bab71'
PATTERNS = ['model-*-of-*.safetensors', 'model.safetensors.index.json', 'config.json',
    'generation_config.json', 'tokenizer*', 'special_tokens_map.json',
    'added_tokens.json', 'chat_template.jinja', 'README.md', 'LICENSE*']


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def main():
    cache = Path('/home/anonymous/.cache/huggingface/hub')
    if shutil.disk_usage(cache).free < 40*2**30:
        raise RuntimeError('Need model download and evaluation disk headroom')
    out = ROOT/'results/mlsys2027_tasks_v1'/('instruct_download_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True)
    manifest = {'repo': REPO, 'revision': REVISION, 'allow_patterns': PATTERNS,
        'source_sha256': digest(Path(__file__)),
        'scope': 'Public Apache-2.0 model preparation only; no gated access acceptance, benchmark data or GPU use'}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('Instruction model preparation: '+str(out), flush=True)
    snapshot = Path(snapshot_download(REPO, revision=REVISION, allow_patterns=PATTERNS,
        cache_dir=str(cache), max_workers=2, local_files_only=False))
    evidence = {p.name: {'size': p.stat().st_size, 'sha256': digest(p)} for p in sorted(snapshot.iterdir()) if p.is_file()}
    index = json.loads((snapshot/'model.safetensors.index.json').read_text())
    if not set(index['weight_map'].values()).issubset(evidence):
        raise RuntimeError('Incomplete checkpoint')
    result = {**manifest, 'snapshot': str(snapshot), 'file_evidence': evidence,
        'config': json.loads((snapshot/'config.json').read_text()), 'gpu_executed': False}
    (out/'analysis.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Verified instruction checkpoint: '+str(snapshot), flush=True)


if __name__ == '__main__':
    main()
