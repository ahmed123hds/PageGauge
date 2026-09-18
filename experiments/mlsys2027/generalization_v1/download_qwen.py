"""Download the audited Qwen3-8B revision, weights/tokenizer only; no GPU work."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import uuid
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[3]
REPO = 'Qwen/Qwen3-8B'
REVISION = 'b968826d9c46dd6066d109eabc6255188de91218'
PATTERNS = ['*.safetensors', '*.safetensors.index.json', 'config.json',
            'generation_config.json', 'tokenizer*', 'vocab.json', 'merges.txt',
            'special_tokens_map.json', 'added_tokens.json', 'chat_template.jinja',
            'README.md', 'LICENSE']


def main():
    cache = Path('/home/anonymous/.cache/huggingface/hub')
    if shutil.disk_usage(cache).free < 40*2**30:
        raise RuntimeError('Insufficient free space for model and evaluation headroom')
    out = ROOT/'results/mlsys2027_generalization_v1'/('qwen_download_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True)
    manifest = {'repo': REPO, 'revision': REVISION, 'allow_patterns': PATTERNS,
                'scope': 'Model preparation only; no inference, evaluation, or final TEST access',
                'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('Qwen download: '+str(out), flush=True)
    snapshot = Path(snapshot_download(REPO, revision=REVISION, allow_patterns=PATTERNS,
                                     cache_dir=str(cache), max_workers=2, local_files_only=False))
    evidence = {}
    for path in sorted(snapshot.iterdir()):
        if path.is_file():
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(8*1024*1024), b''):
                    digest.update(chunk)
            evidence[path.name] = {'size': path.stat().st_size, 'sha256': digest.hexdigest()}
    index = json.loads((snapshot/'model.safetensors.index.json').read_text())
    if not set(index['weight_map'].values()).issubset(evidence):
        raise RuntimeError('Missing weight shard')
    result = {'snapshot': str(snapshot), 'repo': REPO, 'revision': REVISION,
              'file_evidence': evidence, 'total_file_bytes': sum(row['size'] for row in evidence.values()),
              'gpu_executed': False, 'scope': manifest['scope']}
    (out/'analysis.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print('Verified Qwen snapshot: '+str(snapshot), flush=True)


if __name__ == '__main__':
    main()
