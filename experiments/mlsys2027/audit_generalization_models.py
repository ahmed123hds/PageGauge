"""Resolve model revisions and download config ONLY; no weights, corpus or GPU.

Uses existing Hugging Face access, if configured. Does not request gated access,
accept agreements, or print credentials. Unavailable models remain unavailable.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid
from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
REPOS = ('Qwen/Qwen3-8B', 'meta-llama/Llama-3.1-8B')


def main():
    out = ROOT/'results/mlsys2027_generalization_v1'/('model_audit_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True)
    rows = []
    for repo in REPOS:
        row = {'repo': repo, 'config_available': False, 'weights_downloaded': False}
        try:
            info = HfApi().model_info(repo)
            row.update(revision=info.sha, gated=info.gated)
            path = Path(hf_hub_download(repo, 'config.json', revision=info.sha, local_files_only=False))
            config = json.loads(path.read_text())
            row.update(config_available=True, config_path=str(path),
                       config_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), config=config)
            dim = config.get('head_dim', config['hidden_size']//config['num_attention_heads'])
            row['fp16_kv_bytes_B1_C22016'] = config['num_hidden_layers']*22016*config['num_key_value_heads']*dim*4
            row['fp16_kv_bytes_B4_C22016'] = row['fp16_kv_bytes_B1_C22016']*4
            row['cache_estimate_scope'] = 'Dense FP16 KV only; not peak memory or feasible-batch attestation'
        except Exception as exc:
            row['error_type'] = type(exc).__name__
            row['http_status'] = getattr(getattr(exc, 'response', None), 'status_code', None)
        rows.append(row)
        print(json.dumps({k: row[k] for k in ('repo', 'config_available', 'revision', 'gated', 'error_type', 'http_status') if k in row}), flush=True)
    result = {'rows': rows, 'scope': 'Configuration/access audit only; native integration and model tests pending',
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'analysis.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print('Model audit: '+str(out), flush=True)


if __name__ == '__main__':
    main()
