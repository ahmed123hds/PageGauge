"""CPU-only import/provenance check. Does not establish SM120 kernel support."""
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import uuid

ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path('/home/anonymous/pagegauge_baselines/Kitty')


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '-1':
        raise RuntimeError('Import check must not expose GPU')
    import torch
    import transformers
    import tokenizers
    import triton
    from kitty.models.qwen3 import Qwen3ForCausalLM_Kitty
    from kitty.kvcache import get_kvcache_kitty
    from kitty.kvcache.kitty import KittyCache
    fork = SOURCE/'third_party/transformers'
    if subprocess.check_output(['git', '-C', str(fork), 'rev-parse', 'HEAD'], text=True).strip() != '37f8b0b53512e6aae0cfd15746c133c101783178':
        raise ValueError('Wrong native Transformers fork')
    package_root = Path(transformers.__file__).parent
    import transformers.models.qwen3.modeling_qwen3 as qwen
    import kitty.kvcache.kernels.kitty_attention as attention
    files = [Path(inspect.getfile(Qwen3ForCausalLM_Kitty)), Path(inspect.getfile(KittyCache)),
             Path(inspect.getfile(get_kvcache_kitty)), Path(qwen.__file__), Path(attention.__file__),
             package_root/'cache_utils.py', package_root/'masking_utils.py']
    files.append(SOURCE/'src/kitty/kvcache/kernels/kitty_quant_pack.py')
    evidence = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    # Check the installed model/cache/mask modules are exactly from the pinned fork.
    for relative in ('models/qwen3/modeling_qwen3.py', 'cache_utils.py', 'masking_utils.py'):
        if (package_root/relative).read_bytes() != (fork/'src/transformers'/relative).read_bytes():
            raise ValueError('Installed fork differs: '+relative)
    out = ROOT/'results/mlsys2027_baselines_v1'/('kitty_import_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True)
    result = {'torch': torch.__version__, 'transformers': transformers.__version__,
        'tokenizers': tokenizers.__version__, 'triton': triton.__version__,
        'source_commit': subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip(),
        'transformers_fork_commit': '37f8b0b53512e6aae0cfd15746c133c101783178',
        'native_source_diff': subprocess.check_output(['git', '-C', str(SOURCE), 'diff', '--', 'src/kitty'], text=True),
        'installed_file_sha256': evidence, 'gpu_executed': False,
        'scope': 'Native Python import compatibility only; no SM120 compile, numerical correctness, quality or speed claim.'}
    (out/'analysis.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print('Kitty import check: '+str(out), flush=True)
    print(json.dumps({k: result[k] for k in ('torch', 'transformers', 'tokenizers', 'triton', 'gpu_executed')}), flush=True)


if __name__ == '__main__':
    main()
