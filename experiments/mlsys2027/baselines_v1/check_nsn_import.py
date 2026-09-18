"""CPU-only import/provenance check, not a native execution or quality result."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

SOURCE = Path('/home/anonymous/pagegauge_baselines/NSNQuant')
sys.path.insert(0, str(SOURCE))
import torch
import transformers
import tokenizers
import nsn_tools
import fast_hadamard_transform_cuda
from src.quantizers.nsn_quantizer import NSNQuantizer
from src.models.mistral import QuantizedMistralForCausalLM

assert transformers.__version__ == '4.48.1'
assert tokenizers.__version__ == '0.21.4'
assert not torch.cuda.is_initialized(), 'Import unexpectedly initialized CUDA'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

print(json.dumps({
    'source_commit': subprocess.check_output(
        ['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip(),
    'torch': torch.__version__, 'transformers': transformers.__version__,
    'tokenizers': tokenizers.__version__,
    'native_binaries': {str(module.__file__): sha(module.__file__)
                        for module in (nsn_tools, fast_hadamard_transform_cuda)},
    'codebook_sha256': {str(p.relative_to(SOURCE)): sha(p)
                       for p in sorted((SOURCE/'codebooks').glob('*bit_codebook.pt'))},
    'model_class': QuantizedMistralForCausalLM.__name__,
    'quantizer_class': NSNQuantizer.__name__,
    'gpu_executed': False,
    'scope': 'SM120 portability build/import only; native correctness pending',
}, indent=2))
