"""CPU-side compatibility check. Does not construct a model or run a GPU kernel."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

SOURCE = Path('/home/anonymous/pagegauge_baselines/KIVI')
sys.path.insert(0, str(SOURCE))
import torch
import transformers
import tokenizers
import kivi_gemv
from models.mistral_kivi import MistralForCausalLM_KIVI

print(json.dumps({'source_commit':subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip(),
    'torch':torch.__version__, 'transformers':transformers.__version__, 'tokenizers':tokenizers.__version__,
    'kernel_path':kivi_gemv.__file__, 'kernel_sha256':hashlib.sha256(Path(kivi_gemv.__file__).read_bytes()).hexdigest(),
    'model_class':MistralForCausalLM_KIVI.__name__, 'gpu_executed':False}, indent=2))
