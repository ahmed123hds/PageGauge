#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

export PIP_DISABLE_PIP_VERSION_CHECK=1
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0

python -m pip install --quiet --upgrade -r a100_colab/requirements.txt

export TORCH_CUDA_ARCH_LIST="8.0+PTX"
export FLASHINFER_CUDA_ARCH_LIST="8.0"
export TORCH_EXTENSIONS_DIR="$ROOT/build/torch_extensions_a100_sm80"
export MAX_JOBS=${MAX_JOBS:-2}
mkdir -p "$TORCH_EXTENSIONS_DIR" results/a100_cross_gpu_s4_a128_t768

python a100_colab/preflight_a100.py \
  --output results/a100_cross_gpu_s4_a128_t768/a100_preflight.json

if [[ "${PAGEGAUGE_SKIP_MODEL_DOWNLOAD:-0}" != "1" ]]; then
  python - <<'PY'
from pathlib import Path

from huggingface_hub import snapshot_download

revision = "caa1feb0e54d415e2df31207e5f4e273e33509b1"
snapshot = Path(snapshot_download(
    repo_id="mistralai/Mistral-7B-v0.3",
    revision=revision,
    ignore_patterns=["*.gguf", "*.bin", "consolidated.safetensors", "params.json"],
)).resolve()
if snapshot.name != revision:
    raise RuntimeError(f"resolved unexpected model revision: {snapshot}")

# The frozen workers intentionally load with local_files_only=True and the
# public repository ID.  An explicit-commit snapshot download does not always
# materialize refs/main, so bind the local default ref to the frozen commit.
# This performs no network lookup and prevents a future upstream main revision
# from entering the experiment.
main_ref = snapshot.parents[1] / "refs" / "main"
main_ref.parent.mkdir(parents=True, exist_ok=True)
main_ref.write_text(revision, encoding="utf-8")
print(f"Pinned local model ref: {main_ref} -> {revision}")
PY
fi

python scripts/prepare_flashinfer_page_gauge.py

python - <<'PY'
import flashinfer
import torch
import transformers

assert flashinfer.__version__ == "0.6.17", flashinfer.__version__
assert transformers.__version__ == "4.57.6", transformers.__version__
assert torch.cuda.get_device_capability() == (8, 0)
print("Pinned software and A100 sm_80 preflight: PASS")
PY
