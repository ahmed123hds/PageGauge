#!/usr/bin/env bash
# CPU-only isolated Python packages; no main-environment changes or GPU tests.
set -euo pipefail
SOURCE=/home/anonymous/pagegauge_baselines/Kitty
TARGET=/home/anonymous/pagegauge_baselines/kitty_sm120_env
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
test "$(git -C "$SOURCE" rev-parse HEAD)" = dfd2c07b407d6b407179359207c612ab631f3ed1
test "$(git -C "$SOURCE/third_party/transformers" rev-parse HEAD)" = 37f8b0b53512e6aae0cfd15746c133c101783178
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
if [ ! -f "$TARGET/pyvenv.cfg" ]; then
  /home/anonymous/page_gauge_env_protocol_v2/bin/python -m venv "$TARGET"
fi
cp "$HERE/nsn_shared_runtime.pth" "$TARGET/lib/python3.12/site-packages/pagegauge_shared_runtime.pth"
"$TARGET/bin/python" -m pip install --no-deps tokenizers==0.21.4
"$TARGET/bin/python" -m pip install --no-deps --no-build-isolation "$SOURCE/third_party/transformers"
# Upstream documents editable installation; its regular wheel omits the native
# namespace package in this environment. Keep the released source unchanged.
"$TARGET/bin/python" -m pip install --no-deps --no-build-isolation -e "$SOURCE"
"$TARGET/bin/python" "$HERE/check_kitty_import.py"
