#!/usr/bin/env python3
"""Shared runtime helpers for PageGauge serving experiments."""

from __future__ import annotations

import os
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
APPEND_EXTENSION = ROOT / "tests/page_gauge_append_extension.cu"


def load_append_extension():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the PageGauge append extension")
    major, minor = torch.cuda.get_device_capability()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}")
    from torch.utils.cpp_extension import load

    return load(
        name="page_gauge_append_ext",
        sources=[str(APPEND_EXTENSION)],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--threads=2"],
        verbose=True,
    )
