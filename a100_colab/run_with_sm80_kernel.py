#!/usr/bin/env python3
"""Run an existing PageGauge diagnostic with an opt-in SM80 launch cap.

Only the PageGauge wrapper factory is replaced.  Cache representation,
quantization, attention equations, exact regions, and all acceptance gates
remain those of the target diagnostic.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib.util
import json
import math
import sys
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import page_gauge_sm80_fa2 as SM80  # noqa: E402


def load_local_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parse_launcher_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cap",
        type=int,
        choices=(0, *SM80.SUPPORTED_CAPS),
        default=0,
        help="0 keeps the frozen MMA dispatch; 1/2 cap PageGauge paged FA2",
    )
    parser.add_argument(
        "--exact-split-pages",
        type=int,
        default=0,
        help="A100-only split override for the existing exact FP16 wrapper",
    )
    parser.add_argument("--target", choices=("micro", "sustained"), required=True)
    parser.add_argument("target_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    target_args = list(args.target_args)
    if target_args and target_args[0] == "--":
        target_args.pop(0)
    if not target_args:
        parser.error("target arguments are required after --")
    return args, target_args


def output_path(arguments: list[str]) -> Path:
    try:
        index = arguments.index("--output")
        value = arguments[index + 1]
    except (ValueError, IndexError) as error:
        raise SystemExit("target arguments must contain --output PATH") from error
    return Path(value)


def attach_provenance(
    path: Path, *, cap: int, exact_split_pages: int, target: str
) -> None:
    if not path.is_file():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    source_files = {
        "scripts/page_gauge_sm80_fa2.py": Path(SM80.__file__).resolve(),
        "a100_colab/run_with_sm80_kernel.py": Path(__file__).resolve(),
    }
    effective_capacity = None
    if target == "sustained" and exact_split_pages:
        config = payload["configuration"]
        environment = payload["environment"]
        exact_pages = (
            int(config["exact_tail_tokens"]) // 16
            + int(config["exact_prefix_pages"])
            + int(config["exact_static_suffix_pages"])
        )
        maximum_batches = (2 * int(environment["multiprocessor_count"])) // 8
        maximum_chunks = maximum_batches // int(config["batch_size"])
        required_chunks = math.ceil(exact_pages / exact_split_pages)
        effective_capacity = {
            "exact_pages_per_request": exact_pages,
            "fixed_split_pages": exact_split_pages,
            "required_chunks_per_request": required_chunks,
            "required_split_tiles": required_chunks * int(config["batch_size"]),
            "maximum_split_tiles": maximum_batches,
            "within_flashinfer_scheduler_capacity": (
                required_chunks <= maximum_chunks
            ),
        }
        if not effective_capacity["within_flashinfer_scheduler_capacity"]:
            raise RuntimeError("effective exact split exceeds FlashInfer capacity")
    payload["a100_sm80_kernel_override"] = {
        "target": target,
        "sm80_paged_num_mma_kv_cap": cap or None,
        "module_uri": SM80.module_uri(cap) if cap else None,
        "source_hashes": SM80.source_hashes(cap) if cap else None,
        "exact_fp16_fixed_split_pages": exact_split_pages or None,
        "effective_exact_scheduler_capacity": effective_capacity,
        "launcher_source_sha256": {
            name: hashlib.sha256(file.read_bytes()).hexdigest()
            for name, file in source_files.items()
        },
        "page_gauge_math_changed": False,
        "scope": "paged PageGauge FA2 launch specialization only",
    }
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def main() -> None:
    args, target_args = parse_launcher_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    capability = tuple(torch.cuda.get_device_capability())
    name = torch.cuda.get_device_name()
    if capability != (8, 0) or "A100" not in name.upper():
        raise SystemExit(
            f"SM80 launch-cap probe requires an A100 (got {name!r}, CC {capability})"
        )

    if args.exact_split_pages < 0:
        raise SystemExit("--exact-split-pages must be nonnegative")
    factory = (
        functools.partial(
            SM80.make_page_gauge_wrapper, sm80_num_mma_kv_cap=args.cap
        )
        if args.cap
        else None
    )
    if args.target == "micro":
        target = SM80.LEGACY
        if factory is not None:
            target.make_page_gauge_wrapper = factory
        if args.exact_split_pages:
            original_plan = target.plan
            fp16_plan_calls = 0

            def split_exact_plan(*plan_args, **plan_kwargs):
                nonlocal fp16_plan_calls
                if plan_args[4] == torch.float16:
                    fp16_plan_calls += 1
                    # First FP16 plan: full-cache baseline. Second: exact region.
                    if fp16_plan_calls == 2:
                        plan_args = (
                            *plan_args[:5],
                            args.exact_split_pages,
                            *plan_args[6:],
                        )
                return original_plan(*plan_args, **plan_kwargs)

            target.plan = split_exact_plan
    else:
        target = load_local_module(
            "page_gauge_sustained_sm80_launcher",
            ROOT / "diagnostics/benchmark_sustained_dynamic_graphs.py",
        )
        if factory is not None:
            target.PG.PAGE_KERNEL.make_page_gauge_wrapper = factory
        if args.exact_split_pages:
            original_plan = target.PG.GraphDecodeWrapper.plan

            def split_exact_plan(
                self,
                physical_indices,
                logical_tokens,
                last_page_len,
                split_pages,
                page_table_epoch=None,
            ):
                if self.kv_dtype == torch.float16:
                    split_pages = args.exact_split_pages
                return original_plan(
                    self,
                    physical_indices,
                    logical_tokens,
                    last_page_len,
                    split_pages,
                    page_table_epoch,
                )

            target.PG.GraphDecodeWrapper.plan = split_exact_plan

    destination = output_path(target_args)
    previous_argv = sys.argv
    status = 0
    try:
        sys.argv = [str(Path(target.__file__).resolve()), *target_args]
        target.main()
    except SystemExit as error:
        status = int(error.code or 0) if isinstance(error.code, int) else 1
    except BaseException:  # preserve the target's failure artifact contract
        traceback.print_exc()
        status = 1
    finally:
        sys.argv = previous_argv
        attach_provenance(
            destination,
            cap=args.cap,
            exact_split_pages=args.exact_split_pages,
            target=args.target,
        )
    raise SystemExit(status)


if __name__ == "__main__":
    main()
