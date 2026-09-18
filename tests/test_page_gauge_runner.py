from __future__ import annotations

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "page_gauge_validation_runner", ROOT / "scripts/run_page_gauge_validation.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_scheduler_timing_defaults_to_cache_neutral(tmp_path: Path) -> None:
    artifact = tmp_path / "timing.json"
    artifact.write_text(
        json.dumps(
            {
                "timing_modes": {
                    "cache_neutral": {
                        "timings": {
                            "flashinfer_fp16": {"p50_ms": 10.0},
                            "page_gauge": {"p50_ms": 5.0},
                        }
                    },
                    "cache_hot": {
                        "timings": {
                            "flashinfer_fp16": {"p50_ms": 4.0},
                            "page_gauge": {"p50_ms": 1.0},
                        }
                    },
                }
            }
        )
    )

    assert MODULE.load_timing(artifact) == (10.0, 5.0)
    assert MODULE.load_timing(artifact, "cache_hot") == (4.0, 1.0)


def test_benchmark_command_propagates_cache_scrub_size(tmp_path: Path) -> None:
    command = MODULE.benchmark_command(
        tmp_path / "result.json",
        seed=7,
        repeats=11,
        warmup=3,
        baseline_split=128,
        candidate_split=64,
        cache_scrub_mib=384,
    )
    index = command.index("--cache-scrub-mib")
    assert command[index + 1] == "384"


def test_run_arguments_are_json_serializable(tmp_path: Path) -> None:
    payload = MODULE.serializable_arguments(
        Namespace(output_dir=tmp_path / "run", seeds=(1, 2), required=True)
    )
    assert payload == {
        "output_dir": str(tmp_path / "run"),
        "seeds": [1, 2],
        "required": True,
    }
    json.dumps(payload)
