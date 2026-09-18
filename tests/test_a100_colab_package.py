from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ANALYZER_PATH = ROOT / "a100_colab/analyze_a100_cross_gpu.py"


def load_analyzer():
    spec = importlib.util.spec_from_file_location(
        "pagegauge_a100_analyzer", ANALYZER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_colab_notebook_is_valid_json() -> None:
    notebook = json.loads(
        (ROOT / "a100_colab/PageGauge_A100_Colab.ipynb").read_text(encoding="utf-8")
    )
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["colab"]["gpuType"] == "A100"


def test_setup_binds_offline_main_to_frozen_revision() -> None:
    setup = (ROOT / "a100_colab/setup_colab.sh").read_text(encoding="utf-8")
    assert 'revision = "caa1feb0e54d415e2df31207e5f4e273e33509b1"' in setup
    assert 'main_ref = snapshot.parents[1] / "refs" / "main"' in setup
    assert 'main_ref.write_text(revision, encoding="utf-8")' in setup
    assert 'main_ref.write_text(revision + "\\n"' not in setup
    assert '"consolidated.safetensors"' in setup


def test_a100_capacity_derived_candidate_split200() -> None:
    sms, hkv, batch = 108, 8, 4
    max_split_batch = (2 * sms) // hkv
    max_chunks_per_request = max_split_batch // batch
    assert max_chunks_per_request == 6
    assert (1196 + 128 - 1) // 128 == 10
    assert (1196 + 200 - 1) // 200 == 6
    assert (1196 + 256 - 1) // 256 == 5
    assert (1376 + 256 - 1) // 256 == 6


def test_a100_smoke_accepts_small_absolute_near_zero_error(tmp_path: Path) -> None:
    source = tmp_path / "smoke.json"
    output = tmp_path / "gate.json"
    source.write_text(
        json.dumps(
            {
                "environment": {
                    "gpu": "NVIDIA A100-SXM4-40GB",
                    "compute_capability": [8, 0],
                },
                "workload": {
                    "lengths": [20480],
                    "representation": "page_gauge",
                    "exact_tail_tokens": 0,
                    "exact_sink_tokens": 0,
                    "baseline_fixed_split_pages": 256,
                    "candidate_fixed_split_pages": 256,
                },
                "correctness": {
                    "reference": "fp16_reconstructed_cache",
                    "absolute_max": 5.0067901611328125e-6,
                    "relative_l2_max": 0.004843796603381634,
                },
                "timing_modes": {
                    mode: {
                        "timings": {
                            method: {"p50_ms": 0.1}
                            for method in ("flashinfer_fp16", "page_gauge_int8")
                        }
                    }
                    for mode in ("cache_neutral", "cache_hot")
                },
            }
        ),
        encoding="utf-8",
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "a100_colab/validate_kernel_smoke.py"),
            "--input",
            str(source),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    gate = json.loads(output.read_text(encoding="utf-8"))
    assert gate["passed"] is True
    assert gate["timing_is_acceptance_gate"] is False


@pytest.mark.parametrize(
    ("block_name", "backend", "prefix", "suffix"),
    [
        ("block_0_flashinfer_fp16.json", "flashinfer_fp16", 0, 0),
        ("block_1_page_gauge.json", "page_gauge", 4, 128),
    ],
)
def test_a100_worker_validator_accepts_only_ported_identity(
    tmp_path: Path,
    block_name: str,
    backend: str,
    prefix: int,
    suffix: int,
) -> None:
    source = (
        ROOT / "results/final_fixed_suffix_s4_a128_t768/performance/"
        "fresh_process_williams_split128_v2" / block_name
    )
    if not source.is_file():
        pytest.skip("development-only RTX 5090 fixture is not packaged")
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["environment"].update(
        {
            "gpu": "NVIDIA A100-SXM4-40GB",
            "compute_capability": [8, 0],
            "multiprocessor_count": 108,
        }
    )
    payload["configuration"]["candidate_split_pages"] = 200
    payload["pairing"]["configuration"]["candidate_split_pages"] = 200
    assert payload["configuration"]["exact_prefix_pages"] == prefix
    assert payload["configuration"]["exact_static_suffix_pages"] == suffix
    target = tmp_path / block_name
    target.write_text(json.dumps(payload), encoding="utf-8")
    analyzer = load_analyzer()
    validated = analyzer.validate_worker(
        target,
        backend,
        int(payload["configuration"]["seed"]),
        int(payload["configuration"]["token_offset"]),
    )
    assert validated["environment"]["compute_capability"] == [8, 0]


def test_a100_worker_validator_rejects_blackwell_identity(tmp_path: Path) -> None:
    source = (
        ROOT / "results/final_fixed_suffix_s4_a128_t768/performance/"
        "fresh_process_williams_split128_v2/block_0_flashinfer_fp16.json"
    )
    if not source.is_file():
        pytest.skip("development-only RTX 5090 fixture is not packaged")
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["configuration"]["candidate_split_pages"] = 200
    payload["pairing"]["configuration"]["candidate_split_pages"] = 200
    target = tmp_path / source.name
    target.write_text(json.dumps(payload), encoding="utf-8")
    analyzer = load_analyzer()
    with pytest.raises(RuntimeError, match="not A100"):
        analyzer.validate_worker(target, "flashinfer_fp16", 20260861, 0)


def test_full_a100_reducer_contract_on_frozen_fixtures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_dir = (
        ROOT / "results/final_fixed_suffix_s4_a128_t768/performance/"
        "fresh_process_williams_split128_v2"
    )
    if not source_dir.is_dir():
        pytest.skip("development-only RTX 5090 fixtures are not packaged")
    analyzer = load_analyzer()
    for index, backend, _seed, _offset, _pair_id, _order in analyzer.SCHEDULE:
        source = source_dir / f"block_{index}_{backend}.json"
        payload = json.loads(source.read_text(encoding="utf-8"))
        payload["environment"].update(
            {
                "gpu": "NVIDIA A100-SXM4-40GB",
                "compute_capability": [8, 0],
                "multiprocessor_count": 108,
            }
        )
        payload["configuration"]["candidate_split_pages"] = 200
        payload["pairing"]["configuration"]["candidate_split_pages"] = 200
        (tmp_path / source.name).write_text(json.dumps(payload), encoding="utf-8")

    preflight = tmp_path / "preflight.json"
    kernel_gate = tmp_path / "kernel_gate.json"
    output = tmp_path / "analysis.json"
    preflight.write_text('{"passed": true}', encoding="utf-8")
    kernel_gate.write_text('{"passed": true}', encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(ANALYZER_PATH),
            "--run-dir",
            str(tmp_path),
            "--preflight",
            str(preflight),
            "--kernel-gate",
            str(kernel_gate),
            "--output",
            str(output),
        ],
    )
    analyzer.main()
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["passed"] is True
    assert result["acceptance"]["hierarchical_lower_95_gt_1_10"] is True
    assert result["protocol"]["all_blocks_share_exact_software_and_planner_abi"]
