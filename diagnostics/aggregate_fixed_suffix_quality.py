#!/usr/bin/env python3
"""Reduce the six fixed S4/A128/T768 WikiText-2 TEST confirmation windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aggregate_heldout_quality as base


STARTS = (0, 23_600, 47_200, 70_800, 94_400, 118_000)
CONTEXT = 20_480
DECODE_STEPS = 1_536
EXACT_TAIL = 768
EXACT_PREFIX_PAGES = 4
EXACT_STATIC_SUFFIX_PAGES = 128
BOOTSTRAP_SAMPLES = 50_000
PAGE_GAUGE_BOOTSTRAP_SEED = 20_260_875
FLASHINFER_BOOTSTRAP_SEED = 20_260_876


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generic_cluster_bootstrap(
    clusters: list[dict[str, Any]], seed: int
) -> dict[str, Any]:
    require(len(clusters) == len(STARTS), "bootstrap requires exactly six clusters")
    generator = random.Random(seed)
    draws = {
        name: []
        for name in (
            "ppl_ratio",
            "ppl_difference",
            "mean_forward_kl",
            "mean_js",
            "mean_tv",
            "mean_reference_rank",
            "mean_candidate_rank",
            "top1",
        )
    }
    for _ in range(BOOTSTRAP_SAMPLES):
        selected = [clusters[generator.randrange(len(clusters))] for _ in clusters]
        count = sum(int(cluster["token_count"]) for cluster in selected)
        reference_sum = math.fsum(
            float(cluster["reference_nll_sum"]) for cluster in selected
        )
        candidate_sum = math.fsum(
            float(cluster["candidate_nll_sum"]) for cluster in selected
        )
        reference_ppl = math.exp(reference_sum / count)
        candidate_ppl = math.exp(candidate_sum / count)
        draws["ppl_ratio"].append(math.exp((candidate_sum - reference_sum) / count))
        draws["ppl_difference"].append(candidate_ppl - reference_ppl)
        draws["mean_forward_kl"].append(
            math.fsum(float(cluster["forward_kl_sum"]) for cluster in selected)
            / count
        )
        draws["mean_js"].append(
            math.fsum(float(cluster["js_sum"]) for cluster in selected) / count
        )
        draws["mean_tv"].append(
            math.fsum(float(cluster["tv_sum"]) for cluster in selected) / count
        )
        draws["mean_reference_rank"].append(
            math.fsum(float(cluster["reference_rank_sum"]) for cluster in selected)
            / count
        )
        draws["mean_candidate_rank"].append(
            math.fsum(float(cluster["candidate_rank_sum"]) for cluster in selected)
            / count
        )
        draws["top1"].append(
            sum(int(cluster["top1_count"]) for cluster in selected) / count
        )
    percentile = base.percentile
    return {
        "method": "nonparametric percentile bootstrap over six whole corpus windows",
        "cluster_count_per_draw": len(clusters),
        "samples": BOOTSTRAP_SAMPLES,
        "seed": seed,
        "one_sided_95": {
            "ppl_ratio_upper": percentile(draws["ppl_ratio"], 0.95),
            "ppl_difference_upper": percentile(draws["ppl_difference"], 0.95),
            "mean_forward_kl_upper": percentile(draws["mean_forward_kl"], 0.95),
            "mean_js_upper": percentile(draws["mean_js"], 0.95),
            "mean_tv_upper": percentile(draws["mean_tv"], 0.95),
            "mean_reference_rank_upper": percentile(
                draws["mean_reference_rank"], 0.95
            ),
            "mean_candidate_rank_upper": percentile(
                draws["mean_candidate_rank"], 0.95
            ),
            "top1_lower": percentile(draws["top1"], 0.05),
        },
        "two_sided_90": {
            "ppl_ratio": [
                percentile(draws["ppl_ratio"], 0.05),
                percentile(draws["ppl_ratio"], 0.95),
            ]
        },
    }


def validate_record(path: Path, expected_start: int) -> tuple[dict[str, Any], dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    label = str(path)
    require(payload.get("batch_size") == 1, f"{label}: batch size")
    require(payload.get("context") == CONTEXT, f"{label}: context")
    require(payload.get("decode_steps") == DECODE_STEPS, f"{label}: decode steps")
    require(payload.get("exact_tail_tokens") == EXACT_TAIL, f"{label}: exact tail")
    require(
        payload.get("exact_prefix_pages") == EXACT_PREFIX_PAGES,
        f"{label}: exact prefix",
    )
    require(
        payload.get("exact_static_suffix_pages") == EXACT_STATIC_SUFFIX_PAGES,
        f"{label}: fixed exact suffix",
    )
    require(payload.get("tail_attention") == "flashinfer_merge", f"{label}: tail")
    require(payload.get("baseline_split_pages") == 256, f"{label}: FI split")
    require(payload.get("candidate_split_pages") == 256, f"{label}: PG split")
    token_source = payload.get("token_source", {})
    require(token_source.get("split") == "test", f"{label}: split")
    require(
        token_source.get("archive_member") == "wikitext-2-raw/wiki.test.raw",
        f"{label}: member",
    )
    require(
        token_source.get("corpus_window_start_offsets") == [expected_start],
        f"{label}: start",
    )
    require(token_source.get("corpus_windows_disjoint") is True, f"{label}: windows")
    correctness = payload.get("correctness", {})
    require(correctness.get("passed") is True, f"{label}: correctness")
    require(
        correctness.get("exact_prefix_canary", {}).get("passed") is True,
        f"{label}: fixed exact canary",
    )
    finalization = correctness.get("runtime_page_finalization", {})
    require(finalization.get("passed") is True, f"{label}: finalization")
    require(
        finalization.get("runtime_generated_int8_logical_page_count") == 48,
        f"{label}: recurrence count",
    )
    require(
        finalization.get("endpoint_partition_coverage_disjoint") is True,
        f"{label}: endpoint partition",
    )
    for source_path, expected in payload.get("source_sha256", {}).items():
        source = path.parents[3] / source_path
        require(source.is_file(), f"{label}: missing source {source_path}")
        require(sha256_file(source) == expected, f"{label}: source drift {source_path}")
    windows = correctness.get("quality_metric_protocol", {}).get("window_metadata")
    require(isinstance(windows, list) and len(windows) == 1, f"{label}: metadata")
    record = {"path": str(path), "sha256": sha256_file(path), "payload": payload}
    return record, {0: windows[0]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs=6, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    keyed: dict[int, Path] = {}
    for path in args.inputs:
        payload = json.loads(path.read_text(encoding="utf-8"))
        starts = payload.get("token_source", {}).get("corpus_window_start_offsets", [])
        require(len(starts) == 1, f"{path}: one start required")
        start = int(starts[0])
        require(start not in keyed, f"duplicate start {start}")
        keyed[start] = path
    require(tuple(sorted(keyed)) == STARTS, "inputs do not match the six fixed windows")

    pg_clusters: list[dict[str, Any]] = []
    hf_clusters: list[dict[str, Any]] = []
    pg_cosines: list[float] = []
    hf_cosines: list[float] = []
    input_manifest = []
    token_hashes = set()
    cluster_ids = set()
    for start in STARTS:
        record, windows = validate_record(keyed[start], start)
        current_pg, pg_cosine, pg_labels = base.collect_comparison_clusters(
            record, windows, "page_gauge_vs_flashinfer_fp16"
        )
        current_hf, hf_cosine, hf_labels = base.collect_comparison_clusters(
            record, windows, "flashinfer_fp16_vs_hf_sdpa_fp16"
        )
        require(pg_labels == hf_labels, f"{keyed[start]}: label mismatch")
        cluster_id = current_pg[0]["cluster_unit_id"]
        require(cluster_id not in cluster_ids, f"duplicate cluster {cluster_id}")
        cluster_ids.add(cluster_id)
        token_hash = record["payload"]["token_source"]["token_ids_sha256"]
        require(token_hash not in token_hashes, f"duplicate token content {token_hash}")
        token_hashes.add(token_hash)
        pg_clusters.extend(current_pg)
        hf_clusters.extend(current_hf)
        pg_cosines.append(pg_cosine)
        hf_cosines.append(hf_cosine)
        input_manifest.append(
            {
                "path": str(keyed[start]),
                "sha256": record["sha256"],
                "corpus_window_start_offset": start,
                "token_ids_sha256": token_hash,
            }
        )

    pg_point = base.point_summary(pg_clusters, min(pg_cosines))
    hf_point = base.point_summary(hf_clusters, min(hf_cosines))
    pg_bootstrap = generic_cluster_bootstrap(pg_clusters, PAGE_GAUGE_BOOTSTRAP_SEED)
    hf_bootstrap = generic_cluster_bootstrap(hf_clusters, FLASHINFER_BOOTSTRAP_SEED)
    pg_gate = base.gate_page_gauge(pg_point, pg_bootstrap)
    hf_gate = base.gate_flashinfer_hf(hf_point, hf_bootstrap)
    result = {
        "schema_version": 1,
        "experiment": "page_gauge_fixed_suffix_s4_a128_t768_test_confirmation",
        "passed": bool(pg_gate["passed"] and hf_gate["passed"]),
        "protocol": {
            "context": CONTEXT,
            "decode_steps": DECODE_STEPS,
            "exact_prefix_pages": EXACT_PREFIX_PAGES,
            "exact_static_suffix_pages": EXACT_STATIC_SUFFIX_PAGES,
            "exact_tail_tokens": EXACT_TAIL,
            "runtime_generated_int8_pages": 48,
            "window_start_offsets": list(STARTS),
            "cluster_count": len(STARTS),
            "total_labeled_tokens": len(STARTS) * DECODE_STEPS,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "adaptive_policy_disclosure": (
                "S4/A128/T768 was selected after diagnosis of an earlier TEST failure; "
                "this is a fixed-policy confirmation cohort, not untouched held-out selection."
            ),
        },
        "inputs": input_manifest,
        "page_gauge_vs_flashinfer_fp16": {
            "point": pg_point,
            "cluster_bootstrap": pg_bootstrap,
            "gate": pg_gate,
        },
        "flashinfer_fp16_vs_hf_sdpa_fp16": {
            "point": hf_point,
            "cluster_bootstrap": hf_bootstrap,
            "gate": hf_gate,
        },
        "thresholds": {
            "page_gauge_vs_flashinfer_fp16": base.PAGE_GAUGE_GATES,
            "flashinfer_fp16_vs_hf_sdpa_fp16": base.FLASHINFER_HF_GATES,
        },
        "invocation": {"argv": sys.argv, "utc_timestamp": datetime.now(timezone.utc).isoformat()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({"passed": result["passed"], "output": str(args.output), "page_gauge": pg_gate, "flashinfer_hf": hf_gate}, indent=2))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
