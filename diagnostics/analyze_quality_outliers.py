#!/usr/bin/env python3
"""Render PageGauge's posthoc strict-quality outlier diagnostics as Markdown.

This analyzer performs no GPU work and does not recompute or relax any gate. It
only summarizes the ``outlier_diagnostics`` payload emitted by
``benchmark_sustained_dynamic_graphs.py --quality-diagnostics-top-k K``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rows", type=int, default=24)
    return parser.parse_args()


def percent(value: float) -> str:
    return f"{100.0 * float(value):.4f}%"


def markdown_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    output = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        output.append("| " + " | ".join(str(value) for value in row) + " |")
    return output


def load_payload(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    result = json.loads(path.read_text(encoding="utf-8"))
    try:
        diagnostics = result["correctness"]["backend_vs_hf_sdpa_fp16"][
            "outlier_diagnostics"
        ]
    except (KeyError, TypeError) as error:
        raise SystemExit(
            "input lacks outlier diagnostics; rerun the worker with "
            "--quality-diagnostics-top-k K"
        ) from error
    if not isinstance(diagnostics, dict):
        raise SystemExit("outlier diagnostics are disabled or malformed")
    return result, diagnostics


def render(result: dict[str, Any], diagnostic: dict[str, Any], rows: int) -> str:
    configuration = result["configuration"]
    endpoint = result["correctness"]["backend_vs_hf_sdpa_fp16"]
    summary = diagnostic["summary"]
    quantiles = summary["cosine_quantiles"]

    lines = [
        "# PageGauge strict-quality outlier localization",
        "",
        "This is a posthoc diagnostic. It does **not** change the strict gate, "
        "the model computation, or the timing path.",
        "",
        "## Run",
        "",
        f"- Backend: `{result['backend']}`",
        f"- Trajectory: `{configuration['trajectory_mode']}`",
        f"- Batch/context/decode: B={configuration['batch_size']}, "
        f"C={configuration['context']}, D={configuration['decode_steps']}",
        f"- Exact tail: {configuration['exact_tail_tokens']} tokens",
        f"- Strict cosine threshold: {endpoint['minimum_logits_cosine']}",
        f"- Worker pass: `{result['passed']}`",
        "",
        "## Global localization",
        "",
        f"- Checked rows: {summary['checked_rows']}",
        f"- Rows below strict gate: {summary['rows_below_gate']} "
        f"({percent(summary['fraction_below_gate'])})",
        f"- Top-1 mismatches: {summary['top1_mismatches']}",
        f"- Top-1 agreement: {percent(summary['top1_agreement_fraction'])}",
        f"- Cosine minimum / p1 / p5 / median: "
        f"{quantiles['minimum']:.9f} / {quantiles['p1']:.9f} / "
        f"{quantiles['p5']:.9f} / {quantiles['median']:.9f}",
        "",
        "## Per request",
        "",
    ]
    lines.extend(
        markdown_table(
            [
                "Request",
                "Min cosine",
                "Min step",
                "Absolute position",
                "Below gate",
                "Top-1 mismatches",
            ],
            [
                [
                    row["request"],
                    f"{row['minimum_cosine']:.9f}",
                    row["minimum_step"],
                    row["minimum_absolute_position"],
                    row["rows_below_gate"],
                    row["top1_mismatches"],
                ]
                for row in diagnostic["per_request"]
            ],
        )
    )

    failing_offsets = [
        row
        for row in diagnostic["per_page_offset"]
        if row["rows_below_gate"] or row["top1_mismatches"]
    ]
    lines.extend(["", "## Page offsets containing failures", ""])
    if failing_offsets:
        lines.extend(
            markdown_table(
                ["Offset", "Rows", "Min cosine", "Below gate", "Top-1 mismatch"],
                [
                    [
                        row["page_offset"],
                        row["rows"],
                        f"{row['minimum_cosine']:.9f}",
                        row["rows_below_gate"],
                        row["top1_mismatches"],
                    ]
                    for row in failing_offsets
                ],
            )
        )
    else:
        lines.append("No page offset contains a below-gate row.")

    lines.extend(["", "## Consecutive failure clusters", ""])
    clusters = summary["failure_clusters"]
    if clusters:
        lines.extend(
            markdown_table(
                ["Request", "Start", "End", "Length"],
                [
                    [
                        cluster["request"],
                        cluster["start_step"],
                        cluster["end_step_inclusive"],
                        cluster["length"],
                    ]
                    for cluster in clusters
                ],
            )
        )
    else:
        lines.append("No below-threshold clusters were found.")

    retained = diagnostic["worst_rows"][: max(0, rows)]
    lines.extend(["", f"## Worst {len(retained)} request-step rows", ""])
    lines.extend(
        markdown_table(
            [
                "Rank",
                "Step",
                "Req",
                "Pos",
                "Off",
                "Runtime INT8 pages",
                "Cosine",
                "Rel-L2",
                "Max abs",
                "Top-1",
                "Ref margin",
                "Cand margin",
            ],
            [
                [
                    row["rank"],
                    row["step"],
                    row["request"],
                    row["absolute_position"],
                    row["page_offset"],
                    row["generated_int8_pages_visible"],
                    f"{row['cosine']:.9f}",
                    f"{row['relative_l2']:.6f}",
                    f"{row['maximum_absolute_error']:.6f}",
                    (
                        f"{row['reference_top1_id']}={row['candidate_top1_id']}"
                        if row["top1_match"]
                        else f"{row['reference_top1_id']}≠{row['candidate_top1_id']}"
                    ),
                    f"{row['reference_top1_margin']:.5f}",
                    f"{row['candidate_top1_margin']:.5f}",
                ]
                for row in retained
            ],
        )
    )

    lines.extend(
        [
            "",
            "## How to interpret the next decision",
            "",
            "- Failures concentrated at one page offset suggest append/finalization "
            "or boundary-sensitive attention behavior.",
            "- Failures beginning only after `generated_int8_pages_visible > 0` "
            "implicate recurrently finalized pages rather than the initial prefix.",
            "- A single request with isolated steps suggests content-dependent "
            "quantization sensitivity; broad request/page clusters suggest a "
            "systematic representation issue.",
            "- Very small reference norms or tiny top-1 margins can make cosine or "
            "argmax unusually sensitive, but they do not justify weakening the gate.",
            "- This report localizes the rows; it does not by itself attribute the "
            "error to K, V, a layer, or a quantization parameter.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.rows <= 0:
        raise SystemExit("--rows must be positive")
    result, diagnostics = load_payload(args.input)
    text = render(result, diagnostics, args.rows)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(args.output)
    else:
        print(text)


if __name__ == "__main__":
    main()
