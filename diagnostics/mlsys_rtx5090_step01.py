#!/usr/bin/env python3
"""Synthetic correctness of the frozen production S4/A128/T768 attention path.

GPU libraries are imported only inside run_gpu(). Use the guarded entry script.
This checks static cache snapshots and graph replay, not append recurrence,
model fidelity, a fused reconstruction kernel, or end-to-end performance.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "pagegauge_mlsys2027_rtx5090_production_attention_correctness_v1"
CONFIG = {
    "batch_size": 4,
    "initial_context_tokens": 20480,
    "page_size": 16,
    "query_heads": 32,
    "kv_heads": 8,
    "head_dim": 128,
    "exact_prefix_pages": 4,
    "exact_static_suffix_pages": 128,
    "exact_tail_tokens": 768,
    "candidate_split_pages": 128,
    "reference_split_pages": 256,
    "old_value_scale_placement": "probability",
    "center_restore": "attention_add",
    "seed": 2026090801,
    "generated_token_snapshots": [0, 1, 15, 16, 768, 784, 1536],
    "input_distribution": "FP16 Gaussian residual std=0.08; request/channel offset std=0.025; query std=0.35",
    "max_abs": 0.0002,
    "min_cosine": 0.999999,
    "max_lse_abs": 0.0002,
    "graph_replays_per_snapshot": 2,
}


def expected_partition(generated_tokens: int) -> dict:
    """Independent set definition used to check the production page planner."""
    total = CONFIG["initial_context_tokens"] + generated_tokens
    page = CONFIG["page_size"]
    total_pages = math.ceil(total / page)
    initial_pages = CONFIG["initial_context_tokens"] // page
    suffix_begin = initial_pages - CONFIG["exact_static_suffix_pages"]
    exact = set(range(CONFIG["exact_prefix_pages"]))
    exact.update(range(suffix_begin, initial_pages))
    exact.update(range(total_pages - CONFIG["exact_tail_tokens"] // page, total_pages))
    old = sorted(set(range(total_pages)) - exact)
    return {
        "old": old,
        "exact": sorted(exact),
        "total_pages": total_pages,
        "last_page_len": (total - 1) % page + 1,
        "generated_old_pages_in_snapshot": sum(item >= initial_pages for item in old),
    }


def run_gpu(manifest: dict) -> dict:
    import torch
    import torch.nn.functional as functional
    import flashinfer

    sys.path.insert(0, str(ROOT / "scripts"))
    import benchmark_page_gauge_transformer as production

    if tuple(torch.cuda.get_device_capability()) != (12, 0):
        raise RuntimeError("Step 01 requires RTX 5090 compute capability 12.0.")
    if "RTX 5090" not in torch.cuda.get_device_name():
        raise RuntimeError("Step 01 requires RTX 5090 hardware.")
    if flashinfer.__version__ != "0.6.17":
        raise RuntimeError("FlashInfer version drift.")
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = CONFIG
    batch, page, hkv, hq, dim = (cfg[name] for name in ("batch_size", "page_size", "kv_heads", "query_heads", "head_dim"))
    initial_pages = cfg["initial_context_tokens"] // page
    capacity = (cfg["initial_context_tokens"] + max(cfg["generated_token_snapshots"])) // page
    tail_pages = cfg["exact_tail_tokens"] // page
    storage_pages = cfg["exact_prefix_pages"] + cfg["exact_static_suffix_pages"] + tail_pages
    generator = torch.Generator(device="cuda").manual_seed(cfg["seed"])
    shape = (batch, capacity, page, hkv, dim)

    def create_channel_cache():
        offset = torch.randn((batch, 1, 1, hkv, dim), device="cuda", dtype=torch.float16, generator=generator) * 0.025
        original = (torch.randn(shape, device="cuda", dtype=torch.float16, generator=generator) * 0.08 + offset).contiguous()
        # Centers are fixed using the initial prompt only, separately per request.
        center = original[:, :initial_pages].float().mean(dim=(1, 2)).half()
        centered_float = original.float() - center.float()[:, None, None]
        scales = (centered_float.abs().amax(dim=(2, 4)) / 127.0).clamp_min(2.0**-20).half().contiguous()
        codes = (centered_float / scales.float()[:, :, None, :, None]).round().clamp(-127, 127).to(torch.int8).contiguous()
        reconstructed = (codes.float() * scales.float()[:, :, None, :, None]).half().contiguous()
        return center, centered_float.half().contiguous(), codes, scales, reconstructed

    k_center, centered_k, codes_k, scales_k, reconstructed_k = create_channel_cache()
    v_center, centered_v, codes_v, scales_v, reconstructed_v = create_channel_cache()
    query = (torch.randn((batch, hq, dim), device="cuda", dtype=torch.float16, generator=generator) * 0.35).contiguous()
    exact_k = torch.full((batch * storage_pages, page, hkv, dim), 17.0, device="cuda", dtype=torch.float16)
    exact_v = torch.full_like(exact_k, -19.0)
    cache = production.GaugeCache(
        exact_k.unsqueeze(0), exact_v.unsqueeze(0),
        codes_k.flatten(0, 1).unsqueeze(0), codes_v.flatten(0, 1).unsqueeze(0),
        scales_k.flatten(0, 1).unsqueeze(0), scales_v.flatten(0, 1).unsqueeze(0),
        k_center.unsqueeze(0), v_center.unsqueeze(0),
        exact_tail_pages=tail_pages, exact_sink_pages=cfg["exact_prefix_pages"],
        exact_static_suffix_pages=cfg["exact_static_suffix_pages"], initial_context_pages=initial_pages,
    )
    # Call the actual production eager_attention/graph_attention methods with
    # a one-layer synthetic cache. No model weights or decoder initialization.
    decoder = object.__new__(production.TransformerDecoder)
    decoder.backend = "page_gauge"
    decoder.tail_attention = "flashinfer_merge"
    decoder.center_restore = cfg["center_restore"]
    decoder.cache = cache
    decoder.rotated_query = query
    decoder.attention_output = torch.empty((1, batch, hq, dim), device="cuda", dtype=torch.float16)
    decoder.exact_output = torch.empty_like(decoder.attention_output)
    decoder.old_lse = torch.empty((1, batch, hq), device="cuda", dtype=torch.float32)
    decoder.exact_lse = torch.empty_like(decoder.old_lse)
    decoder.old_wrapper = production.GraphDecodeWrapper(flashinfer, capacity, torch.int8, True, batch_size=batch)
    decoder.exact_wrapper = production.GraphDecodeWrapper(flashinfer, storage_pages, torch.float16, False, batch_size=batch)
    reference_wrapper = production.GraphDecodeWrapper(flashinfer, capacity, torch.float16, False, batch_size=batch)
    all_pages = production.request_major_page_table(batch, capacity)
    ring_pages = production.request_major_exact_page_table(
        batch, capacity, tail_pages, cfg["exact_prefix_pages"],
        cfg["exact_static_suffix_pages"], initial_pages,
    )
    old_table = torch.empty((batch, capacity), device="cuda", dtype=torch.int32)
    exact_table = torch.empty((batch, storage_pages), device="cuda", dtype=torch.int32)
    ref_out = torch.empty_like(query)
    ref_lse = torch.empty((batch, hq), device="cuda", dtype=torch.float32)

    def metrics(actual, expected):
        if not bool(torch.isfinite(actual).all() & torch.isfinite(expected).all()):
            return {"finite": False, "max_abs": None, "cosine_min": None, "passed": False}
        left, right = actual.double(), expected.double()
        maximum = float((left - right).abs().max())
        cosine = float(functional.cosine_similarity(left, right, dim=-1).min())
        return {"finite": True, "max_abs": maximum, "cosine_min": cosine,
                "passed": maximum <= cfg["max_abs"] and cosine >= cfg["min_cosine"]}

    def dense_reference(logical_k, logical_v, length):
        # FP64 grouped-query oracle over uncentered reconstructed K/V checks
        # common-key-center cancellation and one-time value restoration.
        outputs = []
        for request in range(batch):
            key = logical_k[request].flatten(0, 1)[:length].double() + k_center[request].double()
            value = logical_v[request].flatten(0, 1)[:length].double() + v_center[request].double()
            q = query[request].double().reshape(hkv, hq // hkv, dim)
            score = torch.einsum("hgd,thd->hgt", q, key) / math.sqrt(dim)
            probabilities = score.softmax(dim=-1)
            outputs.append(torch.einsum("hgt,thd->hgd", probabilities, value).reshape(hq, dim))
        return torch.stack(outputs)

    records = []
    for generated in cfg["generated_token_snapshots"]:
        expected = expected_partition(generated)
        total_tokens = cfg["initial_context_tokens"] + generated
        partition = production.page_gauge_logical_partition(
            expected["total_pages"], tail_pages, cfg["exact_prefix_pages"],
            expected["last_page_len"], cfg["exact_static_suffix_pages"], initial_pages,
        )
        if (list(partition["old_logical_pages"]) != expected["old"]
                or list(partition["exact_logical_pages"]) != expected["exact"]):
            raise RuntimeError("Production partition differs from independent set definition.")
        old_indices = production.write_page_gauge_old_page_table(old_table, all_pages, partition["old_logical_pages"])
        exact_indices = production.write_exact_prefix_tail_page_table(
            exact_table, ring_pages, partition["tail_logical_begin"], expected["total_pages"],
            cfg["exact_prefix_pages"], cfg["exact_static_suffix_pages"], initial_pages,
        )
        # Non-contiguous exact pages use the production fixed-slot/ring map.
        logical_indices = torch.tensor(expected["exact"], device="cuda", dtype=torch.long)
        physical_indices = ring_pages.index_select(1, logical_indices).long().flatten()
        if physical_indices.unique().numel() != physical_indices.numel():
            raise RuntimeError("Active exact pages alias the same physical slot.")
        exact_k.fill_(17.0)
        exact_v.fill_(-19.0)
        exact_k.index_copy_(0, physical_indices, centered_k.index_select(1, logical_indices).flatten(0, 1))
        exact_v.index_copy_(0, physical_indices, centered_v.index_select(1, logical_indices).flatten(0, 1))
        logical_k = reconstructed_k.clone()
        logical_v = reconstructed_v.clone()
        logical_k[:, logical_indices] = centered_k[:, logical_indices]
        logical_v[:, logical_indices] = centered_v[:, logical_indices]
        decoder.old_wrapper.plan(old_indices, partition["old_token_count"], page, cfg["candidate_split_pages"], page_table_epoch=generated)
        decoder.exact_wrapper.plan(exact_indices, partition["exact_token_count"], expected["last_page_len"], cfg["candidate_split_pages"], page_table_epoch=generated)
        reference_wrapper.plan(all_pages[:, :expected["total_pages"]], total_tokens, expected["last_page_len"], cfg["reference_split_pages"], page_table_epoch=generated)
        production.TransformerDecoder.eager_attention(decoder, 0)
        candidate = decoder.attention_output[0].clone()
        candidate_lse = decoder.old_lse[0].clone()
        reference_wrapper.wrapper.run(query, (logical_k.flatten(0, 1), logical_v.flatten(0, 1)),
                                      out=ref_out, lse=ref_lse, return_lse=True)
        ref_out.add_(cache.output_center[0])
        reference = dense_reference(logical_k, logical_v, total_tokens)
        against_fp16 = metrics(candidate, ref_out)
        against_dense = metrics(candidate, reference)
        reference_against_dense = metrics(ref_out, reference)
        lse_finite = bool(torch.isfinite(candidate_lse).all() & torch.isfinite(ref_lse).all())
        lse_max = float((candidate_lse - ref_lse).abs().max()) if lse_finite else None
        # Reuse the production attention body in a graph; no timing claim here.
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                production.TransformerDecoder.graph_attention(decoder, 0)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            production.TransformerDecoder.graph_attention(decoder, 0)
        graph_equal = True
        for _ in range(cfg["graph_replays_per_snapshot"]):
            graph.replay()
            torch.cuda.synchronize()
            graph_equal = graph_equal and bool(torch.equal(decoder.attention_output[0], candidate))
        passed = (against_fp16["passed"] and against_dense["passed"] and reference_against_dense["passed"]
                  and lse_finite and lse_max <= cfg["max_lse_abs"] and graph_equal)
        row = {
            "generated_tokens_in_snapshot": generated,
            "logical_tokens": total_tokens,
            "partition": expected,
            "candidate_vs_reconstructed_fp16": against_fp16,
            "candidate_vs_dense_fp64": against_dense,
            "reconstructed_fp16_vs_dense_fp64": reference_against_dense,
            "centered_lse_max_abs": lse_max,
            "graph_replay_bitwise_equal": graph_equal,
            "passed": passed,
        }
        records.append(row)
        print(f"snapshot D={generated:4d} old={len(expected['old'])} exact={len(expected['exact'])} "
              f"error={against_fp16['max_abs']} cosine={against_fp16['cosine_min']} "
              f"graph={graph_equal} {'PASS' if passed else 'FAIL'}", flush=True)
        del graph, stream, logical_k, logical_v, reference, candidate, candidate_lse
    torch.cuda.synchronize()
    return {
        "schema_version": 1, "experiment": EXPERIMENT,
        "manifest_sha256": manifest["manifest_sha256"],
        "passed": all(row["passed"] for row in records),
        "config": cfg, "records": records,
        "environment": {"gpu": torch.cuda.get_device_name(), "compute_capability": list(torch.cuda.get_device_capability()),
                        "torch": str(torch.__version__), "flashinfer": flashinfer.__version__},
        "production_module": decoder.old_wrapper.wrapper.page_gauge_module_uri,
        "production_source_hashes": decoder.old_wrapper.wrapper.page_gauge_source_hashes,
        "scope": "Synthetic static snapshots through production attention and graph replay. No runtime append/finalize recurrence, model quality, fused reconstruction, or end-to-end speed claim.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from mlsys_rtx5090_entry import atomic_json, canonical_hash, sha256_file

    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output}")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if canonical_hash(body) != manifest.get("manifest_sha256") or manifest.get("config") != CONFIG:
        raise SystemExit("Step 01 manifest integrity/configuration mismatch.")
    for name, digest in manifest["source_sha256"].items():
        source = (ROOT / name).resolve()
        if not source.is_relative_to(ROOT.resolve()) or sha256_file(source) != digest:
            raise SystemExit(f"Source changed after manifest freeze: {name}")
    try:
        result = run_gpu(manifest)
    except Exception as error:
        result = {"schema_version": 1, "experiment": EXPERIMENT, "passed": False,
                  "manifest_sha256": manifest["manifest_sha256"], "error": str(error),
                  "traceback": traceback.format_exc()}
        traceback.print_exc()
    atomic_json(args.output, result)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
