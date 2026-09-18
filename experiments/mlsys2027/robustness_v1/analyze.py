"""Independent CPU replay; no CUDA imports, no timing or kernel-correctness claims."""
import json
from pathlib import Path
import numpy as np
from affine_reference import attention, attention_error_bound, mixed_reconstructions, output_metrics


def analyze_capture(path, plan, replay_path=None):
    with np.load(path, allow_pickle=False) as data:
        q, k, v, ck, cv, hf = (data[key] for key in ("query", "key", "value", "key_center", "value_center", "hf_attention_output"))
    if q.shape != (32, 128) or k.shape[1:] != (8, 128) or v.shape != k.shape:
        raise ValueError("Capture geometry mismatch")
    kr = mixed_reconstructions(k, ck, plan["initial_context_tokens"], plan["policy"])
    vr = mixed_reconstructions(v, cv, plan["initial_context_tokens"], plan["policy"])
    replay = None
    if replay_path is not None:
        with np.load(replay_path, allow_pickle=False) as data:
            replay = {key: data[key] for key in ("factorized", "explicit")}
    heads = []
    for head in range(8):
        query = q[head*4:head*4+4]
        key, value = k[:, head], v[:, head]
        outputs = {"original": attention(query, key, value)}
        for name in ("centered", "real", "half"):
            outputs[name] = attention(query, kr[name][:, head], vr[name][:, head])
        bounds = attention_error_bound(query, key, value, kr["real"][:, head], vr["real"][:, head])
        heads.append({"kv_head": head,
            "centered_storage_vs_original": output_metrics(outputs["centered"], outputs["original"]),
            "mixed_reconstruction_vs_original": output_metrics(outputs["real"], outputs["original"]),
            "quantized_pages_vs_centered_storage": output_metrics(outputs["real"], outputs["centered"]),
            "reconstruction_rounding_only": output_metrics(outputs["half"], outputs["real"]),
            "hf_sdpa_vs_original_fp64": output_metrics(hf[head*4:head*4+4], outputs["original"]),
            "fixed_query_bound": bounds})
        if replay is not None:
            factorized = replay["factorized"][head*4:head*4+4]
            explicit = replay["explicit"][head*4:head*4+4]
            heads[-1]["factorized_vs_explicit_fp16"] = output_metrics(factorized, explicit)
            heads[-1]["factorized_vs_reconstructed_real_fp64"] = output_metrics(factorized, outputs["real"])
            heads[-1]["explicit_fp16_vs_rounded_reconstruction_fp64"] = output_metrics(explicit, outputs["half"])
            heads[-1]["factorized_vs_original_fp64"] = output_metrics(factorized, outputs["original"])
    return {"capture": Path(path).name, "tokens": len(k), "heads": heads,
            "key_scale_range": [kr["scale_min"], kr["scale_max"]],
            "value_scale_range": [vr["scale_min"], vr["scale_max"]],
            "key_clipped_values": kr["clipped_values"], "value_clipped_values": vr["clipped_values"],
            "exact_tokens": int(kr["exact_mask"].sum()), "quantized_tokens": int((~kr["exact_mask"]).sum())}
