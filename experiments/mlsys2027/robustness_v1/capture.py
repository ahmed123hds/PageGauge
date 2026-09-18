"""Future GPU capture worker. Importing this file does not import accelerator libraries."""
from pathlib import Path
import json
import sys


def run(manifest, directory):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.cache_utils import DynamicCache
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    import zipfile

    cfg = manifest["plan"]
    torch.manual_seed(cfg["seed"])
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    model_path = manifest["model_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    # The launch manifest already exists before the TRAIN member is opened.
    with zipfile.ZipFile(manifest["archive"]) as archive:
        text = archive.read(cfg["wikitext_member"]).decode("utf-8")
    ids = tokenizer.encode(text, add_special_tokens=False)
    begin = cfg["token_offset"]
    length = cfg["initial_context_tokens"] + cfg["decode_steps"]
    if len(ids) < begin + length:
        raise RuntimeError("Declared TRAIN window is not available; no wrap or fallback permitted")
    tokens = torch.tensor(ids[begin:begin + length], dtype=torch.long, device="cuda")[None]
    del ids, text
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float16,
                                               local_files_only=True, attn_implementation="sdpa").eval().cuda()
    if (model.config.num_hidden_layers, model.config.num_attention_heads, model.config.num_key_value_heads,
            model.config.hidden_size) != (32, 32, 8, 4096) or model.config.sliding_window is not None:
        raise RuntimeError("This capture protocol requires Mistral 32/32/8/4096 without sliding window")
    original = ALL_ATTENTION_FUNCTIONS["sdpa"]
    records = []
    wanted = {(layer, cfg["initial_context_tokens"] + generated)
              for layer in cfg["layers"] for generated in cfg["generated_token_snapshots"]}

    def capture_sdpa(module, query, key, value, attention_mask, **kwargs):
        output = original(module, query, key, value, attention_mask, **kwargs)
        identity = (module.layer_idx, key.shape[-2])
        if query.shape[-2] == 1 and identity in wanted:
            if attention_mask is not None and not bool((attention_mask == 0).all()):
                raise RuntimeError("Capture has a nontrivial attention mask; unrestricted replay would be invalid")
            if kwargs.get("dropout", 0) != 0 or abs(kwargs["scaling"] - 128**-0.5) > 1e-12:
                raise RuntimeError("Unexpected attention semantics")
            layer, n = identity
            # Convert layout to tokens, KV-head, channel for independent replay.
            k = key[0].transpose(0, 1).contiguous()
            v = value[0].transpose(0, 1).contiguous()
            initial = cfg["initial_context_tokens"]
            ck = k[:initial].float().mean(dim=0).half()
            cv = v[:initial].float().mean(dim=0).half()
            name = f"layer_{layer:02d}_D{n-initial:04d}.npz"
            np.savez(directory / name, query=query[0, :, 0].cpu().numpy(), key=k.cpu().numpy(), value=v.cpu().numpy(),
                     key_center=ck.cpu().numpy(), value_center=cv.cpu().numpy(),
                     hf_attention_output=output[0][0, 0].cpu().numpy())
            records.append({"layer": layer, "generated_tokens": n-initial, "tokens": n, "path": name})
            wanted.remove(identity)
            print(f"Captured {name}", flush=True)
        return output

    ALL_ATTENTION_FUNCTIONS.register("sdpa", capture_sdpa)
    try:
        cache = DynamicCache()
        initial = cfg["initial_context_tokens"]
        spans = [(i, min(i + cfg["prefill_chunk_tokens"], initial)) for i in range(0, initial, cfg["prefill_chunk_tokens"])]
        spans.extend((i, i + 1) for i in range(initial, length))
        for start, end in spans:
            positions = torch.arange(start, end, device="cuda", dtype=torch.long)
            result = model.model(input_ids=tokens[:, start:end], position_ids=positions[None], cache_position=positions,
                                 past_key_values=cache, use_cache=True, return_dict=True)
            cache = result.past_key_values
            del result
        if wanted:
            raise RuntimeError(f"Missing declared captures: {sorted(wanted)}")
        torch.cuda.synchronize()
        return {"records": records, "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
                "capture_count": len(records), "trajectory": cfg["trajectory"],
                "scope": "HF trajectory component fixtures, not production PageGauge trajectories or speed evidence"}
    finally:
        ALL_ATTENTION_FUNCTIONS.register("sdpa", original)
