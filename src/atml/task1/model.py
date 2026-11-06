"""Model construction, starting-weight bases, and independent inference exports."""

from __future__ import annotations

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from atml.common.io import read_json, require_cuda, sha256, tree_bytes, write_json
from atml.task1.adapters import (
    AdapterLinear,
    compute_bases,
    install_adapters,
    load_basis_cache,
    merge_factors,
    target_modules,
)
from atml.task1.config import CONDITIONS, METHODS, RECIPE
from atml.task1.storage import ensure_checkpoint


def tokenizer_for(paths):
    return AutoTokenizer.from_pretrained(paths.artifacts / "tokenizer", use_fast=True)


def starting_model(paths, dtype=torch.float32, device="cpu"):
    revision = read_json(paths.evidence / "asset_lock.json")["model"]
    model = AutoModelForCausalLM.from_pretrained(
        RECIPE.model,
        revision=revision,
        torch_dtype=dtype,
        attn_implementation="eager",
    )
    return model.to(device)


def build_bases(paths):
    device = require_cuda()
    destination = paths.artifacts / "bases.pt"
    if destination.exists():
        load_basis_cache(paths)
        return
    started = time.perf_counter()
    model = starting_model(paths, torch.float32, device)
    bases = {}
    timings = {}
    targets = target_modules(model)
    for index, (name, layer) in enumerate(targets, start=1):
        torch.cuda.synchronize()
        tick = time.perf_counter()
        bases[name] = compute_bases(layer.weight, RECIPE.protection_rank)
        torch.cuda.synchronize()
        timings[name] = time.perf_counter() - tick
        print(f"[bases] {index}/{len(targets)} {name}: {timings[name]:.2f} seconds", flush=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bases, destination)
    write_json(
        paths.evidence / "bases.json",
        {
            "model_revision": read_json(paths.evidence / "asset_lock.json")["model"],
            "rank": RECIPE.protection_rank,
            "dtype": "float32",
            "algorithm": "exact reduced SVD",
            "preprocessing_seconds": time.perf_counter() - started,
            "per_module_seconds": timings,
            "tensor_bytes": sum(
                t.numel() * t.element_size() for pair in bases.values() for t in pair
            ),
            "file_bytes": destination.stat().st_size,
            "sha256": sha256(destination),
        },
    )


def training_model(paths, method, device):
    dtype = torch.float16 if method.rank else torch.float32
    model = starting_model(paths, dtype, device)
    if method.rank:
        bases = load_basis_cache(paths) if method.orthogonal else None
        install_adapters(model, method, bases)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    return model


def load_condition(paths, condition, device, dtype=torch.float16):
    method_name, update = CONDITIONS[condition]
    model = starting_model(paths, dtype, device)
    if method_name is not None:
        checkpoint = ensure_checkpoint(
            paths,
            method_name,
            update,
            ["projected_factors.pt" if METHODS[method_name].rank else "model.pt"],
        )
        if METHODS[method_name].rank:
            factors = torch.load(
                checkpoint / "projected_factors.pt", map_location="cpu", weights_only=True
            )
            merge_factors(model, factors)
        else:
            state = torch.load(checkpoint / "model.pt", map_location="cpu", weights_only=True)
            model.load_state_dict(state)
    model.eval()
    model.config.use_cache = True
    return model


@torch.no_grad()
def verify_merge(paths, method_name, update, device):
    """Verify exported factors against a separately loaded, freshly merged backbone."""
    checkpoint = ensure_checkpoint(paths, method_name, update, ["projected_factors.pt"])
    factors = torch.load(checkpoint / "projected_factors.pt", map_location="cpu", weights_only=True)
    model = starting_model(paths, torch.float16, device).eval()
    # Exported OPLoRA factors need no projection or basis at inference.
    method = METHODS[method_name]
    for name, values in factors.items():
        parent, _, child = name.rpartition(".")
        wrapper = AdapterLinear(model.get_submodule(name), method.rank, method.alpha, RECIPE.seed)
        wrapper.A.copy_(values["A"].to(device))
        wrapper.B.copy_(values["B"].to(device))
        setattr(model.get_submodule(parent), child, wrapper.eval())
    tokenizer = tokenizer_for(paths)
    from atml.task1.data import chat_prompt

    tokenizer.padding_side = "left"
    batch = tokenizer(
        [
            chat_prompt(tokenizer, "What is 2 + 3?"),
            chat_prompt(tokenizer, "Describe a quiet morning."),
        ],
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    ).to(device)
    with torch.autocast(device.type, dtype=torch.float16):
        unmerged = model(**batch).logits.float().cpu()
    del model
    torch.cuda.empty_cache()
    fresh = starting_model(paths, torch.float16, device).eval()
    merge_factors(fresh, factors)
    merged = fresh(**batch).logits.float().cpu()
    difference = (unmerged - merged).abs()
    # FP16 weight-rounding and different GEMM associations prevent exact equality.
    atol, rtol = 0.15, 0.02
    valid = batch.attention_mask.cpu().bool().unsqueeze(-1).expand_as(merged)
    close = torch.isclose(unmerged, merged, atol=atol, rtol=rtol)[valid]
    result = {
        "max_abs_error": difference[valid].max().item(),
        "mean_abs_error": difference[valid].mean().item(),
        "atol": atol,
        "rtol": rtol,
        "passed": bool(close.all()),
        "input_ids": batch.input_ids.cpu().tolist(),
        "explanation": "Fresh FP16 merge compared with exported FP32 factors under FP16 AMP",
    }
    write_json(paths.evidence / "evaluations" / f"{method_name}_final" / "merge_check.json", result)
    if not result["passed"]:
        raise RuntimeError("Merged/unmerged logits failed tolerance; inspect merge_check.json.")
    del fresh
    torch.cuda.empty_cache()
    return result


def checkpoint_inventory(path):
    return {
        "total_bytes": tree_bytes(path),
        "files": {
            str(p.relative_to(path)): p.stat().st_size for p in path.rglob("*") if p.is_file()
        },
    }
