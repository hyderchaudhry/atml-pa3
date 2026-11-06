"""FP32 weight geometry and local responses on fixed baseline activations."""

from __future__ import annotations

import torch

from atml.common.io import read_json, require_cuda, seed_all, sha256, write_json
from atml.task1.adapters import load_basis_cache
from atml.task1.config import CONDITIONS, METHODS, RECIPE
from atml.task1.model import starting_model, tokenizer_for
from atml.task1.storage import ensure_checkpoint
from atml.task1.runtime import Runtime


def projection_name(layer):
    return f"model.layers.{layer}.mlp.down_proj"


@torch.no_grad()
def cache_probes(paths):
    device = require_cuda()
    seed_all(RECIPE.seed)
    destination = paths.artifacts / "baseline_activations.pt"
    if destination.exists() and (paths.evidence / "activation_cache.json").exists():
        metadata = read_json(paths.evidence / "activation_cache.json")
        if metadata["sha256"] != sha256(destination):
            raise ValueError("Activation cache changed.")
        if metadata["probe_manifest_sha256"] != sha256(
            paths.evidence / "probe_ids_and_prompts.json"
        ):
            raise ValueError("Activation probe manifest changed.")
        return
    probes = read_json(paths.evidence / "probe_ids_and_prompts.json")
    model = starting_model(paths, torch.float32, device).eval()
    tokenizer = tokenizer_for(paths)
    tokenizer.padding_side = "left"
    runtime = Runtime.current()
    cached = {str(layer): [] for layer in RECIPE.diagnostic_layers}
    handles = []
    for layer in RECIPE.diagnostic_layers:

        def hook(module, args, output, layer=layer):
            # Left padding puts every last non-padding prompt token at index -1.
            cached[str(layer)].extend(args[0][:, -1].detach().float().cpu().clone().unbind())

        handles.append(model.get_submodule(projection_name(layer)).register_forward_hook(hook))
    try:
        for offset in range(0, len(probes), runtime.probe_batch):
            rows = probes[offset : offset + runtime.probe_batch]
            inputs = tokenizer(
                [r["prompt"] for r in rows],
                padding=True,
                return_tensors="pt",
                add_special_tokens=False,
            ).to(device)
            position_ids = inputs.attention_mask.cumsum(-1) - 1
            position_ids.masked_fill_(inputs.attention_mask == 0, 0)
            # Only hidden states/hooks are needed: avoid the full vocabulary head.
            model.model(**inputs, position_ids=position_ids, use_cache=False)
            print(
                f"[probes] cached {offset + len(rows)}/{len(probes)} starting-model prompts",
                flush=True,
            )
    finally:
        for handle in handles:
            handle.remove()
    tensors = {key: torch.stack(value) for key, value in cached.items()}
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensors, destination)
    write_json(
        paths.evidence / "activation_cache.json",
        {
            "sha256": sha256(destination),
            "probe_manifest_sha256": sha256(paths.evidence / "probe_ids_and_prompts.json"),
            "dtype": "float32",
            "runtime": runtime.as_dict(),
            "position": "last non-padding prompt token; before continuation",
            "shapes": {key: list(value.shape) for key, value in tensors.items()},
        },
    )


def weight_metrics(original, delta, U):
    magnitude_squared = delta.square().sum()
    return {
        "rho": ((U.T @ delta).square().sum() / magnitude_squared).item()
        if magnitude_squared.item()
        else None,
        "eta": (delta.norm() / original.norm()).item(),
        "zero_update": magnitude_squared.item() == 0,
    }


def activation_ratios(original, delta, activations):
    return (activations @ delta.T).norm(dim=-1) / ((activations @ original.T).norm(dim=-1) + 1e-8)


@torch.no_grad()
def diagnose(paths, condition):
    if condition == "baseline":
        raise ValueError("Weight-update diagnostics apply to six trained conditions only.")
    metadata = read_json(paths.evidence / "activation_cache.json")
    if metadata["sha256"] != sha256(paths.artifacts / "baseline_activations.pt"):
        raise ValueError("Activation cache changed.")
    if metadata["probe_manifest_sha256"] != sha256(paths.evidence / "probe_ids_and_prompts.json"):
        raise ValueError("Probe IDs or prompts changed.")
    # FP32 CPU algebra is deliberate: diagnostics need no full GPU model.
    original_model = starting_model(paths, torch.float32, "cpu").eval()
    bases = load_basis_cache(paths)
    activations = torch.load(
        paths.artifacts / "baseline_activations.pt", map_location="cpu", weights_only=True
    )
    probes = read_json(paths.evidence / "probe_ids_and_prompts.json")
    method_name, update = CONDITIONS[condition]
    is_adapter = bool(METHODS[method_name].rank)
    checkpoint = ensure_checkpoint(
        paths, method_name, update, ["projected_factors.pt" if is_adapter else "model.pt"]
    )
    state = torch.load(
        checkpoint / ("projected_factors.pt" if is_adapter else "model.pt"),
        map_location="cpu",
        weights_only=True,
    )
    rows, per_probe = [], []
    for layer in RECIPE.diagnostic_layers:
        name = projection_name(layer)
        original = original_model.get_submodule(name).weight.detach().float()
        if is_adapter:
            factors = state[name]
            delta = factors["scale"] * (factors["B"].float() @ factors["A"].float())
        else:
            delta = state[name + ".weight"].float() - original
        U, V = bases[name]
        metrics = weight_metrics(original, delta, U)
        ratios = activation_ratios(original, delta, activations[str(layer)].float())
        for probe, value in zip(probes, ratios.tolist()):
            per_probe.append(
                {"layer": layer, "task": probe["task"], "doc_id": probe["doc_id"], "R": value}
            )
        summaries = {}
        for task in ("gsm8k_cot", "hellaswag"):
            values = torch.tensor(
                [r["R"] for r in per_probe if r["layer"] == layer and r["task"] == task]
            )
            summaries[task] = {
                "mean": values.mean().item(),
                "median": values.median().item(),
                "std": values.std().item(),
                "count": len(values),
            }
        rows.append(
            {
                "layer": layer,
                **metrics,
                "activation": summaries,
                "left_constraint_relative": (
                    (U.T @ delta).norm() / delta.norm().clamp_min(1e-30)
                ).item(),
                "right_constraint_relative": (
                    (delta @ V).norm() / delta.norm().clamp_min(1e-30)
                ).item(),
            }
        )
    write_json(
        paths.evidence / "diagnostics" / f"{condition}.json",
        {
            "condition": condition,
            "dtype": "float32",
            "protection_rank": RECIPE.protection_rank,
            "layers": rows,
            "per_probe": per_probe,
            "scope": "Same cached starting-model activations; measures local updates only.",
        },
    )
