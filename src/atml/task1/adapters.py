"""Original factor-based LoRA and OPLoRA; no PEFT implementation is used."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from atml.common.io import sha256
from atml.task1.config import RECIPE, TARGETS


class AdapterLinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: int, seed: int, bases=None):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.scale = alpha / rank
        # A private CPU generator makes LoRA-16/OPLoRA-16 exactly match, independent
        # of SVD, backbone loading, device, and dropout RNG consumption.
        generator = torch.Generator(device="cpu").manual_seed(seed)
        bound = 1 / math.sqrt(base.in_features)
        initial = torch.empty(rank, base.in_features, dtype=torch.float32)
        initial.uniform_(-bound, bound, generator=generator)
        self.A = nn.Parameter(initial.to(base.weight.device))
        self.B = nn.Parameter(
            torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32)
        )
        self.dropout = nn.Dropout(RECIPE.dropout)
        self.register_buffer("U", bases[0].to(base.weight.device) if bases else None)
        self.register_buffer("V", bases[1].to(base.weight.device) if bases else None)

    def effective_factors(self):
        # Projection is FP32 even under FP16 autocast. No dense projectors are built.
        with torch.autocast(self.A.device.type, enabled=False):
            if self.U is None:
                return self.A, self.B
            return self.A - (self.A @ self.V) @ self.V.T, self.B - self.U @ (self.U.T @ self.B)

    def forward(self, inputs):
        A, B = self.effective_factors()
        correction = F.linear(F.linear(self.dropout(inputs), A), B) * self.scale
        return self.base(inputs) + correction


def target_modules(model):
    return [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in TARGETS
    ]


def install_adapters(model, method, bases=None):
    model.requires_grad_(False)
    targets = target_modules(model)
    expected = model.config.num_hidden_layers * len(TARGETS)
    if len(targets) != expected:
        raise ValueError(f"Expected {expected} target projections, found {len(targets)}.")
    for index, (name, base) in enumerate(targets):
        parent, _, child = name.rpartition(".")
        layer = AdapterLinear(
            base,
            method.rank,
            method.alpha,
            RECIPE.seed + index,
            bases[name] if method.orthogonal else None,
        )
        setattr(model.get_submodule(parent), child, layer)
    return model


def adapters(model):
    return {
        name: module for name, module in model.named_modules() if isinstance(module, AdapterLinear)
    }


@torch.no_grad()
def export_factors(model):
    exported = {}
    for name, layer in adapters(model).items():
        A, B = layer.effective_factors()
        exported[name] = {
            "A": A.detach().cpu().clone(),
            "B": B.detach().cpu().clone(),
            "scale": layer.scale,
        }
    return exported


@torch.no_grad()
def merge_factors(fresh_model, factors):
    for name, values in factors.items():
        weight = fresh_model.get_submodule(name).weight
        merged = weight.float() + values["scale"] * (
            values["B"].to(weight.device) @ values["A"].to(weight.device)
        )
        weight.copy_(merged.to(weight.dtype))
    return fresh_model


@torch.no_grad()
def compute_bases(weight, k):
    if k > min(weight.shape):
        raise ValueError(f"Protection rank {k} exceeds weight shape {tuple(weight.shape)}.")
    # Exact reduced SVD, computed once on original FP32 weights.
    with torch.autocast(weight.device.type, enabled=False):
        U, _, Vh = torch.linalg.svd(weight.float(), full_matrices=False)
    return U[:, :k].contiguous().cpu(), Vh[:k].T.contiguous().cpu()


def load_basis_cache(paths):
    from atml.common.io import read_json

    manifest = read_json(paths.evidence / "bases.json")
    path = paths.artifacts / "bases.pt"
    if sha256(path) != manifest["sha256"]:
        raise ValueError("Starting-weight basis cache was modified.")
    return torch.load(path, map_location="cpu", weights_only=True)
