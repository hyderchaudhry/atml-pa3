"""Fast Kaggle environment checks; never starts training or downloads model weights."""

from __future__ import annotations

import importlib.metadata
import os
import shutil
import sys
import tempfile

import torch
from huggingface_hub import HfApi
from packaging.requirements import Requirement

from atml.common.io import tree_bytes, write_json
from atml.task1.config import RECIPE

EXPECTED = [
    "torch==2.6.0",
    "transformers==4.51.3",
    "tokenizers==0.21.4",
    "datasets==3.6.0",
    "huggingface-hub==0.30.2",
    "lm-eval==0.4.9.1",
    "peft==0.15.2",
    "accelerate==1.6.0",
    "numpy==2.2.5",
    "matplotlib==3.10.1",
    "PyYAML==6.0.2",
]


def doctor(paths):
    checks = []

    def check(name, passed, detail):
        checks.append({"check": name, "passed": bool(passed), "detail": detail})
        print(f"[{'OK' if passed else 'FAIL'}] {name}: {detail}", flush=True)

    check("Python", (3, 10) <= sys.version_info[:2] < (3, 13), sys.version.split()[0])
    for specification in EXPECTED:
        requirement = Requirement(specification)
        try:
            installed = importlib.metadata.version(requirement.name)
            check(requirement.name, installed in requirement.specifier, installed)
        except importlib.metadata.PackageNotFoundError:
            check(requirement.name, False, "not installed")
    for path in (paths.artifacts, paths.evidence):
        try:
            path.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryFile(dir=path) as stream:
                stream.write(b"write check")
            free = shutil.disk_usage(path).free
            check(
                f"Writable output: {path}",
                free >= 12 * 1024**3,
                f"{free / 1024**3:.1f} GiB free on filesystem; one live Full FT checkpoint needs several GiB",
            )
        except OSError as error:
            check(f"Writable output: {path}", False, str(error))
    saved_bytes = tree_bytes(paths.artifacts) + tree_bytes(paths.evidence)
    print(
        f"[info] Task output uses {saved_bytes / 1e9:.2f} GB; offload checkpoints between stages.",
        flush=True,
    )
    check("CUDA", torch.cuda.is_available(), f"runtime={torch.version.cuda}")
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        print(
            f"[info] Using GPU 0: {properties.name}, {properties.total_memory / 1024**3:.1f} GiB; "
            f"visible GPUs={torch.cuda.device_count()} (memory is not pooled)",
            flush=True,
        )
        try:
            layer = torch.nn.Linear(8, 8, device="cuda", dtype=torch.float32)
            optimizer = torch.optim.SGD(layer.parameters(), lr=0.0)
            scaler = torch.amp.GradScaler("cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                loss = layer(torch.ones(2, 8, device="cuda")).square().mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            check(
                "FP16 AMP and scaler",
                all(torch.isfinite(p.grad).all() for p in layer.parameters()),
                "toy arithmetic only; no pretrained weights or experiment steps",
            )
            del layer, optimizer, scaler, loss
            torch.cuda.empty_cache()
        except RuntimeError as error:
            check("FP16 AMP and scaler", False, str(error))
    try:
        info = HfApi().model_info(RECIPE.model, timeout=10)
        check("Hub access", bool(info.sha), "model metadata reachable; no weights downloaded")
    except Exception as error:
        # Print the exception class, not a request that could contain credentials.
        check("Hub access", False, f"{type(error).__name__}; enable Internet in Kaggle")
    print(
        f"[info] Isolated environment: {sys.prefix != sys.base_prefix}; "
        f"checkpoint repository configured: {bool(os.environ.get('PA3_BACKUP_REPO'))}",
        flush=True,
    )
    report = {
        "passed": all(row["passed"] for row in checks),
        "checks": checks,
        "scope": "Environment checks only; peak model memory and runtime must be measured on your GPU.",
    }
    write_json(paths.evidence / "system" / "doctor.json", report)
    return report["passed"]


def memory_check(paths, method_name):
    """One synthetic worst-length step; discard all weights and optimizer state."""
    from atml.common.io import require_cuda, seed_all
    from atml.task1.config import METHODS
    from atml.task1.loss import response_loss
    from atml.task1.model import training_model
    from atml.task1.training import StartingPoint, optimizer_for

    device = require_cuda()
    seed_all(RECIPE.seed)
    model, optimizer, reference, loss = None, None, None, None
    report = {
        "method": method_name,
        "microbatch": RECIPE.microbatch,
        "sequence_length": RECIPE.max_length,
        "scope": "Synthetic memory check; no data, models, or scores saved",
    }
    try:
        model = training_model(paths, METHODS[method_name], device)
        reference = StartingPoint(model) if METHODS[method_name].l2_lambda else None
        optimizer = optimizer_for(model)
        # Allocate FP32 Adam moments before forward: later training updates retain
        # these states even while activations/logits/gradients are live.
        for parameter in model.parameters():
            if parameter.requires_grad:
                optimizer.state[parameter] = {
                    "step": torch.zeros((), dtype=torch.float32),
                    "exp_avg": torch.zeros_like(parameter),
                    "exp_avg_sq": torch.zeros_like(parameter),
                }
        tokens = torch.arange(RECIPE.max_length, device=device).repeat(RECIPE.microbatch, 1)
        tokens = tokens % model.config.vocab_size
        labels = tokens.clone()
        labels[:, 0] = -100
        batch = {"input_ids": tokens, "attention_mask": torch.ones_like(tokens), "labels": labels}
        scaler = torch.amp.GradScaler("cuda")
        torch.cuda.reset_peak_memory_stats()
        with torch.autocast("cuda", dtype=torch.float16):
            loss = response_loss(model, batch) / (labels[:, 1:] != -100).sum()
        scaler.scale(loss).backward()
        if reference:
            reference.penalty_and_gradient(
                model, METHODS[method_name].l2_lambda, scaler.get_scale()
            )
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], RECIPE.clip_norm
        )
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        report.update(
            passed=True,
            peak_gpu_bytes=torch.cuda.max_memory_allocated(),
            gpu_bytes=torch.cuda.get_device_properties(0).total_memory,
        )
    except torch.OutOfMemoryError:
        report.update(
            passed=False,
            error="CUDA out of memory; no batch, precision, or length settings were reduced",
        )
    finally:
        del model, optimizer, reference, loss
        torch.cuda.empty_cache()
    write_json(paths.evidence / "system" / f"memory_{method_name}.json", report)
    print(report, flush=True)
    return report["passed"]
