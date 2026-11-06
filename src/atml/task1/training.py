"""One-epoch SFT with token-normalized accumulation and complete recovery state."""

from __future__ import annotations

import math
import time

import torch
from datasets import load_from_disk
from torch.nn import functional as F

from atml.common.io import (
    append_jsonl,
    environment,
    read_json,
    read_jsonl,
    require_cuda,
    restore_rng,
    rng_state,
    seed_all,
    sha256,
    write_json,
)
from atml.task1.adapters import adapters, export_factors
from atml.task1.config import METHODS, RECIPE, protocol
from atml.task1.data import collate, verify_preparation
from atml.task1.model import checkpoint_inventory, starting_model, tokenizer_for, training_model
from atml.task1.loss import LOSS_CHUNK_TOKENS, response_loss
from atml.task1.storage import archive_checkpoint, ensure_checkpoint, latest_checkpoint, private_api


def summed_response_loss(logits, labels):
    return F.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.shape[-1]),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )


def optimizer_for(model):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():  # tied parameters deduplicated by PyTorch
        if not parameter.requires_grad:
            continue
        (no_decay if name.endswith("bias") or parameter.ndim < 2 else decay).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": RECIPE.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=1.0,
        betas=RECIPE.adam_betas,
        eps=RECIPE.adam_epsilon,
        foreach=False,
    )


def learning_rate(update, peak):
    """1-based update number: 38 linear warmup updates, then cosine to zero."""
    if update <= RECIPE.warmup:
        return peak * update / RECIPE.warmup
    progress = (update - RECIPE.warmup) / (RECIPE.updates - RECIPE.warmup)
    return peak * 0.5 * (1.0 + math.cos(math.pi * progress))


class StartingPoint:
    """CPU FP32 reference, transferred one tensor at a time to bound GPU overhead."""

    def __init__(self, model):
        self.reference = {
            name: p.detach().float().cpu().clone() for name, p in model.named_parameters()
        }
        self.storage_bytes = sum(p.numel() * p.element_size() for p in self.reference.values())

    @torch.no_grad()
    def penalty_and_gradient(self, model, coefficient=0.0, gradient_scale=1.0):
        omega = torch.zeros((), device=next(model.parameters()).device, dtype=torch.float32)
        self.last_transfer_seconds = 0.0
        self.last_transfer_bytes = 0
        for name, parameter in model.named_parameters():
            tick = time.perf_counter()
            reference = self.reference[name].to(parameter.device)
            self.last_transfer_seconds += time.perf_counter() - tick
            if parameter.device != self.reference[name].device:
                self.last_transfer_bytes += reference.numel() * reference.element_size()
            difference = parameter.detach().float() - reference
            omega.add_(difference.square().sum() * 0.5)
            if coefficient:
                if parameter.grad is None:
                    parameter.grad = torch.zeros_like(parameter)
                parameter.grad.add_(difference, alpha=coefficient * gradient_scale)
        return omega.item()


def _cpu_state(state):
    # Preserve exact aliases (e.g. tied embedding/LM-head weights) in the export.
    # Cloning both names separately wastes host RAM and checkpoint disk space.
    unique, result = {}, {}
    for name, value in state.items():
        key = (
            value.untyped_storage().data_ptr(),
            value.storage_offset(),
            tuple(value.shape),
            tuple(value.stride()),
            value.dtype,
            value.device,
        )
        if key not in unique:
            unique[key] = value.detach().cpu().clone()
        result[name] = unique[key]
    return result


def save_checkpoint(paths, method, model, optimizer, scaler, update, stats):
    destination = paths.checkpoint(method.name, update)
    temporary = destination.with_name(destination.name + ".incomplete")
    if destination.exists() or temporary.exists():
        raise FileExistsError(
            f"Checkpoint already exists: {destination}; preserve or move it before retrying."
        )
    temporary.mkdir(parents=True)
    tick = time.perf_counter()
    if method.rank:
        state = {
            f"{name}.{factor}": getattr(layer, factor)
            for name, layer in adapters(model).items()
            for factor in ("A", "B")
        }
        torch.save(_cpu_state(state), temporary / "adapter.pt")
        torch.save(export_factors(model), temporary / "projected_factors.pt")
    else:
        torch.save(_cpu_state(model.state_dict()), temporary / "model.pt")
    export_seconds = time.perf_counter() - tick
    # Optimizer tensors and RNG are saved at a completed effective-batch boundary.
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "rng": rng_state(),
            "update": update,
            "examples_seen": update * 32,
            "scheduler": {
                "last_update": update,
                "next_lr": learning_rate(update + 1, method.learning_rate)
                if update < RECIPE.updates
                else 0.0,
            },
            "protocol": protocol(),
            "method": method.name,
            "asset_lock": read_json(paths.evidence / "asset_lock.json"),
            "preparation_sha256": sha256(paths.evidence / "preparation.json"),
            "stats": stats,
        },
        temporary / "recovery.pt",
    )
    write_json(
        temporary / "metadata.json",
        {
            "method": method.name,
            "update": update,
            "examples_seen": update * 32,
            "export_seconds": export_seconds,
            "checkpoint_seconds": time.perf_counter() - tick,
            "resume_stats": {
                **stats,
                "export_seconds": stats.get("export_seconds", 0) + export_seconds,
                "checkpoint_seconds": stats.get("checkpoint_seconds", 0)
                + time.perf_counter()
                - tick,
            },
        },
    )
    temporary.rename(destination)
    inventory = checkpoint_inventory(destination)
    append_jsonl(
        paths.evidence / "runs" / method.name / "checkpoints.jsonl",
        {**read_json(destination / "metadata.json"), **inventory},
    )
    return export_seconds


def _restore(paths, method, model, optimizer, scaler, checkpoint):
    # Only load recovery files created by this project: they contain Python RNG state.
    recovery = torch.load(checkpoint / "recovery.pt", map_location="cpu", weights_only=False)
    recovery["stats"] = read_json(checkpoint / "metadata.json")["resume_stats"]
    if recovery["protocol"] != protocol() or recovery["method"] != method.name:
        raise ValueError("Resume protocol or method mismatch.")
    if recovery["asset_lock"] != read_json(paths.evidence / "asset_lock.json"):
        raise ValueError("Resume starting assets changed.")
    if recovery["preparation_sha256"] != sha256(paths.evidence / "preparation.json"):
        raise ValueError("Resume data preparation changed.")
    state = torch.load(
        checkpoint / ("adapter.pt" if method.rank else "model.pt"),
        map_location="cpu",
        weights_only=True,
    )
    if method.rank:
        with torch.no_grad():
            for name, layer in adapters(model).items():
                for factor in ("A", "B"):
                    getattr(layer, factor).copy_(state[f"{name}.{factor}"])
    else:
        model.load_state_dict(state)
    optimizer.load_state_dict(recovery["optimizer"])
    # AdamW's exp_avg/exp_avg_sq follow FP32 master/adaptor parameters.
    for parameter, state in optimizer.state.items():
        for key in ("exp_avg", "exp_avg_sq"):
            if key in state:
                state[key] = state[key].to(parameter.device, dtype=torch.float32)
    scaler.load_state_dict(recovery["scaler"])
    restore_rng(recovery["rng"])
    return recovery


def train(
    paths, method_name, resume=None, stop_after=None, checkpoint_repo=None, resume_latest=False
):
    device = require_cuda()
    verify_preparation(paths)
    method = METHODS[method_name]
    if checkpoint_repo:
        shared_path = paths.evidence / "storage" / "shared.json"
        if not shared_path.exists():
            raise FileNotFoundError("Run backup --repo before training with checkpoint offloading.")
        shared = read_json(shared_path)
        if shared.get("repo") != checkpoint_repo or not shared.get("verified"):
            raise ValueError(
                "Checkpoint repository must match the verified shared-artifact backup."
            )
        private_api(checkpoint_repo)  # Fail before training if credentials/access have expired.
    if resume_latest:
        latest = latest_checkpoint(paths, method_name)
        if latest == RECIPE.updates:
            print(
                f"[{method_name}] Final checkpoint already exists locally or in verified backup.",
                flush=True,
            )
            return
        resume = ensure_checkpoint(paths, method_name, latest) if latest else None
    seed_all(RECIPE.seed)
    directory = paths.evidence / "runs" / method_name
    directory.mkdir(parents=True, exist_ok=True)
    log = directory / "training.jsonl"
    if log.exists() and resume is None:
        raise FileExistsError(
            "Training log exists. Resume a checkpoint or use a new evidence directory."
        )
    setup_start = time.perf_counter()
    print(
        f"[{method_name}] Loading independent starting model and preparing optimizer...", flush=True
    )
    model = training_model(paths, method, device)
    reference_start = time.perf_counter()
    reference = StartingPoint(model) if method.l2_lambda else None
    reference_seconds = time.perf_counter() - reference_start
    optimizer = optimizer_for(model)
    scaler = torch.amp.GradScaler("cuda")
    dataset = load_from_disk(str(paths.prepared))
    tokenizer = tokenizer_for(paths)
    stats = {
        "train_seconds": 0.0,
        "checkpoint_seconds": 0.0,
        "regularization_seconds": 0.0,
        "reference_transfer_seconds": 0.0,
        "reference_transfer_bytes": 0,
        "export_seconds": 0.0,
        "skipped_updates": 0,
        "successful_updates": 0,
        "peak_gpu_bytes": 0,
    }
    start_update = 0
    if resume:
        recovery = _restore(paths, method, model, optimizer, scaler, resume)
        start_update, stats = recovery["update"], recovery["stats"]
        if recovery["examples_seen"] != start_update * 32:
            raise ValueError("Resume data position is not a complete optimizer update.")
        if log.exists():
            # Preserve a crash tail as evidence, then discard it from the active curve.
            rows = read_jsonl(log)
            tail = [row for row in rows if row["update"] > start_update]
            if tail:
                write_json(directory / f"discarded_after_update_{start_update}.json", tail)
            log.write_text("", encoding="utf-8")
            for row in rows:
                if row["update"] <= start_update:
                    append_jsonl(log, row)
    if stop_after == "next":
        stop_after = next(u for u in RECIPE.recovery_updates if u > start_update)
    stop_after = int(stop_after) if stop_after is not None else RECIPE.updates
    if stop_after not in RECIPE.recovery_updates or stop_after <= start_update:
        raise ValueError("Stop only at a required recovery checkpoint after the resume position.")
    suffix = f"_resume_{start_update}" if resume else ""
    write_json(directory / f"environment{suffix}.json", environment())
    write_json(
        directory / f"configuration{suffix}.json",
        {
            "protocol": protocol(),
            "method": method_name,
            "resume_from": str(resume) if resume else None,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "reference_storage_bytes": reference.storage_bytes if reference else 0,
            "reference_creation_seconds": reference_seconds,
            "setup_seconds": time.perf_counter() - setup_start,
            "loss_chunk_tokens": LOSS_CHUNK_TOKENS,
            "checkpoint_repo": checkpoint_repo,
        },
    )
    if (
        checkpoint_repo
        and resume
        and resume.resolve() == paths.checkpoint(method_name, start_update).resolve()
    ):
        # Optimizer/model/RNG are restored in memory. Free the verified previous
        # checkpoint before creating the next one in Kaggle's saved-output disk.
        archive_checkpoint(paths, method_name, start_update, checkpoint_repo, offload=True)
    torch.cuda.reset_peak_memory_stats()
    print(
        f"[{method_name}] Training segment: updates {start_update + 1}-{stop_after} of {RECIPE.updates}, "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters, "
        f"microbatch {RECIPE.microbatch} x accumulation {RECIPE.accumulation}",
        flush=True,
    )
    for update in range(start_update + 1, stop_after + 1):
        torch.cuda.synchronize()
        tick = time.perf_counter()
        first = (update - 1) * 32
        print(
            f"[{method_name}] Update {update}/{RECIPE.updates} starting "
            f"({first:,}/{RECIPE.examples:,} examples processed)",
            flush=True,
        )
        rows = [dataset[i] for i in range(first, first + 32)]
        valid_tokens = sum(sum(x != -100 for x in row["labels"][1:]) for row in rows)
        optimizer.zero_grad(set_to_none=True)
        lr = learning_rate(update, method.learning_rate)
        for group in optimizer.param_groups:
            group["lr"] = lr
        loss_sum = 0.0
        for offset in range(0, 32, RECIPE.microbatch):
            batch = collate(
                rows[offset : offset + RECIPE.microbatch], tokenizer.pad_token_id, device
            )
            with torch.autocast("cuda", dtype=torch.float16):
                loss = response_loss(model, batch)
            loss_sum += loss.detach().item()
            scaler.scale(loss / valid_tokens).backward()
            del loss, batch
            microbatch = offset // RECIPE.microbatch + 1
            if microbatch % 4 == 0:
                print(
                    f"[{method_name}]   microbatch {microbatch}/{RECIPE.accumulation}", flush=True
                )
        omega, regularization_seconds = 0.0, 0.0
        if reference:
            torch.cuda.synchronize()
            penalty_start = time.perf_counter()
            omega = reference.penalty_and_gradient(model, method.l2_lambda, scaler.get_scale())
            torch.cuda.synchronize()
            regularization_seconds = time.perf_counter() - penalty_start
            stats["reference_transfer_seconds"] += reference.last_transfer_seconds
            stats["reference_transfer_bytes"] += reference.last_transfer_bytes
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], RECIPE.clip_norm
        )
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        skipped = scaler.get_scale() < old_scale
        stats["skipped_updates"] += int(skipped)
        stats["successful_updates"] += int(not skipped)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - tick
        stats["train_seconds"] += seconds
        stats["regularization_seconds"] += regularization_seconds
        stats["peak_gpu_bytes"] = max(stats["peak_gpu_bytes"], torch.cuda.max_memory_allocated())
        append_jsonl(
            log,
            {
                "update": update,
                "examples_seen": update * 32,
                "response_tokens": valid_tokens,
                "sft_loss": loss_sum / valid_tokens,
                "lr": lr,
                "omega": omega,
                "lambda_omega": method.l2_lambda * omega,
                "total_loss": loss_sum / valid_tokens + method.l2_lambda * omega,
                "gradient_norm": norm.item() if torch.isfinite(norm) else None,
                "skipped": skipped,
                "gradient_scale": scaler.get_scale(),
                "seconds": seconds,
                "regularization_seconds": regularization_seconds,
                "reference_transfer_seconds": reference.last_transfer_seconds if reference else 0.0,
                "reference_transfer_bytes": reference.last_transfer_bytes if reference else 0,
            },
        )
        if update in RECIPE.recovery_updates:
            print(f"[{method_name}] Saving recovery checkpoint at update {update}...", flush=True)
            if not method.rank and (
                update == RECIPE.updates or (method_name == "full_ft" and update == 625)
            ):
                if reference:
                    displacement = math.sqrt(2 * reference.penalty_and_gradient(model))
                else:
                    original = starting_model(paths, torch.float32, "cpu")
                    temporary_reference = StartingPoint(original)
                    del original
                    displacement = math.sqrt(2 * temporary_reference.penalty_and_gradient(model))
                    del temporary_reference
                append_jsonl(
                    directory / "displacement.jsonl", {"update": update, "l2": displacement}
                )
            checkpoint_start = time.perf_counter()
            stats["export_seconds"] += save_checkpoint(
                paths, method, model, optimizer, scaler, update, stats
            )
            stats["checkpoint_seconds"] += time.perf_counter() - checkpoint_start
            write_json(directory / "resources.json", stats)
            write_json(
                directory / "status.json",
                {
                    "last_completed_update": update,
                    "examples_seen": update * 32,
                    "complete": update == RECIPE.updates,
                },
            )
            print(
                f"[{method_name}] Checkpoint saved: {paths.checkpoint(method_name, update)}",
                flush=True,
            )
            if checkpoint_repo:
                archive_checkpoint(paths, method_name, update, checkpoint_repo, offload=True)
        average_seconds = stats["train_seconds"] / update
        remaining_seconds = average_seconds * (RECIPE.updates - update)
        print(
            f"[{method_name}] {update}/{RECIPE.updates} ({100 * update / RECIPE.updates:.1f}%) | "
            f"examples {update * 32:,}/{RECIPE.examples:,} | SFT {loss_sum / valid_tokens:.5f} | "
            f"LR {lr:.3e} | skipped {stats['skipped_updates']} | "
            f"elapsed {stats['train_seconds'] / 60:.1f} min | ETA {remaining_seconds / 60:.1f} min",
            flush=True,
        )
    write_json(directory / "resources.json", stats)
    if stop_after < RECIPE.updates:
        print(
            f"[{method_name}] Segment finished at update {stop_after}; resume for the remaining examples.",
            flush=True,
        )
