"""Applied-update curriculum, exact sampler/RNG recovery, and atomic checkpoints."""

import gc
import math
import shutil
import time
from collections import Counter

import torch

from atml.common.io import (
    environment,
    read_json,
    require_cuda,
    restore_rng,
    rng_state,
    seed_all,
    sha256,
    write_json,
    write_jsonl,
)

from .config import SEED, protocol, stage_at
from .data import Sampler, assert_isolation
from .features import FeatureCache, save_tensor
from .model import load_model


def latest(paths):
    checkpoints = sorted((paths.artifacts / "checkpoints").glob("update_*/complete.json"))
    return int(checkpoints[-1].parent.name.split("_")[-1]) if checkpoints else None


def scope(paths):
    return dict(
        protocol=protocol(),
        selection_sha256=sha256(paths.manifest),
        assets=read_json(paths.evidence / "asset_lock.json"),
        features_sha256=sha256(paths.artifacts / "features" / "index.json"),
    )


def save_checkpoint(paths, update, state):
    if state["update"] != update:
        raise ValueError("Checkpoint path must match its applied update")
    destination = paths.checkpoint(update)
    temporary = destination.with_name("." + destination.name + ".incomplete")
    if destination.exists():
        raise FileExistsError(f"Never overwrite a committed checkpoint: {destination}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    save_tensor(temporary / "recovery.pt", state)
    save_tensor(temporary / "weights.pt", state["weights"])
    write_json(
        temporary / "complete.json",
        dict(
            update=update,
            scope=state["scope"],
            files={n: sha256(temporary / n) for n in ("recovery.pt", "weights.pt")},
        ),
    )
    temporary.rename(destination)
    reconcile(paths, state)
    print(f"[checkpoint] Applied update {update} saved: {destination}", flush=True)


def load_checkpoint(paths, update, recovery=True):
    folder = paths.checkpoint(update)
    receipt = read_json(folder / "complete.json")
    if receipt["update"] != update:
        raise ValueError("Checkpoint receipt update mismatch")
    if receipt["scope"] != scope(paths):
        raise ValueError("Checkpoint protocol, data, assets or features differ from this run")
    name = "recovery.pt" if recovery else "weights.pt"
    if sha256(folder / name) != receipt["files"][name]:
        raise IOError("Checkpoint checksum mismatch")
    # Recovery is our own trusted local/private backup, containing Python/NumPy RNG state.
    result = torch.load(folder / name, map_location="cpu", weights_only=not recovery)
    if recovery and result["update"] != update:
        raise ValueError("Checkpoint payload update mismatch")
    return result


def reconcile(paths, state):
    write_jsonl(paths.evidence / "training.jsonl", state["history"])
    if state.get("stage_resources"):
        write_json(paths.evidence / "resources.json", state["stage_resources"])
    write_json(
        paths.evidence / "status.json",
        dict(
            update=state["update"],
            stage=state["stage"],
            examples=state["update"] * 16,
            mixture=state["mixture"],
            training_seconds=state["training_seconds"],
        ),
    )


@torch.no_grad()
def calibrate(model, cache, manifest):
    model.eval()
    selected, seen = [], set()
    for row in manifest["sets"]["caption_train"]:
        if row["image_id"] not in seen:
            seen.add(row["image_id"])
            selected.append(row)
        if len(selected) == 64:
            break
    visual_sum = text_sum = 0.0
    visual_n = text_n = 0
    for row in selected:
        feature = cache.get(row).to(model.visual_scale.device, torch.float32)
        visual = model.connector(feature).float()
        ids = model.tokenizer.encode(row["target"], add_special_tokens=False)
        text = model.language.get_input_embeddings()(
            torch.tensor(ids, device=feature.device)
        ).float()
        visual_sum += visual.square().sum().item()
        text_sum += text.square().sum().item()
        visual_n += visual.numel()
        text_n += text.numel()
    scale = math.sqrt((text_sum / text_n) / (visual_sum / visual_n))
    model.visual_scale.fill_(scale)
    return dict(
        rule=protocol()["calibration"],
        scale=scale,
        caption_ids=[r["caption_id"] for r in selected],
        image_ids=[r["image_id"] for r in selected],
        text_rms=math.sqrt(text_sum / text_n),
        connector_rms=math.sqrt(visual_sum / visual_n),
    )


def initialize(paths):
    require_cuda()
    if latest(paths) is not None:
        return
    seed_all(SEED)
    manifest = read_json(paths.manifest)
    assert_isolation(manifest)
    model, cache = load_model(paths), FeatureCache(paths)
    calibration = calibrate(model, cache, manifest)
    write_json(paths.evidence / "calibration.json", calibration)
    write_json(paths.evidence / "protocol.json", protocol())
    write_json(paths.evidence / "environment.json", environment())
    state = dict(
        update=0,
        stage=1,
        scope=scope(paths),
        weights=model.trainable_state(),
        sampler=Sampler(manifest).state(),
        rng=rng_state(),
        optimizer=None,
        scheduler=None,
        scaler=None,
        history=[],
        mixture={},
        training_seconds=0,
        stage_resources={},
        calibration=calibration,
    )
    save_checkpoint(paths, 0, state)
    del model, cache
    gc.collect()
    torch.cuda.empty_cache()


def lr_factor(applied, total):
    warmup = math.ceil(total * 0.05)
    if applied <= warmup:
        return applied / warmup
    progress = (applied - warmup) / (total - warmup)
    return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2


def make_optimizer(model, stage):
    groups = [
        dict(
            params=list(model.connector.parameters()),
            lr=stage.connector_lr,
            peak_lr=stage.connector_lr,
        )
    ]
    if stage.number > 1:
        groups.append(
            dict(
                params=[p for p in model.language.parameters() if p.requires_grad],
                lr=stage.adapter_lr,
                peak_lr=stage.adapter_lr,
            )
        )
    return torch.optim.AdamW(groups, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8, foreach=False)


def train_segment(paths, stop=None):
    require_cuda()
    update = latest(paths)
    if update is None:
        initialize(paths)
        update = 0
    if update == 4500:
        return
    state = load_checkpoint(paths, update)
    reconcile(paths, state)
    stage = stage_at(update)
    stop = min(stop or update + 250, stage.end)
    if stop <= update or stop % 250:
        raise ValueError("Stop must be a later 250-update checkpoint within the current stage")
    seed_all(SEED)
    model = load_model(paths, adapters=update > 2250)
    model.load_trainables(state["weights"])
    if stage.number > 1:
        model.enable_adapters()
    model.train()
    if any(p.dtype != torch.float32 for p in model.parameters() if p.requires_grad):
        raise TypeError("All trainable master parameters must be FP32")
    if any(
        p.requires_grad and not n.endswith((".a", ".b"))
        for n, p in model.language.named_parameters()
    ):
        raise ValueError("Original Qwen parameters must remain frozen")
    optimizer = make_optimizer(model, stage)
    scaler = torch.amp.GradScaler("cuda")
    if state["stage"] == stage.number and state["optimizer"] is not None:
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        expected = dict(applied=update - stage.start, total=stage.end - stage.start)
        if state["scheduler"] != expected:
            raise ValueError("Scheduler progress differs from applied optimizer updates")
    restore_rng(state["rng"])
    sampler = Sampler(read_json(paths.manifest), state["sampler"])
    cache = FeatureCache(paths)
    feature_resources = cache.preload_cuda()
    history, mixture = state["history"], Counter(state["mixture"])
    resources = state["stage_resources"]
    torch.cuda.reset_peak_memory_stats()
    print(f"[stage {stage.number}] {update} → {stop}; {model.parameter_counts()}", flush=True)
    for applied in range(update + 1, stop + 1):
        started = time.perf_counter()
        rows, draw_counts = sampler.batch(stage)
        features = [cache.get(row) for row in rows]
        batch_rng = rng_state()
        factor = lr_factor(applied - stage.start, stage.end - stage.start)
        for group in optimizer.param_groups:
            group["lr"] = group["peak_lr"] * factor
        for attempt in range(32):
            optimizer.zero_grad(set_to_none=True)
            if attempt:
                restore_rng(batch_rng)
            losses = []
            for micro in range(2):
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = model.losses(
                        rows[micro * 8 : (micro + 1) * 8], features[micro * 8 : (micro + 1) * 8]
                    ).mean()
                losses.append(loss.detach().item())
                scaler.scale(loss / 2).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= old_scale:
                break
        else:
            raise FloatingPointError(
                "32 AMP retries exhausted; committed checkpoint remains available"
            )
        if applied == stage.start + 1:
            gradients = [p.grad for p in model.connector.parameters()]
            if not all(g is not None and torch.isfinite(g).all() for g in gradients):
                raise FloatingPointError("Connector gradient is missing or nonfinite")
            if not any(g.abs().sum() > 0 for g in gradients):
                raise RuntimeError("Frozen backbone blocked connector gradients")
            file = paths.evidence / "interface_checks.json"
            checks = read_json(file) if file.exists() else {}
            checks[str(stage.number)] = dict(
                **model.parameter_counts(),
                connector_gradients_finite_and_nonzero=True,
                original_qwen_frozen=True,
                trainable_dtype="float32",
                cached_patch_shape=[196, 768],
                optimizer_moments_fp32=all(
                    v.dtype == torch.float32
                    for value in optimizer.state.values()
                    for key, v in value.items()
                    if key in ("exp_avg", "exp_avg_sq")
                ),
            )
            write_json(file, checks)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        state["training_seconds"] += elapsed
        mixture.update(draw_counts)
        history.append(
            dict(
                update=applied,
                stage=stage.number,
                loss=sum(losses) / 2,
                examples=applied * 16,
                mixture=draw_counts,
                amp_retries=attempt,
                grad_norm=float(norm),
                connector_lr=optimizer.param_groups[0]["lr"],
                adapter_lr=optimizer.param_groups[-1]["lr"] if stage.number > 1 else 0,
                seconds=elapsed,
            )
        )
        print(
            f"[train] {applied}/4500 loss={sum(losses) / 2:.5f} retries={attempt} seconds={elapsed:.1f}",
            flush=True,
        )
        if applied % 250 == 0:
            old = resources.get(str(stage.number), {})
            resources[str(stage.number)] = dict(
                **model.parameter_counts(),
                **feature_resources,
                peak_gpu_bytes=max(old.get("peak_gpu_bytes", 0), torch.cuda.max_memory_allocated()),
                training_seconds=sum(r["seconds"] for r in history if r["stage"] == stage.number),
            )
            state.update(
                update=applied,
                stage=stage.number,
                scope=scope(paths),
                weights=model.trainable_state(),
                optimizer=optimizer.state_dict(),
                scheduler=dict(applied=applied - stage.start, total=stage.end - stage.start),
                scaler=scaler.state_dict(),
                sampler=sampler.state(),
                rng=rng_state(),
                mixture=dict(mixture),
                stage_resources=resources,
            )
            save_checkpoint(paths, applied, state)
            write_json(paths.evidence / "resources.json", resources)
    del model, optimizer, cache
    gc.collect()
    torch.cuda.empty_cache()
