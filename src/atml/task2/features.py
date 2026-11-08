"""Frozen deterministic SigLIP patch cache, including crash-safe shard commits."""

import gc
import time
from functools import lru_cache

import torch
from PIL import Image

from atml.common.io import environment, read_json, require_cuda, sha256, write_json

from .config import SIGLIP
from .data import fetch_images, flatten, image_path


def save_tensor(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        import os

        os.fsync(stream.fileno())
    tmp.replace(path)


def pool_patches(features):
    """Optional spatial 2×2 averaging, preserving row-major spatial order."""
    shape = features.shape
    if shape[-2:] != (196, 768):
        raise ValueError(f"Unexpected SigLIP patch shape: {shape}")
    return (
        features.reshape(*shape[:-2], 7, 2, 7, 2, 768)
        .float()
        .mean(dim=(-4, -2))
        .reshape(*shape[:-2], 49, 768)
        .to(features.dtype)
    )


def cache_features(paths):
    from transformers import AutoImageProcessor, SiglipVisionModel

    device = require_cuda()
    manifest = read_json(paths.manifest)
    fetch_images(paths, manifest)
    lock = read_json(paths.evidence / "asset_lock.json")
    folder = paths.artifacts / "features"
    index_path = folder / "index.json"
    scope = dict(selection_sha256=sha256(paths.manifest), revision=lock[SIGLIP])
    index = read_json(index_path) if index_path.exists() else dict(scope=scope, shards=[])
    if index["scope"] != scope:
        raise ValueError("Feature cache belongs to another selection/model")
    images = sorted(
        {(r["split"], r["image_id"]) for v in manifest["sets"].values() for r in flatten(v)}
    )
    if index.get("complete"):
        for shard in index["shards"]:
            if sha256(folder / shard["file"]) != shard["sha256"]:
                raise ValueError("Feature cache checksum mismatch")
        return
    processor = AutoImageProcessor.from_pretrained(SIGLIP, revision=lock[SIGLIP], use_fast=False)
    # SiglipVisionModel.last_hidden_state already includes post_layernorm.
    # Do not apply post_layernorm again or use the attention-pooled image vector.
    vision = (
        SiglipVisionModel.from_pretrained(
            SIGLIP, revision=lock[SIGLIP], torch_dtype=torch.float16, attn_implementation="sdpa"
        )
        .to(device)
        .eval()
    )
    vision.requires_grad_(False)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    existing = {tuple(i) for s in index["shards"] for i in s["images"]}
    pending = [i for i in images if i not in existing]
    offset = 0
    batch_size = index.get("batch_size", 512)
    while offset < len(pending):
        selected = pending[offset : offset + batch_size]
        pics = []
        for split, image in selected:
            with Image.open(image_path(paths, split, image)) as im:
                pics.append(im.convert("RGB"))
        inputs = output = None
        try:
            inputs = processor(images=pics, return_tensors="pt")["pixel_values"].to(
                device, torch.float16
            )
            with torch.inference_mode():
                output = vision(pixel_values=inputs).last_hidden_state
        except torch.OutOfMemoryError:
            del inputs, output, pics
            torch.cuda.empty_cache()
            if batch_size == 1:
                raise
            batch_size = max(1, batch_size // 2)
            print(f"[features] Reducing inference batch to {batch_size} after OOM", flush=True)
            continue
        if output.shape[1:] != (196, 768) or not torch.isfinite(output).all():
            raise ValueError("Invalid SigLIP final patch tensor")
        tensor = output.cpu().half()
        del inputs, output, pics
        for start in range(0, len(selected), 128):
            filename = f"shard_{len(index['shards']):04d}.pt"
            save_tensor(folder / filename, tensor[start : start + 128].clone())
            index["shards"].append(
                dict(
                    file=filename,
                    images=selected[start : start + 128],
                    sha256=sha256(folder / filename),
                )
            )
        index["batch_size"] = batch_size
        index["seconds"] = index.get("seconds", 0) + time.perf_counter() - started
        started = time.perf_counter()
        write_json(index_path, index)
        print(f"[features] {len(existing) + offset + len(selected)}/{len(images)}", flush=True)
        offset += len(selected)
    index["complete"] = True
    write_json(index_path, index)
    write_json(
        paths.evidence / "feature_cache.json",
        dict(
            **scope,
            images=len(images),
            shape=[196, 768],
            dtype="float16",
            batch_size=batch_size,
            final_norm_applications=1,
            preprocessing=processor.to_dict(),
            seconds=index["seconds"],
            peak_gpu_bytes=torch.cuda.max_memory_allocated(),
            environment=environment(),
            shards=index["shards"],
        ),
    )
    del vision, processor
    gc.collect()
    torch.cuda.empty_cache()


class FeatureCache:
    def __init__(self, paths):
        self.folder = paths.artifacts / "features"
        self.index = read_json(self.folder / "index.json")
        if not self.index.get("complete"):
            raise ValueError("Feature caching is incomplete")
        if self.index["scope"]["selection_sha256"] != sha256(paths.manifest):
            raise ValueError("Feature selection mismatch")
        self.locations = {
            tuple(image): (s["file"], i)
            for s in self.index["shards"]
            for i, image in enumerate(s["images"])
        }
        self.hashes = {s["file"]: s["sha256"] for s in self.index["shards"]}
        self.shard = lru_cache(maxsize=128)(self._read_shard)
        self.gpu_shards = {}

    def preload_cuda(self, rows=None):
        """Use spare 4090 VRAM for frozen features, retaining 12 GiB for training.

        This changes transfer cost only; cached feature values and example order
        remain identical. A CPU shard cache handles anything outside the budget.
        """
        free, total = torch.cuda.mem_get_info()
        budget = max(0, min(int(total * 0.25), free - 12 * 1024**3))
        wanted = (
            {self.locations[(r["split"], r["image_id"])][0] for r in rows}
            if rows is not None
            else set(self.hashes)
        )
        used = 0
        started = time.perf_counter()
        for name in sorted(wanted):
            value = self.shard(name)
            size = value.numel() * value.element_size()
            if used + size > budget:
                continue
            try:
                self.gpu_shards[name] = value.to("cuda")
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                break
            used += size
        self.shard.cache_clear()
        torch.cuda.synchronize()
        return dict(feature_gpu_bytes=used, feature_transfer_seconds=time.perf_counter() - started)

    def _read_shard(self, name):
        if sha256(self.folder / name) != self.hashes[name]:
            raise ValueError("Corrupt feature shard")
        return torch.load(self.folder / name, weights_only=True, map_location="cpu")

    def get(self, row, condition="correct"):
        if condition == "no_image":
            return None
        key = {
            "correct": "image_id",
            "complementary": "partner_image_id",
            "mismatched": "mismatch_image_id",
        }[condition]
        file, offset = self.locations[(row["split"], row[key])]
        return (self.gpu_shards[file] if file in self.gpu_shards else self.shard(file))[offset]
