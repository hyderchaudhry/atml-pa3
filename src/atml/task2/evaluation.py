"""Resumable development likelihood and final four-condition evaluation."""

import gc
import hashlib
import json

import torch

from atml.common.io import (
    append_jsonl_many,
    read_json,
    recover_jsonl,
    require_cuda,
    seed_all,
    sha256,
    write_json,
)

from .config import CONDITIONS, DEV_UPDATES, SEED
from .data import flatten
from .features import FeatureCache
from .metrics import normalize, soft_score
from .model import load_model
from .training import load_checkpoint


def journal_scope(path, scope):
    target = path.with_suffix(".scope.json")
    if target.exists() and read_json(target) != scope:
        raise ValueError(f"Evaluation journal belongs to another run: {path}")
    write_json(target, scope)
    rows = recover_jsonl(path)
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate evaluation evidence: {path}")
    return rows


def identity(paths, update):
    return dict(
        selection_sha256=sha256(paths.manifest),
        update=update,
        weights_sha256=sha256(paths.checkpoint(update) / "weights.pt"),
        feature_index_sha256=sha256(paths.artifacts / "features" / "index.json"),
    )


def evaluate(paths, update, final=False):
    require_cuda()
    if final and update != 4500:
        raise ValueError("Final-test evaluation uses only the completed Stage 3 model")
    if not final and update not in DEV_UPDATES:
        raise ValueError("Not a prescribed development checkpoint")
    seed_all(SEED)
    weights = load_checkpoint(paths, update, recovery=False)
    model = load_model(paths, adapters=update > 2250)
    model.load_trainables(weights)
    model.eval()
    cache = FeatureCache(paths)
    manifest = read_json(paths.manifest)
    folder = paths.evidence / ("final" if final else f"development/update_{update:04d}")
    completed = {}
    names = (
        ("grounding_test", "natural_test")
        if final
        else ("caption_dev", "grounding_dev", "natural_dev")
    )
    if hasattr(cache, "preload_cuda"):
        cache.preload_cuda([r for name in names for r in flatten(manifest["sets"][name])])
    batch_size = 8 if final else 32
    torch.cuda.reset_peak_memory_stats()
    for name in names:
        examples = flatten(manifest["sets"][name])
        conditions = (
            ("correct",)
            if name.startswith("caption")
            else (CONDITIONS if final else ("correct", "complementary", "no_image"))
        )
        for condition in conditions:
            path = folder / f"{name}_{condition}.jsonl"
            signature = dict(
                **identity(paths, update),
                suite=name,
                condition=condition,
                batch=batch_size,
                generation_cap=16 if final else 0,
                termination_in_nll=name.startswith("caption"),
                scorer="task2-vqa-1",
            )
            saved = journal_scope(path, signature)
            known = {r["id"] for r in saved}
            expected_ids = {r["id"] for r in examples}
            if not known <= expected_ids:
                raise ValueError("Journal contains examples outside the fixed evaluation set")
            pending = [r for r in examples if r["id"] not in known]
            for offset in range(0, len(pending), batch_size):
                batch = pending[offset : offset + batch_size]
                features = [cache.get(row, condition) for row in batch]
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                    nlls = model.losses(
                        batch, features, include_eot=name.startswith("caption")
                    ).tolist()
                    generations = model.generate(batch, features) if final else [{} for _ in batch]
                records = []
                for row, nll, generation in zip(batch, nlls, generations):
                    result = dict(**row, **generation, condition=condition, nll=nll, update=update)
                    if final:
                        prediction = generation["prediction"]
                        partner = next(
                            r
                            for r in examples
                            if r["pair_id"] == row["pair_id"] and r["id"] != row["id"]
                        )
                        result.update(
                            exact=int(normalize(prediction) == normalize(row["target"])),
                            partner_exact=int(
                                normalize(prediction) == normalize(partner["target"])
                            ),
                            accuracy=(
                                soft_score(prediction, row["references"])
                                if name.startswith("natural")
                                else float(normalize(prediction) == normalize(row["target"]))
                            ),
                            partner_target=partner["target"],
                            input_image_id=(
                                None
                                if condition == "no_image"
                                else row[
                                    {
                                        "correct": "image_id",
                                        "complementary": "partner_image_id",
                                        "mismatched": "mismatch_image_id",
                                    }[condition]
                                ]
                            ),
                        )
                    records.append(result)
                append_jsonl_many(path, records)
                saved.extend(records)
                print(
                    f"[{'final' if final else 'dev'} {update}] {name}/{condition} {len(saved)}/{len(examples)}",
                    flush=True,
                )
            completed[f"{name}/{condition}"] = len(saved)
    write_json(
        folder / "complete.json",
        dict(
            identity=identity(paths, update),
            counts=completed,
            journals={p.name: sha256(p) for p in sorted(folder.glob("*.jsonl"))},
            peak_gpu_bytes=torch.cuda.max_memory_allocated(),
            evaluation_config_sha256=hashlib.sha256(
                json.dumps(signature, sort_keys=True).encode()
            ).hexdigest(),
        ),
    )
    del model, cache
    gc.collect()
    torch.cuda.empty_cache()
