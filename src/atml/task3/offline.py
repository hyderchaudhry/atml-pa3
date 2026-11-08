"""Resumable per-question experiments with development-only policy fitting."""

import gzip
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from atml.common.io import read_json, sha256, write_json

from .config import BUDGETS, DATASETS, GAMMAS, Paths, TAUS, protocol
from .data import load_question
from .selection import (
    choose_policy,
    methods,
    quality,
    select,
    signal_diagnostics,
    stop,
    stopping_trace,
    visible_groups,
)


def write_compressed(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps(value, separators=(",", ":"), allow_nan=False).encode()
    with temporary.open("wb") as stream:
        stream.write(gzip.compress(payload, mtime=0))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def read_compressed(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def scope(paths):
    scorer = read_json(paths.evidence / "scorer.json")
    for field, package in (
        ("math_verify", "math-verify"),
        ("sympy", "sympy"),
        ("latex_parser", "latex2sympy2_extended"),
    ):
        if scorer[field] != importlib.metadata.version(package):
            raise ValueError(
                f"Scorer package version changed: {package}; restore the pinned environment"
            )
    if scorer["implementation_sha256"] != sha256(Path(__file__).with_name("scoring.py")):
        raise ValueError("Scorer implementation changed after preparation")
    return dict(
        preparation_sha256=sha256(paths.evidence / "preparation.json"),
        scorer_sha256=sha256(paths.evidence / "scorer.json"),
        orders_sha256=sha256(paths.evidence / "orders.json"),
        protocol=protocol(),
        implementation_sha256={
            name: sha256(Path(__file__).with_name(name))
            for name in ("selection.py", "scoring.py", "offline.py", "data.py")
        },
    )


def result_path(paths, dataset, split, key):
    return (
        paths.evidence
        / "offline"
        / dataset
        / split
        / (hashlib.sha256(key.encode()).hexdigest() + ".json.gz")
    )


def policy_grid():
    result = [dict(id="always32", family="always32")]
    for tau in TAUS:
        result.append(dict(id=f"agreement:{tau}", family="agreement", tau=tau))
        for gamma in GAMMAS:
            result.append(dict(id=f"quality:{tau}:{gamma}", family="quality", tau=tau, gamma=gamma))
    return result


def evaluate_question(pool, correct, equivalent, dataset, orders, policies):
    """Choices are finalized from Pool; only then index the separate correctness array."""
    fixed, adaptive = [], []
    for order_id, order_list in enumerate(orders):
        order = np.asarray(order_list, dtype=np.int64)
        if sorted(order.tolist()) != list(range(32)):
            raise ValueError("Each saved order must be a permutation of all 32 rollout IDs")
        for n in BUDGETS:
            indices = order[:n]
            groups = visible_groups([pool.answers[i] for i in indices], equivalent)
            scores, calibration = quality(pool, indices)
            # Selection takes no correctness argument. Oracle/failure accounting is retrospective.
            choices = {
                method: int(indices[select(method, groups, scores)]) for method in methods(dataset)
            }
            oracle = int(np.any(correct[indices]))
            for method, chosen in choices.items():
                fixed.append(
                    dict(
                        order=order_id,
                        n=n,
                        method=method,
                        candidate_ids=[pool.rollout_ids[i] for i in indices],
                        consumed_ids=[pool.rollout_ids[indices[0]]]
                        if method == "first"
                        else [pool.rollout_ids[i] for i in indices],
                        selected_id=pool.rollout_ids[chosen],
                        selected_answer=pool.answers[chosen],
                        accuracy=int(correct[chosen]),
                        oracle=oracle,
                        distinct_groups=len(set(groups)),
                        sampling_failure=1 - oracle,
                        selection_failure=int(oracle and not correct[chosen]),
                        tokens=int(pool.tokens[indices[0]])
                        if method == "first"
                        else int(pool.tokens[indices].sum()),
                        q05=calibration["q05"] if calibration else None,
                        q95=calibration["q95"] if calibration else None,
                    )
                )
        trace = stopping_trace(pool, order, equivalent, dataset)
        fixed32 = trace[-1]["selected"]
        for policy in policies:
            chosen = stop(trace, policy["family"], policy.get("tau", 1.0), policy.get("gamma", 0.0))
            i, n = chosen["selected"], chosen["n"]
            adaptive.append(
                dict(
                    order=order_id,
                    policy=policy["id"],
                    n=n,
                    candidate_ids=[pool.rollout_ids[j] for j in order[:n]],
                    selected_id=pool.rollout_ids[i],
                    selected_answer=pool.answers[i],
                    accuracy=int(correct[i]),
                    tokens=chosen["tokens"],
                    fixed32_accuracy=int(correct[fixed32]),
                    wrong_early_stop=int(n < 32 and not correct[i]),
                    fixed32_would_recover=int(n < 32 and not correct[i] and correct[fixed32]),
                    retrospective_correct_pool_fraction=float(np.mean(correct)),
                )
            )
    return dict(
        fixed=fixed, adaptive=adaptive, signals=signal_diagnostics(pool, correct, equivalent)
    )


def process_one(arguments):
    root, dataset, split, key, orders, policies, signature = arguments
    paths = Paths(Path(root))
    path = result_path(paths, dataset, split, key)
    if path.exists():
        saved = read_compressed(path)
        if saved["scope"] != signature or saved["key"] != key:
            raise ValueError("Offline result belongs to another source/scorer/order/protocol")
        return key
    question = load_question(paths, dataset, key)
    result = evaluate_question(
        question["pool"],
        question["correct"],
        question["scorer"].equivalent,
        dataset,
        orders,
        policies,
    )
    result.update(dataset=dataset, key=key, split=split, scope=signature)
    write_compressed(path, result)
    return key


def run_offline(paths, split, workers=4, limit=None, checkpoint_callback=None):
    if split not in ("development", "evaluation"):
        raise ValueError("Unknown question split")
    signature = scope(paths)
    splits, orders = (
        read_json(paths.evidence / "splits.json"),
        read_json(paths.evidence / "orders.json"),
    )
    frozen = None
    if split == "evaluation":
        frozen = read_json(paths.evidence / "policies.json")
        if frozen["scope"] != signature:
            raise ValueError("Frozen policies belong to a different development experiment")
    for dataset in DATASETS:
        policies = (
            policy_grid() if split == "development" else [dict(id="always32", family="always32")]
        )
        if frozen:
            selected = [frozen[dataset][family] for family in ("agreement", "quality")]
            # Both families may select the same always32 fallback; evaluate it once.
            policies = list({r["id"]: r for r in policies + selected}.values())
        keys = splits[dataset][split]
        if limit is not None:
            keys = keys[:limit]
        arguments = [
            (str(paths.root), dataset, split, key, orders[dataset][key], policies, signature)
            for key in keys
        ]
        if workers == 1:
            iterator = map(process_one, arguments)
            for i, _ in enumerate(iterator, 1):
                print(f"[offline] {dataset}/{split} {i}/{len(keys)}", flush=True)
                if checkpoint_callback and i % 50 == 0:
                    checkpoint_callback()
        else:
            with ProcessPoolExecutor(
                max_workers=workers, mp_context=multiprocessing.get_context("spawn")
            ) as executor:
                for i, _ in enumerate(executor.map(process_one, arguments, chunksize=1), 1):
                    print(f"[offline] {dataset}/{split} {i}/{len(keys)}", flush=True)
                    if checkpoint_callback and i % 50 == 0:
                        checkpoint_callback()
        if limit is None:
            files = {
                result_path(paths, dataset, split, k).name: sha256(
                    result_path(paths, dataset, split, k)
                )
                for k in keys
            }
            write_json(
                paths.evidence / "offline" / dataset / split / "complete.json",
                dict(scope=signature, questions=len(keys), files=files),
            )


def fit_policies(paths):
    target = paths.evidence / "policies.json"
    signature = scope(paths)
    if target.exists():
        if read_json(target)["scope"] != signature:
            raise ValueError("Do not refit policies after changing source/scorer/order settings")
        return
    splits = read_json(paths.evidence / "splits.json")
    result = dict(scope=signature, selection_rule=protocol()["stopping"], development={})
    for dataset in DATASETS:
        # This function reads development result paths exclusively.
        keys = splits[dataset]["development"]
        totals = {p["id"]: dict(accuracy=0.0, tokens=0.0) for p in policy_grid()}
        hashes = {}
        for key in keys:
            path = result_path(paths, dataset, "development", key)
            data = read_compressed(path)
            if data["scope"] != signature:
                raise ValueError("Development result scope mismatch")
            hashes[path.name] = sha256(path)
            for row in data["adaptive"]:
                for metric in ("accuracy", "tokens"):
                    totals[row["policy"]][metric] += row[metric] / (len(keys) * 10)
        baseline = totals["always32"]["accuracy"]
        selected = {}
        for family in ("agreement", "quality"):
            candidates = [
                {**p, **totals[p["id"]]}
                for p in policy_grid()
                if p["family"] in (family, "always32")
            ]
            selected[family] = choose_policy(candidates, baseline)
        result[dataset] = selected
        result["development"][dataset] = dict(baseline_accuracy=baseline, grid=totals, files=hashes)
    write_json(target, result)
    print("[policies] Selected on development only and frozen before held-out analysis", flush=True)
