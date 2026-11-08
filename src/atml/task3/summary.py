"""Machine-readable results and scientific figures; no report prose is generated."""

import csv
import hashlib
import html
import json
import os
import zipfile
from collections import Counter, defaultdict

os.environ["MPLBACKEND"] = "Agg"
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from atml.common.io import read_json, sha256, write_json

from .config import DATASETS
from .data import load_question
from .live import CALLS, live_scope, path_for, strategies, validate_live
from .offline import read_compressed, result_path, scope
from .selection import bootstrap_questions, quality


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean_finite(values):
    finite = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(finite)) if finite else None


def offline_tables(paths):
    splits = read_json(paths.evidence / "splits.json")
    frozen = read_json(paths.evidence / "policies.json")
    signature = scope(paths)
    if frozen["scope"] != signature:
        raise ValueError(
            "Frozen policies/source implementation changed; use the original code to resume"
        )
    main, differences, signals, token_signals, adaptive, adaptive_questions = [], [], [], [], [], []
    exemplars = {}
    for dataset in DATASETS:
        fixed_values, policy_values = defaultdict(list), defaultdict(list)
        score_values, trajectory_values = defaultdict(list), []
        cases = dict(disagreement=[], sampling_failure=[], selection_failure=[], premature_stop=[])
        for key in splits[dataset]["evaluation"]:
            data = read_compressed(result_path(paths, dataset, "evaluation", key))
            if data["scope"] != signature:
                raise ValueError("Held-out result scope mismatch")
            qfixed = defaultdict(list)
            for row in data["fixed"]:
                qfixed[(row["method"], row["n"])].append(row)
            for setting, rows in qfixed.items():
                fixed_values[setting].append(
                    {
                        k: float(np.mean([r[k] for r in rows]))
                        for k in (
                            "accuracy",
                            "oracle",
                            "distinct_groups",
                            "sampling_failure",
                            "selection_failure",
                            "tokens",
                        )
                    }
                )
            qpolicies = defaultdict(list)
            for row in data["adaptive"]:
                qpolicies[row["policy"]].append(row)
            for policy, rows in qpolicies.items():
                result = {
                    k: float(np.mean([r[k] for r in rows]))
                    for k in (
                        "accuracy",
                        "tokens",
                        "n",
                        "wrong_early_stop",
                        "fixed32_would_recover",
                    )
                }
                counts = Counter(r["n"] for r in rows)
                result.update({f"stop_{n}": counts[n] / 10 for n in (4, 8, 16, 32)})
                policy_values[policy].append(result)
                adaptive_questions.append(dict(dataset=dataset, key=key, policy=policy, **result))
            for name, values in data["signals"]["scores"].items():
                score_values[name].append(values)
            if dataset == "mathvista":
                diag = data["signals"]
                trajectory_values.append(
                    dict(
                        correlation=mean_finite(diag["token_correlations"]),
                        defined_rollouts=sum(v is not None for v in diag["token_correlations"]),
                        covariance=float(np.mean(diag["token_covariances"])),
                        fusion_difference=float(np.mean(diag["fusion_difference"])),
                    )
                )
            at32 = [r for r in data["fixed"] if r["n"] == 32 and r["order"] == 0]
            if len({r["selected_id"] for r in at32}) > 1 and len(cases["disagreement"]) < 2:
                cases["disagreement"].append(key)
            if at32[0]["sampling_failure"] and len(cases["sampling_failure"]) < 2:
                cases["sampling_failure"].append(key)
            if any(r["selection_failure"] for r in at32) and len(cases["selection_failure"]) < 2:
                cases["selection_failure"].append(key)
            if (
                any(r["fixed32_would_recover"] for r in data["adaptive"])
                and len(cases["premature_stop"]) < 2
            ):
                cases["premature_stop"].append(key)
        exemplars[dataset] = cases
        for (method, n), values in sorted(fixed_values.items()):
            accuracy = np.asarray([r["accuracy"] for r in values])
            ci = bootstrap_questions(accuracy)
            row = dict(
                dataset=dataset,
                method=method,
                n=n,
                questions=len(values),
                accuracy=ci["mean"][0],
                accuracy_low=ci["low"][0],
                accuracy_high=ci["high"][0],
            )
            row.update(
                {
                    k: float(np.mean([r[k] for r in values]))
                    for k in (
                        "oracle",
                        "distinct_groups",
                        "sampling_failure",
                        "selection_failure",
                        "tokens",
                    )
                }
            )
            main.append(row)
            for reference in ("first", "consensus"):
                baseline = np.asarray([r["accuracy"] for r in fixed_values[(reference, n)]])
                stats = bootstrap_questions(accuracy - baseline)
                differences.append(
                    dict(
                        dataset=dataset,
                        method=method,
                        n=n,
                        reference=reference,
                        mean=stats["mean"][0],
                        low=stats["low"][0],
                        high=stats["high"][0],
                    )
                )
        for name, values in sorted(score_values.items()):
            result = dict(dataset=dataset, score=name, questions=len(values))
            result.update({k: mean_finite([r[k] for r in values]) for k in values[0]})
            ci = bootstrap_questions([r["top_minus_bottom"] for r in values])
            result.update(top_minus_bottom_low=ci["low"][0], top_minus_bottom_high=ci["high"][0])
            signals.append(result)
        if trajectory_values:
            token_signals.append(
                dict(
                    dataset=dataset,
                    questions=len(trajectory_values),
                    **{
                        k: mean_finite([r[k] for r in trajectory_values])
                        for k in trajectory_values[0]
                    },
                )
            )
        policy_names = [("fixed32", "always32")] + [
            (family, frozen[dataset][family]["id"]) for family in ("agreement", "quality")
        ]
        for family, policy in policy_names:
            values = policy_values[policy]
            ci = bootstrap_questions([r["accuracy"] for r in values])
            row = dict(
                dataset=dataset,
                family=family,
                policy=policy,
                accuracy=ci["mean"][0],
                accuracy_low=ci["low"][0],
                accuracy_high=ci["high"][0],
            )
            row.update(
                {k: float(np.mean([r[k] for r in values])) for k in values[0] if k != "accuracy"}
            )
            baseline = np.asarray([r["accuracy"] for r in policy_values["always32"]])
            diff = bootstrap_questions(np.asarray([r["accuracy"] for r in values]) - baseline)
            row.update(
                accuracy_difference_fixed32=diff["mean"][0],
                difference_low=diff["low"][0],
                difference_high=diff["high"][0],
            )
            adaptive.append(row)
    tables = paths.evidence / "tables"
    for name, rows in (
        ("sampling_selection", main),
        ("main_accuracy_differences", differences),
        ("signal_associations", signals),
        ("token_confidence_vision", token_signals),
        ("adaptive_evaluation", adaptive),
        ("adaptive_per_question", adaptive_questions),
    ):
        write_csv(tables / (name + ".csv"), rows)
    development = []
    for dataset in DATASETS:
        for policy, values in frozen["development"][dataset]["grid"].items():
            development.append(dict(dataset=dataset, policy=policy, **values))
    write_csv(tables / "adaptive_development.csv", development)
    write_json(
        paths.evidence / "illustrative_selection.json",
        dict(
            trajectory_rule="first three held-out question IDs in saved permutation; first two rollout IDs in saved order 0",
            case_rule="first two question IDs in saved held-out permutation satisfying each named retrospective condition",
            cases=exemplars,
        ),
    )
    return main, adaptive


def live_tables(paths):
    keys = read_json(paths.evidence / "splits.json")["live_math"]
    data = [row for key in keys for row in strategies(paths, key)]
    write_csv(paths.evidence / "tables/live_per_question.csv", data)
    summary = []
    for name in ("long", "parallel", "refinement"):
        rows = [r for r in data if r["strategy"] == name]
        ci = bootstrap_questions([r["accuracy"] for r in rows])
        initial_wrong = sum(not r["initial_accuracy"] for r in rows)
        initial_correct = len(rows) - initial_wrong
        summary.append(
            dict(
                strategy=name,
                questions=len(rows),
                accuracy=ci["mean"][0],
                accuracy_low=ci["low"][0],
                accuracy_high=ci["high"][0],
                mean_input_tokens=float(np.mean([r["input_tokens"] for r in rows])),
                mean_output_tokens=float(np.mean([r["output_tokens"] for r in rows])),
                mean_retained_output_tokens=float(
                    np.mean([r["retained_output_tokens"] for r in rows])
                ),
                total_seconds=sum(r["seconds"] for r in rows),
                mean_seconds=float(np.mean([r["seconds"] for r in rows])),
                peak_gpu_bytes=max(r["peak_gpu_bytes"] for r in rows),
                extraction_failure_rate=float(np.mean([r["extraction_failed"] for r in rows])),
                final_truncation_rate=float(np.mean([r["final_truncation"] for r in rows])),
                any_truncation_rate=float(np.mean([r["any_truncation"] for r in rows])),
                wrong_to_correct=sum(r["wrong_to_correct"] for r in rows),
                correct_to_wrong=sum(r["correct_to_wrong"] for r in rows),
                wrong_to_correct_rate=sum(r["wrong_to_correct"] for r in rows) / initial_wrong
                if initial_wrong
                else None,
                correct_to_wrong_rate=sum(r["correct_to_wrong"] for r in rows) / initial_correct
                if initial_correct
                else None,
            )
        )
    write_csv(paths.evidence / "tables/live_summary.csv", summary)
    paired = []
    for method, reference in (
        ("parallel", "long"),
        ("refinement", "long"),
        ("refinement", "parallel"),
    ):
        values = {
            name: np.asarray([r["accuracy"] for r in data if r["strategy"] == name])
            for name in (method, reference)
        }
        ci = bootstrap_questions(values[method] - values[reference])
        paired.append(
            dict(
                method=method,
                reference=reference,
                mean=ci["mean"][0],
                low=ci["low"][0],
                high=ci["high"][0],
            )
        )
    write_csv(paths.evidence / "tables/live_accuracy_differences.csv", paired)
    refinements = [r for r in data if r["strategy"] == "refinement"]
    write_json(
        paths.evidence / "live_examples.json",
        dict(
            rule="first five live question IDs, plus first two in the saved order for each revision transition",
            fixed=keys[:5],
            wrong_to_correct=[r["key"] for r in refinements if r["wrong_to_correct"]][:2],
            correct_to_wrong=[r["key"] for r in refinements if r["correct_to_wrong"]][:2],
        ),
    )


def plots(paths, main, adaptive):
    folder = paths.evidence / "plots"
    folder.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for i, dataset in enumerate(DATASETS):
        for method in sorted({r["method"] for r in main if r["dataset"] == dataset}):
            rows = sorted(
                [r for r in main if r["dataset"] == dataset and r["method"] == method],
                key=lambda r: r["n"],
            )
            axes[i, 0].plot(
                [r["n"] for r in rows], [r["accuracy"] for r in rows], marker=".", label=method
            )
            axes[i, 1].plot(
                [r["tokens"] for r in rows], [r["accuracy"] for r in rows], marker=".", label=method
            )
        base = sorted(
            [r for r in main if r["dataset"] == dataset and r["method"] == "consensus"],
            key=lambda r: r["n"],
        )
        axes[i, 0].plot(
            [r["n"] for r in base], [r["oracle"] for r in base], "k--", label="oracle coverage"
        )
        axes[i, 2].plot([r["n"] for r in base], [r["distinct_groups"] for r in base], marker=".")
        for j, label in enumerate(
            ("Candidate budget", "Retained output-token cost", "Candidate budget")
        ):
            axes[i, j].set_xlabel(label)
            axes[i, j].set_title(dataset)
        axes[i, 0].set_ylabel("Answer accuracy / oracle coverage")
        axes[i, 1].set_ylabel("Answer accuracy")
        axes[i, 2].set_ylabel("Distinct answer groups")
        axes[i, 0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(folder / "sampling_selection.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, dataset in zip(axes, DATASETS):
        fixed = sorted(
            [r for r in main if r["dataset"] == dataset and r["method"] == "consensus"],
            key=lambda r: r["n"],
        )
        axis.plot(
            [r["tokens"] for r in fixed],
            [r["accuracy"] for r in fixed],
            ".-",
            label="fixed-N consensus",
        )
        for row in [r for r in adaptive if r["dataset"] == dataset]:
            axis.errorbar(
                row["tokens"],
                row["accuracy"],
                yerr=[
                    [row["accuracy"] - row["accuracy_low"]],
                    [row["accuracy_high"] - row["accuracy"]],
                ],
                fmt="o",
                label=row["family"],
            )
        axis.set(title=dataset, xlabel="Mean retained output tokens", ylabel="Answer accuracy")
        axis.legend()
    fig.tight_layout()
    fig.savefig(folder / "adaptive_accuracy_cost.png", dpi=160)
    plt.close(fig)


def qualitative(paths):
    splits, orders = (
        read_json(paths.evidence / "splits.json"),
        read_json(paths.evidence / "orders.json"),
    )
    selected = read_json(paths.evidence / "illustrative_selection.json")
    body = [
        "<meta charset='utf-8'><style>body{font-family:sans-serif;max-width:1200px;margin:2em auto}pre{white-space:pre-wrap}details{margin:1em 0}img{max-width:100%}</style>",
        "<h1>Task 3 evidence examples</h1><pre>"
        + html.escape(json.dumps({k: v for k, v in selected.items() if k != "cases"}, indent=2))
        + "</pre>",
    ]
    for dataset in DATASETS:
        trajectory_keys = splits[dataset]["evaluation"][:3]
        keys = list(
            dict.fromkeys(
                trajectory_keys
                + [q for values in selected["cases"][dataset].values() for q in values]
            )
        )
        for key in keys:
            question = load_question(paths, dataset, key)
            result = read_compressed(result_path(paths, dataset, "evaluation", key))
            body.append(
                "<h2>"
                + html.escape(dataset + " / " + key)
                + "</h2><pre>"
                + html.escape(question["generations"][0]["question_text"])
                + "</pre>"
            )
            body.append(
                "<pre>Reference: "
                + html.escape(str(question["generations"][0]["gold_answer"]))
                + "</pre>"
            )
            rows = [r for r in result["fixed"] if r["n"] == 32 and r["order"] == 0]
            body.append("<pre>" + html.escape(json.dumps(rows, indent=2)) + "</pre>")
            body.append(
                "<pre>"
                + html.escape(
                    json.dumps([r for r in result["adaptive"] if r["order"] == 0], indent=2)
                )
                + "</pre>"
            )
            ids = (
                orders[dataset][key][0][:2]
                if key in trajectory_keys
                else list(dict.fromkeys(r["selected_id"] for r in rows))[:3]
            )
            for rid in ids:
                raw = question["generations"][rid]
                body.append(
                    "<details><summary>Rollout "
                    + str(rid)
                    + "</summary><pre>"
                    + html.escape(raw["generated_text"])
                    + "</pre></details>"
                )
            if key in trajectory_keys:
                visual = question["pool"].vision is not None
                fig, axes = plt.subplots(3 if visual else 2, 1, figsize=(11, 7 if visual else 5))
                _, calibration = quality(question["pool"], np.arange(32))
                for rid in ids:
                    pool = question["pool"]
                    axes[0].plot(pool.kl[rid], label=f"rollout {rid} KL")
                    axes[1].plot(
                        pool.vision[rid] if pool.vision is not None else pool.logp[rid],
                        label=f"rollout {rid}",
                    )
                    if visual:
                        c = calibration["calibrated"][rid]
                        axes[2].plot(c, label=f"rollout {rid} confidence")
                        axes[2].plot(
                            c * pool.vision[rid], linestyle=":", label=f"rollout {rid} token fusion"
                        )
                axes[0].set_ylabel("Token KL(U || p)")
                axes[1].set_ylabel(
                    "Visual engagement" if dataset == "mathvista" else "Sampled-token log p"
                )
                if visual:
                    axes[2].set_ylabel("N=32 calibrated confidence / fusion")
                axes[-1].set_xlabel("Retained generated-token position")
                for axis in axes:
                    axis.legend()
                fig.tight_layout()
                filename = (
                    "trajectory_"
                    + dataset
                    + "_"
                    + hashlib.sha256(key.encode()).hexdigest()[:12]
                    + ".png"
                )
                fig.savefig(paths.evidence / "plots" / filename, dpi=140)
                plt.close(fig)
                body.append("<img src='plots/" + filename + "' alt='Supplied token trajectories'>")
    example_file = paths.evidence / "live_examples.json"
    examples = read_json(example_file) if example_file.exists() else {}
    live_keys = list(
        dict.fromkeys(
            splits["live_math"][:5]
            + examples.get("wrong_to_correct", [])
            + examples.get("correct_to_wrong", [])
        )
    )
    for key in live_keys:
        if not all(path_for(paths, key, kind).exists() for kind in CALLS):
            continue
        body.append("<h2>Live / " + html.escape(key) + "</h2>")
        for kind in CALLS:
            row = read_json(path_for(paths, key, kind))
            body.append(
                "<details><summary>"
                + html.escape(kind)
                + "</summary><pre>"
                + html.escape(row["generated_text"])
                + "</pre></details>"
            )
    (paths.evidence / "qualitative.html").write_text("\n".join(body), encoding="utf8")


def audit(paths):
    checks = {}
    for name in ("preparation.json", "splits.json", "orders.json", "scorer.json", "policies.json"):
        checks[name] = (paths.evidence / name).exists()
    if not all(checks.values()):
        result = dict(
            checks=checks,
            computational_complete=False,
            pending=[k for k, v in checks.items() if not v],
        )
        write_json(paths.evidence / "completeness.json", result)
        return result
    splits = read_json(paths.evidence / "splits.json")
    signature = scope(paths)
    preparation = read_json(paths.evidence / "preparation.json")
    db = paths.artifacts / "prepared.sqlite"
    checks["prepared_source_hash"] = db.exists() and sha256(db) == preparation["database_sha256"]
    checks["supplied_sources_64000_rollouts"] = (
        preparation["questions_per_dataset"] == 1000
        and all(r["rows"] == 32000 for r in preparation["files"].values())
        and len(preparation["files"]) == 5
    )
    for dataset in DATASETS:
        all_keys = splits[dataset]["development"] + splits[dataset]["evaluation"]
        expected = np.random.default_rng(6304).permutation(sorted(all_keys)).tolist()
        checks[f"splits/{dataset}"] = (
            len(splits[dataset]["development"]) == 200
            and len(splits[dataset]["evaluation"]) == 800
            and not set(splits[dataset]["development"]) & set(splits[dataset]["evaluation"])
            and len(set(all_keys)) == 1000
            and all_keys == expected
        )
        for split, count in (("development", 200), ("evaluation", 800)):
            marker = paths.evidence / "offline" / dataset / split / "complete.json"
            checks[f"offline/{dataset}/{split}"] = marker.exists()
            if marker.exists():
                record = read_json(marker)
                checks[f"offline_hashes/{dataset}/{split}"] = (
                    record["scope"] == signature
                    and record["questions"] == count
                    and set(record["files"])
                    == {result_path(paths, dataset, split, k).name for k in splits[dataset][split]}
                    and all(
                        sha256(marker.parent / n) == digest for n, digest in record["files"].items()
                    )
                )
    policies = read_json(paths.evidence / "policies.json")
    checks["development_only_policy_scope"] = policies["scope"] == signature
    checks["frozen_development_hashes"] = all(
        sha256(paths.evidence / "offline" / dataset / "development" / name) == digest
        for dataset in DATASETS
        for name, digest in policies["development"][dataset]["files"].items()
    )
    checks["live_first50_math"] = (
        splits["live_math"] == splits["math"]["evaluation"][:50] and len(splits["live_math"]) == 50
    )
    marker = paths.evidence / "live/complete.json"
    checks["live_complete"] = marker.exists()
    if marker.exists():
        record = read_json(marker)
        checks["live_250_calls"] = record["questions"] == 50 and record["calls"] == 250
        checks["live_hashes"] = (
            all(sha256(marker.parent / n) == digest for n, digest in record["files"].items())
            and len(record["files"]) == 250
            and record["scope"] == live_scope(paths)
        )
        try:
            validate_live(paths, write_marker=False)
            checks["live_prompts_caps_seeds_gpu"] = True
        except (ValueError, KeyError, FileNotFoundError):
            checks["live_prompts_caps_seeds_gpu"] = False
    for name in (
        "tables/sampling_selection.csv",
        "tables/main_accuracy_differences.csv",
        "tables/signal_associations.csv",
        "tables/token_confidence_vision.csv",
        "tables/adaptive_development.csv",
        "tables/adaptive_evaluation.csv",
        "tables/adaptive_per_question.csv",
        "tables/live_summary.csv",
        "tables/live_per_question.csv",
        "tables/live_accuracy_differences.csv",
        "plots/sampling_selection.png",
        "plots/adaptive_accuracy_cost.png",
        "qualitative.html",
    ):
        checks[name] = (paths.evidence / name).exists()
    result = dict(
        checks=checks,
        computational_complete=all(checks.values()),
        pending=[k for k, v in checks.items() if not v],
    )
    write_json(paths.evidence / "completeness.json", result)
    return result


def summarize(paths):
    main, adaptive = offline_tables(paths)
    if (paths.evidence / "live/complete.json").exists():
        live_tables(paths)
    plots(paths, main, adaptive)
    qualitative(paths)
    return audit(paths)


def export(paths):
    target = paths.root / "task3-evidence.zip"
    temporary = target.with_suffix(".tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(paths.evidence.rglob("*")):
            if file.is_file() and not file.name.endswith((".tmp", ".partial.json", ".crash_tail")):
                archive.write(file, file.relative_to(paths.evidence.parent))
    temporary.replace(target)
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None:
            raise IOError("Evidence ZIP integrity check failed")
    digest = sha256(target)
    target.with_suffix(".zip.sha256").write_text(digest + "  " + target.name + "\n")
    return target
