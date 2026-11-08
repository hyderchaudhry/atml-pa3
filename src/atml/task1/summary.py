"""Machine-readable tables and figures; no generated report prose or interpretation."""

from __future__ import annotations

import csv
import os

# Notebook kernels can export an inline backend unavailable in the CLI's venv.
# Set this before importing matplotlib: import itself validates MPLBACKEND.
os.environ["MPLBACKEND"] = "Agg"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from atml.common.io import read_json, read_jsonl, write_json
from atml.task1.config import BENCHMARKS, CONDITIONS, METHODS
from atml.task1.qualitative import ANNOTATION_FIELDS, group_and_prepare_annotations
from atml.task1.storage import checkpoint_available


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(paths):
    directory = paths.evidence / "tables"
    figures = paths.evidence / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    benchmark_rows, all_scores = [], {}
    for condition in CONDITIONS:
        scores = read_json(paths.evidence / "evaluations" / condition / "summary.json")["scores"]
        all_scores[condition] = scores
    baseline = all_scores["baseline"]
    for condition, scores in all_scores.items():
        row = {"condition": condition}
        for task, (_, metric) in BENCHMARKS.items():
            row[f"{task}_primary_percent"] = 100 * scores[task][metric]
            row[f"{task}_change_pp"] = 100 * (scores[task][metric] - baseline[task][metric])
            if "acc,none" in scores[task]:
                row[f"{task}_raw_accuracy_percent"] = 100 * scores[task]["acc,none"]
        benchmark_rows.append(row)
    write_csv(directory / "benchmarks.csv", benchmark_rows)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for axis, (task, (_, metric)) in zip(axes.flat, BENCHMARKS.items()):
        values = [100 * all_scores[c][task][metric] for c in CONDITIONS]
        axis.bar(list(CONDITIONS), values)
        axis.set(title=f"{task}: {metric.split(',')[0]}", ylabel="Score (%)")
        axis.tick_params(axis="x", rotation=65, labelsize=8)
    fig.savefig(figures / "benchmarks.png", dpi=180)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(9, 5), layout="constrained")
    resources, regularization = [], []
    for method in METHODS:
        run = paths.evidence / "runs" / method
        logs = read_jsonl(run / "training.jsonl")
        if len(logs) != 1250 or [r["update"] for r in logs] != list(range(1, 1251)):
            raise ValueError(f"{method}: expected exactly 1250 ordered training updates.")
        axis.plot(
            [r["update"] for r in logs], [r["sft_loss"] for r in logs], label=method, linewidth=0.9
        )
        config = read_json(run / "configuration.json")
        stats = read_json(run / "resources.json")
        if stats.get("successful_updates") != 1250:
            raise ValueError(
                f"{method}: the required 1,250 applied optimizer updates are incomplete."
            )
        checkpoints = read_jsonl(run / "checkpoints.jsonl")
        bases = read_json(paths.evidence / "bases.json") if method == "oplora16" else {}
        resources.append(
            {
                "method": method,
                "trainable_parameters": config["trainable_parameters"],
                "training_seconds": stats["train_seconds"],
                "setup_seconds": config["setup_seconds"],
                "peak_gpu_bytes": stats["peak_gpu_bytes"],
                "final_checkpoint_bytes": checkpoints[-1]["total_bytes"],
                "all_checkpoint_bytes": sum(c["total_bytes"] for c in checkpoints),
                "checkpoint_seconds": stats["checkpoint_seconds"],
                "export_seconds": stats["export_seconds"],
                "reference_storage_bytes": config["reference_storage_bytes"],
                "reference_creation_seconds": config["reference_creation_seconds"],
                "reference_transfer_and_penalty_seconds": stats["regularization_seconds"],
                "reference_transfer_seconds": stats["reference_transfer_seconds"],
                "reference_transfer_bytes": stats["reference_transfer_bytes"],
                "basis_storage_bytes": bases.get("tensor_bytes", 0),
                "basis_preprocessing_seconds": bases.get("preprocessing_seconds", 0),
                "skipped_updates": stats["skipped_updates"],
            }
        )
        if method in ("full_ft", "l2sp"):
            displacement = read_jsonl(run / "displacement.jsonl")
            regularization.append(
                {
                    "method": method,
                    "lambda": METHODS[method].l2_lambda,
                    "final_displacement_l2": displacement[-1]["l2"],
                    "final_training_sft": logs[-1]["sft_loss"],
                    "last_pre_step_omega": logs[-1]["omega"],
                    "last_pre_step_lambda_omega": logs[-1]["lambda_omega"],
                    "last_pre_step_total_loss": logs[-1]["total_loss"],
                }
            )
    axis.set(xlabel="Applied optimizer update", ylabel="Response-token SFT loss")
    axis.legend()
    fig.savefig(figures / "learning_curves.png", dpi=180)
    plt.close(fig)
    write_csv(directory / "resources.csv", resources)
    write_csv(directory / "regularization.csv", regularization)
    penalty = read_jsonl(paths.evidence / "runs" / "l2sp" / "training.jsonl")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for field in ("sft_loss", "lambda_omega", "total_loss"):
        axes[0].plot([r["update"] for r in penalty], [r[field] for r in penalty], label=field)
    axes[0].set(xlabel="Optimizer update", ylabel="Loss", title="L2-SP objective components")
    axes[0].legend()
    axes[1].bar(
        [r["method"] for r in regularization], [r["final_displacement_l2"] for r in regularization]
    )
    axes[1].set(
        ylabel="Full-model displacement (L2 norm)", title="Final displacement from starting model"
    )
    fig.savefig(figures / "regularization.png", dpi=180)
    plt.close(fig)
    diagnostic_rows = []
    for condition in CONDITIONS:
        if condition == "baseline":
            continue
        data = read_json(paths.evidence / "diagnostics" / f"{condition}.json")
        for row in data["layers"]:
            diagnostic_rows.append(
                {
                    "condition": condition,
                    "layer": row["layer"],
                    "rho_128": row["rho"],
                    "eta": row["eta"],
                    "zero_update": row["zero_update"],
                    "math_R_mean": row["activation"]["gsm8k_cot"]["mean"],
                    "nonmath_R_mean": row["activation"]["hellaswag"]["mean"],
                    "left_constraint_relative": row["left_constraint_relative"],
                    "right_constraint_relative": row["right_constraint_relative"],
                }
            )
    write_csv(directory / "diagnostics.csv", diagnostic_rows)
    group_and_prepare_annotations(paths)


def audit(paths):
    """Completeness audit distinguishes unrun experiments and pending human annotations."""
    checks = []

    def check(name, success, detail=""):
        checks.append({"requirement": name, "complete": bool(success), "detail": detail})

    for filename in (
        "protocol.json",
        "asset_lock.json",
        "preparation.json",
        "training_ids.json",
        "probe_ids_and_prompts.json",
        "task_definitions.json",
        "bases.json",
        "activation_cache.json",
        "qualitative_prompts.json",
        "first_plot_questions.yaml",
    ):
        check(filename, (paths.evidence / filename).exists())
    for condition in CONDITIONS:
        directory = paths.evidence / "evaluations" / condition
        check(f"{condition}: aggregate benchmarks", (directory / "summary.json").exists())
        for task, (expected, _) in BENCHMARKS.items():
            filename = directory / f"{task}_samples.jsonl"
            rows = read_jsonl(filename) if filename.exists() else []
            unique = {row["doc_id"] for row in rows}
            complete = len(rows) == expected and unique == set(range(expected))
            if task == "gsm8k_cot":
                complete = complete and all(
                    "extracted_answer" in r and "truncated" in r.get("generation", {}) for r in rows
                )
            check(
                f"{condition}: {task} complete per-example evidence",
                complete,
                f"{len(unique)}/{expected}",
            )
        filename = paths.evidence / "qualitative" / f"{condition}.jsonl"
        rows = read_jsonl(filename) if filename.exists() else []
        pairs = {(str(row["question_id"]), row["format"]) for row in rows}
        check(
            f"{condition}: qualitative responses", len(rows) == len(pairs) == 24, f"{len(rows)}/24"
        )
        if condition != "baseline":
            diagnostic_path = paths.evidence / "diagnostics" / f"{condition}.json"
            diagnostic = read_json(diagnostic_path) if diagnostic_path.exists() else {}
            check(
                f"{condition}: FP32 diagnostics",
                diagnostic.get("dtype") == "float32"
                and [row["layer"] for row in diagnostic.get("layers", [])] == [0, 13, 27]
                and all(
                    sum(row["layer"] == layer for row in diagnostic.get("per_probe", [])) == 64
                    for layer in (0, 13, 27)
                ),
            )
        method, _ = CONDITIONS[condition]
        if method and METHODS[method].rank:
            filename = directory / "merge_check.json"
            check(
                f"{condition}: merge verification",
                filename.exists() and read_json(filename)["passed"],
            )
    for method in METHODS:
        directory = paths.evidence / "runs" / method
        log = directory / "training.jsonl"
        rows = read_jsonl(log) if log.exists() else []
        check(
            f"{method}: training curve",
            [row["update"] for row in rows] == list(range(1, 1251)),
            f"{len(rows)}/1250",
        )
        resources = directory / "resources.json"
        successful = read_json(resources).get("successful_updates", 0) if resources.exists() else 0
        check(
            f"{method}: applied optimizer updates",
            successful == 1250,
            f"{successful}/1250; legacy runs that advanced on AMP skips require restarting",
        )
        for update in (312, 625, 937, 1250):
            check(
                f"{method}: recovery checkpoint {update}",
                checkpoint_available(paths, method, update),
            )
        check(f"{method}: resource measurements", (directory / "resources.json").exists())
    filename = paths.evidence / "qualitative" / "annotations.csv"
    annotated = []
    if filename.exists():
        with filename.open(newline="", encoding="utf-8") as stream:
            annotated = list(csv.DictReader(stream))
    check(
        "42 human annotations",
        len(annotated) == 42
        and all(
            all(row.get(field, "").strip() for field in ANNOTATION_FIELDS) for row in annotated
        ),
        "Fill all annotation fields; write 'none' where no excerpt/issue applies.",
    )
    for filename in ("benchmarks.csv", "resources.csv", "regularization.csv", "diagnostics.csv"):
        check(f"summary table: {filename}", (paths.evidence / "tables" / filename).exists())
    report = {"complete": all(row["complete"] for row in checks), "checks": checks}
    write_json(paths.evidence / "completeness.json", report)
    for row in checks:
        print(
            f"{'OK' if row['complete'] else 'MISSING'} {row['requirement']} {row['detail']}",
            flush=True,
        )
    return report["complete"]
