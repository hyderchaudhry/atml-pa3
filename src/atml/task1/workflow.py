"""Choose the next fixed Task 1 step so Kaggle users need only one run command."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from atml.task1.config import BENCHMARKS, CONDITIONS, METHODS, RECIPE, Paths


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Step:
    label: str
    arguments: tuple[str, ...]


def latest_update(paths, method):
    updates = []
    for update in RECIPE.recovery_updates:
        local = paths.checkpoint(method, update) / "recovery.pt"
        receipt = paths.evidence / "storage" / "checkpoints" / method / f"update_{update:04d}.json"
        if local.exists() or (receipt.exists() and read(receipt).get("verified")):
            updates.append(update)
    return max(updates, default=0)


def condition_step(paths, condition):
    directory = paths.evidence / "evaluations" / condition
    for task in BENCHMARKS:
        if not (directory / f"{task}_results.json").exists():
            return Step(
                f"Evaluate {condition}: {task}",
                ("evaluate", "--condition", condition, "--benchmark", task),
            )
    if not (directory / "summary.json").exists():
        return Step(
            f"Finalize benchmark scores: {condition}", ("evaluate", "--condition", condition)
        )
    if condition != "baseline" and not (paths.evidence / "diagnostics" / f"{condition}.json").exists():
        return Step(f"Save diagnostics: {condition}", ("diagnose", "--condition", condition))
    generated = paths.evidence / "qualitative" / f"{condition}.jsonl"
    rows = []
    if generated.exists():
        rows = [json.loads(line) for line in generated.read_text().splitlines() if line.strip()]
    if len(rows) != 24 or len({(str(row["question_id"]), row["format"]) for row in rows}) != 24:
        return Step(
            f"Generate qualitative responses: {condition}",
            ("qualitative", "--condition", condition),
        )
    return None


def next_step(paths, repo, questions=None, prompt_map=None):
    # If an earlier upload failed, preserve that completed checkpoint before
    # starting another expensive step. This also frees inference-only copies.
    for method in METHODS:
        for update in RECIPE.recovery_updates:
            if paths.checkpoint(method, update).exists():
                return Step(
                    "Back up pending checkpoint files", ("backup", "--repo", repo, "--offload")
                )
    if not (paths.evidence / "qualitative_prompts.json").exists():
        if not questions:
            return Step("Download the question file linked by the manual", ("fetch-prompts",))
        if not Path(questions).is_file():
            raise FileNotFoundError(
                f"No question file at {questions}. Unset PA3_QUESTIONS to download the official file automatically."
            )
        arguments = ["import-prompts", str(questions)]
        if prompt_map:
            arguments.extend(["--prompt-map", str(prompt_map)])
        return Step("Import assignment questions", tuple(arguments))
    report = paths.evidence / "system" / "doctor.json"
    if not report.exists() or not read(report).get("passed"):
        return Step("Check the Kaggle environment", ("doctor",))
    if not (paths.evidence / "preparation.json").exists():
        return Step("Prepare the fixed training data and probe IDs", ("prepare",))
    if not (paths.evidence / "bases.json").exists():
        return Step("Compute the shared starting-weight bases", ("bases",))
    if not (paths.evidence / "activation_cache.json").exists():
        return Step("Cache baseline probe activations", ("cache-probes",))
    step = condition_step(paths, "baseline")
    if step:
        return step
    for method in METHODS:
        update = latest_update(paths, method)
        if method == "full_ft" and update >= 625:
            step = condition_step(paths, "full_ft_midpoint")
            if step:
                return step
        if update < RECIPE.updates:
            memory = paths.evidence / "system" / f"memory_{method}.json"
            if not memory.exists() or not read(memory).get("passed"):
                return Step(f"Check GPU memory: {method}", ("memory-check", "--method", method))
            boundary = next(u for u in RECIPE.recovery_updates if u > update)
            return Step(
                f"Train {method}: update {update} to {boundary}",
                ("train", "--method", method, "--resume-latest", "--stop-after", "next",
                 "--checkpoint-repo", repo),
            )
        step = condition_step(paths, f"{method}_final")
        if step:
            return step
    required = [
        paths.evidence / "tables" / name
        for name in ("benchmarks.csv", "resources.csv", "regularization.csv", "diagnostics.csv")
    ]
    required.extend(
        paths.evidence / "figures" / name
        for name in ("benchmarks.png", "learning_curves.png", "regularization.png")
    )
    required.extend([
        paths.evidence / "qualitative" / "annotations.csv",
        paths.evidence / "qualitative" / "grouped_responses.json",
    ])
    if not all(path.exists() for path in required):
        return Step("Create tables, plots, and the annotation worksheet", ("summarize",))
    return None


def execute(paths, *arguments, check=True):
    command = [
        sys.executable, "-m", "atml.task1", "--artifacts", str(paths.artifacts),
        "--evidence", str(paths.evidence), *arguments,
    ]
    return subprocess.run(command, check=check)


def restore_if_needed(paths, repo):
    from atml.task1.storage import download_files, private_api

    api = private_api(repo)  # Validate credentials before model downloads or training.
    shared = paths.evidence / "storage" / "shared.json"
    if not shared.exists():
        info = api.repo_info(repo, repo_type="dataset")
        if any(row.rfilename == "bundle.json" for row in info.siblings):
            print("[resume] Restoring saved progress from your private backup.", flush=True)
            execute(paths, "restore", "--repo", repo)
    else:
        receipt = read(shared)
        if receipt["repo"] != repo:
            raise ValueError("PA3_BACKUP_REPO differs from this experiment's saved repository.")
        # A resumed working directory may retain logs but need its shared files.
        download_files(repo, receipt["revision"], "shared", receipt["files"], paths.artifacts)


def export_evidence(paths):
    output = paths.evidence.parent.parent
    archive = shutil.make_archive(
        str(output / "task1-evidence"), "zip",
        root_dir=paths.evidence.parent, base_dir=paths.evidence.name,
    )
    print(
        f"[download] {archive} (extract its task1/ folder into your local evidence/ folder)",
        flush=True,
    )


def status(paths, repo, questions=None, prompt_map=None):
    for method in METHODS:
        print(f"{method:10s} {latest_update(paths, method):4d}/{RECIPE.updates} updates saved", flush=True)
    completed = sum(
        (paths.evidence / "evaluations" / condition / "summary.json").exists()
        for condition in CONDITIONS
    )
    print(f"Benchmark conditions complete: {completed}/{len(CONDITIONS)}", flush=True)
    try:
        step = next_step(paths, repo, questions, prompt_map)
    except FileNotFoundError as error:
        print(f"Next: {error}", flush=True)
        return
    label = step.label if step else "Experiments complete; fill the annotation worksheet."
    print(f"Next: {label}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "next", "status", "export"])
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    paths = Paths(args.artifacts, args.evidence)
    if args.command == "export":
        export_evidence(paths)
        return
    repo = os.environ["PA3_BACKUP_REPO"]
    questions = os.environ.get("PA3_QUESTIONS")
    prompt_map = os.environ.get("PA3_PROMPT_MAP")
    if args.command == "status":
        status(paths, repo, questions, prompt_map)
        return
    restore_if_needed(paths, repo)
    while True:
        step = next_step(paths, repo, questions, prompt_map)
        if step is None:
            audit_path = paths.evidence / "completeness.json"
            previous_time = audit_path.stat().st_mtime_ns if audit_path.exists() else None
            result = execute(paths, "audit", check=False)
            if (
                result.returncode not in (0, 1)
                or not audit_path.exists()
                or audit_path.stat().st_mtime_ns == previous_time
            ):
                raise RuntimeError("Could not audit the evidence; inspect the error above.")
            unfinished = [
                row["requirement"] for row in read(audit_path)["checks"]
                if not row["complete"] and row["requirement"] != "42 human annotations"
            ]
            execute(paths, "backup", "--repo", repo, "--offload")
            export_evidence(paths)
            if unfinished:
                raise RuntimeError(
                    f"The evidence audit still reports missing work: {', '.join(unfinished)}"
                )
            print(
                "[done] Experiment outputs are saved. "
                "Fill qualitative/annotations.csv yourself if still pending.", flush=True,
            )
            return
        print(f"\n[next] {step.label}", flush=True)
        execute(paths, *step.arguments)
        if step.arguments[0] != "backup":
            execute(paths, "backup", "--repo", repo, "--offload")
        if next_step(paths, repo, questions, prompt_map) == step:
            raise RuntimeError("The last step did not advance. Inspect its saved output before retrying.")
        if args.command == "next":
            print("[saved] Step complete. Run the same command again to continue.", flush=True)
            return


if __name__ == "__main__":
    main()
