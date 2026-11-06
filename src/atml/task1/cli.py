"""Small command-line entry point; importing it does not load a model."""

from __future__ import annotations

import argparse
from pathlib import Path

from atml.task1.config import CONDITIONS, METHODS, Paths, protocol


def main():
    parser = argparse.ArgumentParser(description="ATML PA3 Task 1 (fixed manual protocol)")
    parser.add_argument(
        "--artifacts", type=Path, default=Path("artifacts/task1"), help="Persistent large artifacts"
    )
    parser.add_argument(
        "--evidence", type=Path, default=Path("evidence/task1"), help="Persistent logs and results"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("protocol", help="Print the fixed recipe without downloading anything")
    smoke_test = commands.add_parser(
        "smoke", help="Offline tiny-model checks; no assignment experiments"
    )
    smoke_test.add_argument(
        "--cuda", action="store_true", help="Also check tiny FP16 CUDA generation"
    )
    commands.add_parser(
        "prepare", help="Freeze revisions, select training data, and save fixed probe IDs"
    )
    questions = commands.add_parser(
        "import-prompts", help="Import the supplied qualitative YAML verbatim"
    )
    questions.add_argument("source", type=Path)
    questions.add_argument("--prompt-map", type=Path)
    commands.add_parser(
        "fetch-prompts", help="Download the exact question YAML linked by the manual"
    )
    commands.add_parser("bases", help="Compute and persist shared FP32 starting-weight SVD bases")
    commands.add_parser("cache-probes", help="Cache FP32 baseline prompt activations once")
    training = commands.add_parser("train", help="Train one of the five independent variants")
    training.add_argument("--method", required=True, choices=METHODS)
    recovery = training.add_mutually_exclusive_group()
    recovery.add_argument("--resume", type=Path, help="A complete update_NNNN checkpoint directory")
    recovery.add_argument(
        "--resume-latest", action="store_true", help="Restore latest local/archived checkpoint"
    )
    training.add_argument(
        "--stop-after",
        choices=["next", "312", "625", "937", "1250"],
        help="Stop at a required recovery boundary; next is useful for Kaggle sessions",
    )
    training.add_argument(
        "--checkpoint-repo", help="Private Hub dataset: verify backup and offload each checkpoint"
    )
    for command in ("evaluate", "diagnose", "qualitative"):
        item = commands.add_parser(command)
        item.add_argument("--condition", choices=[*CONDITIONS, "all"], default="all")
        if command == "evaluate":
            from atml.task1.config import BENCHMARKS

            item.add_argument("--benchmark", choices=[*BENCHMARKS, "all"], default="all")
    commands.add_parser(
        "doctor", help="Check GPU, installed dependencies, writable paths, and disk space"
    )
    memory = commands.add_parser(
        "memory-check", help="One synthetic maximum-length step; discard weights afterward"
    )
    memory.add_argument("--method", choices=METHODS, required=True)
    for command in ("backup", "restore"):
        item = commands.add_parser(
            command, help="Private Hub checkpoint/shared-artifact/evidence storage"
        )
        item.add_argument("--repo", required=True)
        if command == "backup":
            item.add_argument(
                "--offload",
                action="store_true",
                help="Delete local checkpoint copies only after verified backup",
            )
    commands.add_parser(
        "summarize", help="Generate required CSV tables, plots, and annotation worksheet"
    )
    commands.add_parser("audit", help="Save evidence completeness audit; exit 1 when incomplete")
    commands.add_parser(
        "run", help="Run the entire Task 1 pipeline (requires CUDA and imported YAML)"
    )
    args = parser.parse_args()
    paths = Paths(args.artifacts.resolve(), args.evidence.resolve())
    if args.command == "protocol":
        import json

        print(json.dumps(protocol(), indent=2))
    elif args.command == "smoke":
        from atml.task1.smoke import smoke

        smoke(paths, args.cuda)
    elif args.command == "import-prompts":
        from atml.task1.data import import_questions

        import_questions(args.source, paths.evidence, args.prompt_map)
    elif args.command == "fetch-prompts":
        from atml.task1.data import fetch_questions

        fetch_questions(paths)
    elif args.command == "prepare":
        from atml.task1.data import prepare

        prepare(paths)
    elif args.command == "bases":
        from atml.task1.model import build_bases

        build_bases(paths)
    elif args.command == "cache-probes":
        from atml.task1.diagnostics import cache_probes

        cache_probes(paths)
    elif args.command == "train":
        from atml.task1.training import train

        train(
            paths,
            args.method,
            args.resume,
            args.stop_after,
            args.checkpoint_repo,
            args.resume_latest,
        )
    elif args.command in ("evaluate", "diagnose", "qualitative"):
        selected = list(CONDITIONS) if args.condition == "all" else [args.condition]
        for condition in selected:
            if args.command == "evaluate":
                from atml.task1.evaluation import evaluate

                evaluate(paths, condition, args.benchmark)
            elif args.command == "diagnose":
                from atml.task1.diagnostics import diagnose

                if condition != "baseline":
                    diagnose(paths, condition)
            else:
                from atml.task1.qualitative import qualitative

                qualitative(paths, condition)
    elif args.command == "summarize":
        from atml.task1.summary import summarize

        summarize(paths)
    elif args.command == "audit":
        from atml.task1.summary import audit

        raise SystemExit(0 if audit(paths) else 1)
    elif args.command == "doctor":
        from atml.task1.preflight import doctor

        raise SystemExit(0 if doctor(paths) else 1)
    elif args.command == "memory-check":
        from atml.task1.preflight import memory_check

        raise SystemExit(0 if memory_check(paths, args.method) else 1)
    elif args.command in ("backup", "restore"):
        from atml.task1.storage import backup, restore

        if args.command == "backup":
            backup(paths, args.repo, args.offload)
        else:
            restore(paths, args.repo)
    elif args.command == "run":
        run_all(paths)


def run_all(paths):
    from atml.common.io import require_cuda
    from atml.task1.data import prepare
    from atml.task1.diagnostics import cache_probes, diagnose
    from atml.task1.evaluation import evaluate
    from atml.task1.model import build_bases
    from atml.task1.qualitative import qualitative
    from atml.task1.summary import audit, summarize
    from atml.task1.training import train
    from atml.task1.storage import checkpoint_available

    require_cuda()
    if not (paths.evidence / "qualitative_prompts.json").exists():
        raise FileNotFoundError(
            "First import the supplied first_plot_questions.yaml with import-prompts."
        )
    if not (paths.evidence / "preparation.json").exists():
        prepare(paths)
    build_bases(paths)
    cache_probes(paths)  # Fixed before any model scores are inspected.
    evaluate(paths, "baseline")
    qualitative(paths, "baseline")
    for method in METHODS:
        if not checkpoint_available(paths, method, 1250):
            train(paths, method, resume_latest=True)
        conditions = (
            ["full_ft_midpoint", "full_ft_final"] if method == "full_ft" else [f"{method}_final"]
        )
        for condition in conditions:
            evaluate(paths, condition)
            diagnose(paths, condition)
            qualitative(paths, condition)
    summarize(paths)
    audit(paths)  # Human annotation remains explicitly pending, never fabricated.
    print(
        "Experiment pipeline finished. Fill qualitative/annotations.csv, then run audit again.",
        flush=True,
    )


if __name__ == "__main__":
    main()
