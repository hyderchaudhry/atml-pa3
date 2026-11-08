"""CPU analysis and the separate prescribed cloud-GPU live experiment."""

import argparse
import json
import os
from pathlib import Path

from filelock import FileLock

from atml.common.io import environment, require_cuda, write_json

from .config import DATASETS, Paths, protocol


def next_step(paths, offline_only=False):
    if not (paths.evidence / "preparation.json").exists():
        return "prepare"
    for dataset in DATASETS:
        if not (paths.evidence / "offline" / dataset / "development/complete.json").exists():
            return "development"
    if not (paths.evidence / "policies.json").exists():
        return "fit"
    for dataset in DATASETS:
        if not (paths.evidence / "offline" / dataset / "evaluation/complete.json").exists():
            return "evaluation"
    if not offline_only and not (paths.evidence / "live/complete.json").exists():
        return "live"
    return "summarize"


def dispatch(paths, command, source=None, metadata=None, workers=4, checkpoint_callback=None):
    if command == "prepare":
        from .data import prepare

        prepare(paths, source, metadata)
    elif command in ("development", "evaluation"):
        from .offline import run_offline

        run_offline(paths, command, workers=workers, checkpoint_callback=checkpoint_callback)
    elif command == "fit":
        from .offline import fit_policies

        fit_policies(paths)
    elif command == "live":
        from .live import run_live

        run_live(paths, checkpoint_callback)
    elif command == "summarize":
        from .summary import export, summarize

        print(json.dumps(summarize(paths), indent=2))
        print(f"[download] {export(paths)} (extract task3/ into local evidence/)")
    else:
        raise ValueError(command)


def run(paths, args, offline_only=False):
    from huggingface_hub import HfApi
    from .storage import backup, restore

    if not offline_only:
        require_cuda()
    if not args.repo:
        raise ValueError("Set PA3_TASK3_BACKUP_REPO=your-user/atml-pa3-task3 or pass --repo")
    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)
    if not api.repo_info(args.repo, repo_type="dataset").private:
        raise ValueError("Use a private recovery dataset")
    if not (
        paths.evidence / "preparation.json"
    ).exists() and "task3-bundle.json" in api.list_repo_files(args.repo, repo_type="dataset"):
        restore(paths, args.repo)

    def checkpoint():
        backup(paths, args.repo)

    while True:
        step = next_step(paths, offline_only)
        print(f"[next] {step}", flush=True)
        dispatch(paths, step, args.source, args.metadata, args.workers, checkpoint)
        checkpoint()
        if step == "summarize":
            break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("PA3_ROOT", ".")))
    parser.add_argument("--repo", default=os.environ.get("PA3_TASK3_BACKUP_REPO"))
    parser.add_argument("--source", type=Path, default=os.environ.get("PA3_TASK3_DATA"))
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument(
        "command",
        choices=[
            "protocol",
            "doctor",
            "auth",
            "smoke",
            "run",
            "analyze",
            "next",
            "fetch-data",
            "prepare",
            "development",
            "fit",
            "evaluation",
            "live",
            "summarize",
            "audit",
            "export",
            "backup",
            "restore",
        ],
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    paths = Paths(args.root.expanduser().resolve())
    if args.command == "protocol":
        print(json.dumps(protocol(), indent=2))
        return
    if args.command == "smoke":
        from .smoke import smoke

        smoke()
        return
    if args.command == "auth":
        import getpass
        from huggingface_hub import login

        login(
            token=getpass.getpass("Hugging Face Write token (hidden): "),
            add_to_git_credential=False,
        )
        return
    paths.artifacts.mkdir(parents=True, exist_ok=True)
    with FileLock(str(paths.artifacts / ".workflow.lock"), timeout=0):
        if args.command in ("run", "analyze"):
            run(paths, args, offline_only=args.command == "analyze")
        elif args.command == "next":
            print(next_step(paths))
        elif args.command == "doctor":
            import torch

            require_cuda()
            info = environment()
            info["gpu_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
            info["persistent_root"] = str(paths.root)
            write_json(paths.evidence / "doctor.json", info)
            print(json.dumps(info, indent=2))
        elif args.command == "fetch-data":
            from .data import fetch

            print(fetch(paths))
        elif args.command in ("backup", "restore"):
            if not args.repo:
                parser.error("Set PA3_TASK3_BACKUP_REPO or pass --repo")
            from . import storage

            getattr(storage, args.command)(paths, args.repo)
        elif args.command == "audit":
            from .summary import audit

            print(json.dumps(audit(paths), indent=2))
        elif args.command == "export":
            from .summary import export

            print(export(paths))
        else:
            dispatch(paths, args.command, args.source, args.metadata, args.workers)


if __name__ == "__main__":
    main()
