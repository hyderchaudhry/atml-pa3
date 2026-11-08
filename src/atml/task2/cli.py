"""Provider-independent persistent-disk workflow for the prescribed main run."""

import argparse
import json
import os
from pathlib import Path

from filelock import FileLock

from atml.common.io import environment, read_json, require_cuda, write_json

from .config import DEV_UPDATES, Paths, protocol


def next_step(paths):
    from .training import latest

    if not paths.manifest.exists():
        return "prepare", None
    index = paths.artifacts / "features/index.json"
    if not index.exists() or not read_json(index).get("complete"):
        return "features", None
    update = latest(paths)
    if update is None:
        return "initialize", None
    for due in DEV_UPDATES:
        if (
            due <= update
            and not (paths.evidence / f"development/update_{due:04d}/complete.json").exists()
        ):
            return "develop", due
    if update < 4500:
        return "train", update + 250
    if not (paths.evidence / "final/complete.json").exists():
        return "evaluate", 4500
    return "summarize", None


def dispatch(paths, command, update=None):
    if command == "prepare":
        from .data import prepare

        prepare(paths)
    elif command == "features":
        from .features import cache_features

        cache_features(paths)
        from .summary import worksheets

        worksheets(paths)
    elif command == "initialize":
        from .training import initialize

        initialize(paths)
    elif command == "train":
        from .training import train_segment

        train_segment(paths, update)
    elif command in ("develop", "evaluate"):
        from .evaluation import evaluate

        evaluate(paths, update, final=command == "evaluate")
    elif command == "summarize":
        from .summary import ensure_worksheet_images, export, summarize

        ensure_worksheet_images(paths)
        result = summarize(paths)
        print(json.dumps(result, indent=2))
        print(f"[download] {export(paths)} (extract task2/ into local evidence/)")
    else:
        raise ValueError(command)


def run(paths, repo):
    from huggingface_hub import HfApi
    from .storage import backup, restore

    require_cuda()
    if not repo:
        raise ValueError("Set PA3_TASK2_BACKUP_REPO=your-user/atml-pa3-task2 or pass --repo")
    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    if not api.repo_info(repo, repo_type="dataset").private:
        raise ValueError("Use a private dataset for recovery backups")
    if not paths.manifest.exists() and "task2-bundle.json" in api.list_repo_files(
        repo, repo_type="dataset"
    ):
        restore(paths, repo)
    while True:
        command, update = next_step(paths)
        print(f"[next] {command}" + (f" {update}" if update is not None else ""), flush=True)
        dispatch(paths, command, update)
        backup(paths, repo)
        if command == "summarize":
            break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("PA3_ROOT", ".")))
    parser.add_argument("--repo", default=os.environ.get("PA3_TASK2_BACKUP_REPO"))
    parser.add_argument(
        "command",
        choices=[
            "protocol",
            "doctor",
            "auth",
            "smoke",
            "run",
            "next",
            "prepare",
            "features",
            "initialize",
            "train",
            "develop",
            "evaluate",
            "summarize",
            "audit",
            "backup",
            "restore",
            "export",
            "images",
        ],
    )
    parser.add_argument("--update", type=int)
    args = parser.parse_args()
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
        if args.command == "doctor":
            import torch

            require_cuda()
            info = environment()
            info["gpu_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
            info["persistent_root"] = str(paths.root)
            write_json(paths.evidence / "doctor.json", info)
            print(json.dumps(info, indent=2))
        elif args.command == "run":
            run(paths, args.repo)
        elif args.command == "next":
            print(next_step(paths))
        elif args.command in ("backup", "restore"):
            if not args.repo:
                parser.error("--repo or PA3_TASK2_BACKUP_REPO is required")
            from . import storage

            getattr(storage, args.command)(paths, args.repo)
        elif args.command == "audit":
            from .summary import audit

            print(json.dumps(audit(paths), indent=2))
        elif args.command == "export":
            from .summary import export

            print(export(paths))
        elif args.command == "images":
            from .data import fetch_images
            from .summary import worksheets

            fetch_images(paths, read_json(paths.manifest))
            worksheets(paths, include_predictions=True)
        else:
            update = args.update
            if args.command == "develop" and update is None:
                parser.error("develop requires --update (a prescribed development checkpoint)")
            if args.command == "evaluate":
                update = 4500 if update is None else update
            dispatch(paths, args.command, update)


if __name__ == "__main__":
    main()
