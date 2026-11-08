"""Incremental verified Task 3 backups, including immutable prepared source cache."""

import io
import json
import shutil
from pathlib import Path, PurePosixPath

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

from atml.common.io import read_json, sha256, write_json
from atml.task2.storage import record


def safe_relative(name):
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or ".." in path.parts
        or "\\" in name
        or path.parts[:2] not in (("artifacts", "task3"), ("evidence", "task3"))
    ):
        raise ValueError("Unsafe or non-Task-3 recovery path")
    return Path(*path.parts)


def backup(paths, repo):
    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    if not api.repo_info(repo, repo_type="dataset").private:
        raise ValueError("Recovery repository must be a private dataset")
    receipt = paths.artifacts / "backup_receipt.json"
    prior = read_json(receipt) if receipt.exists() else dict(repo=repo, files={})
    if prior["repo"] != repo:
        raise ValueError("Recovery repository changed")
    candidates = list(paths.evidence.rglob("*")) + list((paths.artifacts / "metadata").rglob("*"))
    db = paths.artifacts / "prepared.sqlite"
    if (paths.evidence / "preparation.json").exists():
        candidates.append(db)
    files, changed = {}, []
    for file in sorted(candidates):
        if (
            not file.is_file()
            or file.suffix in (".tmp", ".part", ".crash_tail")
            or file.name.startswith(".")
        ):
            continue
        name = file.relative_to(paths.root).as_posix()
        stat = file.stat()
        previous = prior["files"].get(name, {})
        # Prepared/source/result files are immutable after an atomic commit.
        # Cache their local hash by size+nanosecond mtime; restore always rehashes.
        if previous.get("mtime_ns") == stat.st_mtime_ns and previous.get("bytes") == stat.st_size:
            files[name] = previous
            continue
        item = record(file)
        item["mtime_ns"] = stat.st_mtime_ns
        if item["sha256"] == previous.get("sha256"):
            item["revision"] = previous["revision"]
        else:
            changed.append((name, file))
        files[name] = item
    for start in range(0, len(changed), 32):
        batch = changed[start : start + 32]
        commit = api.create_commit(
            repo,
            repo_type="dataset",
            commit_message="Task 3 durable progress",
            operations=[
                CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(file))
                for name, file in batch
            ],
        )
        remote = {
            r.rfilename: r
            for r in api.repo_info(
                repo, repo_type="dataset", revision=commit.oid, files_metadata=True
            ).siblings
        }
        for name, _ in batch:
            expected, actual = files[name], remote[name]
            lfs = actual.lfs
            value = (
                (lfs.get("sha256") if isinstance(lfs, dict) else lfs.sha256)
                if lfs
                else actual.blob_id
            )
            if (
                actual.size != expected["bytes"]
                or value != expected["sha256" if lfs else "git_blob"]
            ):
                raise IOError(f"Remote checksum mismatch: {name}")
            expected["revision"] = commit.oid
    bundle = dict(schema=1, repo=repo, files=files)
    commit = api.upload_file(
        repo_id=repo,
        repo_type="dataset",
        path_in_repo="task3-bundle.json",
        path_or_fileobj=io.BytesIO(json.dumps(bundle, indent=2).encode()),
        commit_message="Publish verified Task 3 recovery inventory",
    )
    bundle["bundle_revision"] = commit.oid
    write_json(receipt, bundle)
    print(f"[backup] {len(files)} Task 3 files verified in {repo}@{commit.oid}", flush=True)


def restore(paths, repo):
    bundle = read_json(Path(hf_hub_download(repo, "task3-bundle.json", repo_type="dataset")))
    if bundle["repo"] != repo:
        raise ValueError("Recovery inventory repository mismatch")
    for name, item in bundle["files"].items():
        target = paths.root / safe_relative(name)
        if target.exists():
            if sha256(target) != item["sha256"]:
                raise FileExistsError(f"Local file conflicts with backup: {target}")
            continue
        source = Path(hf_hub_download(repo, name, repo_type="dataset", revision=item["revision"]))
        if source.stat().st_size != item["bytes"] or sha256(source) != item["sha256"]:
            raise IOError("Recovery download checksum mismatch")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        shutil.copyfile(source, temporary)
        temporary.replace(target)
    write_json(paths.artifacts / "backup_receipt.json", bundle)
    print(
        "[restore] Task 3 prepared source, results and live-generation progress restored",
        flush=True,
    )
