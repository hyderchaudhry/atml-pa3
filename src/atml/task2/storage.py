"""Incremental, hash-verified private Hub backup; no local checkpoint deletion."""

import hashlib
import io
import json
import shutil
from pathlib import Path, PurePosixPath

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

from atml.common.io import read_json, sha256, write_json


def record(path):
    size = path.stat().st_size
    blob = hashlib.sha1(f"blob {size}\0".encode())
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            blob.update(block)
            digest.update(block)
    return dict(bytes=size, sha256=digest.hexdigest(), git_blob=blob.hexdigest())


def safe_relative(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"Unsafe backup path: {name}")
    if path.parts[:2] not in (("artifacts", "task2"), ("evidence", "task2")):
        raise ValueError("Backup must be scoped to Task 2")
    return Path(*path.parts)


def backup(paths, repo):
    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    if not api.repo_info(repo, repo_type="dataset").private:
        raise ValueError("Task 2 backup must be a private dataset")
    receipt_path = paths.artifacts / "backup_receipt.json"
    prior = read_json(receipt_path) if receipt_path.exists() else dict(files={})
    if prior.get("repo", repo) != repo:
        raise ValueError("Backup repository changed")
    files, changed = {}, []
    candidates = list(paths.evidence.rglob("*"))
    for folder in ("tokenizer", "features", "checkpoints"):
        candidates.extend((paths.artifacts / folder).rglob("*"))
    for file in sorted(candidates):
        if not file.is_file() or any(
            part.startswith(".") for part in file.relative_to(paths.root).parts
        ):
            continue
        if file.suffix in {".tmp", ".part", ".crash_tail"}:
            continue
        name = file.relative_to(paths.root).as_posix()
        item = record(file)
        if prior["files"].get(name, {}).get("sha256") == item["sha256"]:
            files[name] = prior["files"][name]
        else:
            files[name] = item
            changed.append((name, file))
    for start in range(0, len(changed), 32):
        batch = changed[start : start + 32]
        commit = api.create_commit(
            repo,
            repo_type="dataset",
            commit_message="Task 2 recovery files",
            operations=[
                CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(p)) for name, p in batch
            ],
        )
        remote = {
            s.rfilename: s
            for s in api.repo_info(
                repo, repo_type="dataset", revision=commit.oid, files_metadata=True
            ).siblings
        }
        for name, _ in batch:
            expected, actual = files[name], remote[name]
            lfs = actual.lfs
            digest = (
                (lfs.get("sha256") if isinstance(lfs, dict) else lfs.sha256)
                if lfs
                else actual.blob_id
            )
            if (
                actual.size != expected["bytes"]
                or digest != expected["sha256" if lfs else "git_blob"]
            ):
                raise IOError(f"Remote verification failed: {name}")
            expected["revision"] = commit.oid
    # Publish the inventory only after every referenced immutable commit was verified.
    bundle = dict(schema=1, repo=repo, files=files)
    commit = api.upload_file(
        repo_id=repo,
        repo_type="dataset",
        path_in_repo="task2-bundle.json",
        path_or_fileobj=io.BytesIO(json.dumps(bundle, indent=2).encode()),
        commit_message="Publish verified Task 2 recovery inventory",
    )
    bundle["bundle_revision"] = commit.oid
    write_json(receipt_path, bundle)
    print(f"[backup] {len(files)} files verified in {repo}@{commit.oid}", flush=True)


def restore(paths, repo):
    """Restore a fresh workspace; refuse conflicting local experiment files."""
    bundle_file = hf_hub_download(repo, "task2-bundle.json", repo_type="dataset")
    bundle = read_json(Path(bundle_file))
    if bundle["repo"] != repo:
        raise ValueError("Backup inventory repository mismatch")
    for name, item in bundle["files"].items():
        target = paths.root / safe_relative(name)
        if target.exists():
            if sha256(target) != item["sha256"]:
                raise FileExistsError(f"Local file conflicts with backup: {target}")
            continue
        source = Path(hf_hub_download(repo, name, repo_type="dataset", revision=item["revision"]))
        if source.stat().st_size != item["bytes"] or sha256(source) != item["sha256"]:
            raise IOError(f"Downloaded backup checksum mismatch: {name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        shutil.copyfile(source, tmp)
        tmp.replace(target)
    write_json(paths.artifacts / "backup_receipt.json", bundle)
    print("[restore] Task 2 evidence, feature cache and recovery checkpoints restored", flush=True)
