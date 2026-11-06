"""Verified private Hub backups and lazy restore for Kaggle's limited saved disk."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import tempfile
import time
from pathlib import Path, PurePosixPath

from huggingface_hub import HfApi, hf_hub_download

from atml.common.io import read_json, sha256, write_json
from atml.task1.config import METHODS, RECIPE

SHARED_PATTERNS = ["prepared/**", "tokenizer/**", "bases.pt", "baseline_activations.pt"]


def file_record(path):
    size = path.stat().st_size
    digest = hashlib.sha256()
    git_digest = hashlib.sha1(f"blob {size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            git_digest.update(block)
    return {"bytes": size, "sha256": digest.hexdigest(), "git_blob": git_digest.hexdigest()}


def inventory(root):
    return {
        p.relative_to(root).as_posix(): file_record(p)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def archive_path(paths, method, update):
    return paths.evidence / "storage" / "checkpoints" / method / f"update_{update:04d}.json"


def checkpoint_available(paths, method, update):
    if (paths.checkpoint(method, update) / "recovery.pt").exists():
        return True
    record = archive_path(paths, method, update)
    return record.exists() and read_json(record).get("verified", False)


def latest_checkpoint(paths, method):
    available = [u for u in RECIPE.recovery_updates if checkpoint_available(paths, method, u)]
    return available[-1] if available else 0


def private_api(repo):
    api = HfApi()  # HF_TOKEN or an existing authenticated Hub login; never log tokens.
    api.create_repo(repo_id=repo, repo_type="dataset", private=True, exist_ok=True)
    if not api.repo_info(repo, repo_type="dataset").private:
        raise ValueError("Checkpoint backup requires a private dataset repository.")
    return api


def verify_remote(api, repo, revision, prefix, files):
    siblings = api.repo_info(
        repo, repo_type="dataset", revision=revision, files_metadata=True
    ).siblings
    remote = {row.rfilename: row for row in siblings}
    for name, expected in files.items():
        key = f"{prefix}/{name}"
        row = remote.get(key)
        if row is None or row.size != expected["bytes"]:
            raise RuntimeError(f"Remote backup is missing or has wrong size: {key}")
        lfs = row.lfs
        if lfs:
            actual = lfs.get("sha256") if isinstance(lfs, dict) else lfs.sha256
            valid = actual == expected["sha256"]
        else:
            valid = row.blob_id == expected["git_blob"]
        if not valid:
            raise RuntimeError(f"Remote backup hash mismatch: {key}")


def upload_json(api, repo, name, value):
    return api.upload_file(
        repo_id=repo,
        repo_type="dataset",
        path_in_repo=name,
        path_or_fileobj=io.BytesIO(json.dumps(value, indent=2).encode()),
        commit_message=f"Save {name}",
    )


def archive_checkpoint(paths, method, update, repo, offload=False):
    """A failed upload/verification leaves every local checkpoint file intact."""
    target = paths.checkpoint(method, update)
    record_path = archive_path(paths, method, update)
    if not (target / "recovery.pt").exists():
        if record_path.exists() and read_json(record_path).get("verified"):
            record = read_json(record_path)
            if record["repo"] != repo:
                raise ValueError("Use the same backup repository for this experiment.")
            if offload and target.exists():
                # Evaluation restores only inference files, not optimizer/RNG
                # payloads. Release such partial copies against the full receipt.
                if any(
                    record["files"].get(name) != item for name, item in inventory(target).items()
                ):
                    raise ValueError("Local inference copy differs from its archived checkpoint.")
                api = private_api(repo)
                verify_remote(api, repo, record["revision"], record["prefix"], record["files"])
                sync_evidence(paths, api, repo)
                shutil.rmtree(target)
            return
        raise FileNotFoundError(target)
    started = time.perf_counter()
    api = private_api(repo)
    files = inventory(target)
    prefix = f"checkpoints/{method}/update_{update:04d}"
    prior = read_json(record_path) if record_path.exists() else {}
    if prior.get("files") == files and prior.get("repo") == repo and prior.get("verified"):
        verify_remote(api, repo, prior["revision"], prefix, files)
        record = prior
    else:
        print(
            f"[backup] Uploading {method} update {update}: {sum(r['bytes'] for r in files.values()) / 1e9:.2f} GB",
            flush=True,
        )
        commit = api.upload_folder(
            repo_id=repo,
            repo_type="dataset",
            folder_path=str(target),
            path_in_repo=prefix,
            commit_message=f"Checkpoint {method} {update}",
        )
        verify_remote(api, repo, commit.oid, prefix, files)
        record = {
            "repo": repo,
            "revision": commit.oid,
            "prefix": prefix,
            "files": files,
            "verified": True,
            "upload_and_verification_seconds": time.perf_counter() - started,
        }
        write_json(record_path, record)
    if offload:
        # The committed bundle must include this checkpoint's curve, resources,
        # and receipt together, otherwise a later resume could lose log rows.
        sync_evidence(paths, api, repo)
        shutil.rmtree(target)
        print(
            f"[backup] Verified and released local checkpoint: {method} update {update}", flush=True
        )


def _safe_relative(name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts or "\\" in name:
        raise ValueError(f"Unsafe archive path: {name}")
    return Path(*relative.parts)


def download_files(repo, revision, prefix, files, destination):
    with tempfile.TemporaryDirectory(prefix="pa3-hub-download-") as temporary:
        for name, expected in files.items():
            relative = _safe_relative(name)
            target = destination / relative
            if target.exists():
                if sha256(target) != expected["sha256"]:
                    raise FileExistsError(f"Existing file differs from backup: {target}")
                continue
            source = Path(
                hf_hub_download(
                    repo_id=repo,
                    repo_type="dataset",
                    revision=revision,
                    filename=f"{prefix}/{name}",
                    local_dir=temporary,
                )
            )
            if source.stat().st_size != expected["bytes"] or sha256(source) != expected["sha256"]:
                raise RuntimeError(f"Downloaded backup failed verification: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary_target = target.with_name(target.name + ".download_incomplete")
            shutil.copyfile(source, temporary_target)
            if sha256(temporary_target) != expected["sha256"]:
                raise RuntimeError(f"Restored file failed verification: {target}")
            temporary_target.replace(target)


def ensure_checkpoint(paths, method, update, needed=None):
    target = paths.checkpoint(method, update)
    required = needed or [
        "recovery.pt",
        "metadata.json",
        "adapter.pt" if METHODS[method].rank else "model.pt",
    ]
    if all((target / name).exists() for name in required):
        return target
    record = read_json(archive_path(paths, method, update))
    if not record.get("verified"):
        raise ValueError("Checkpoint has no verified archive receipt.")
    files = {name: record["files"][name] for name in needed} if needed else record["files"]
    print(
        f"[restore] {method} update {update} from {record['repo']}@{record['revision']}", flush=True
    )
    download_files(record["repo"], record["revision"], record["prefix"], files, target)
    return target


def backup(paths, repo, offload=False):
    api = private_api(repo)
    if not (paths.evidence / "storage" / "shared.json").exists():
        existing = api.repo_info(repo, repo_type="dataset").siblings
        if any(row.rfilename == "bundle.json" for row in existing):
            raise FileExistsError(
                "This repository already has an experiment. Restore it first or use a new repository."
            )
    files = {}
    for root in (paths.prepared, paths.artifacts / "tokenizer"):
        if root.exists():
            files.update({f"{root.name}/{name}": value for name, value in inventory(root).items()})
    for name in ("bases.pt", "baseline_activations.pt"):
        if (paths.artifacts / name).exists():
            files[name] = file_record(paths.artifacts / name)
    shared_path = paths.evidence / "storage" / "shared.json"
    prior = read_json(shared_path) if shared_path.exists() else {}
    if prior.get("repo") and prior["repo"] != repo:
        raise ValueError("Use the same backup repository for this experiment.")
    if prior.get("files") != files or not prior.get("verified"):
        if files:
            commit = api.upload_folder(
                repo_id=repo,
                repo_type="dataset",
                folder_path=str(paths.artifacts),
                path_in_repo="shared",
                allow_patterns=SHARED_PATTERNS,
                commit_message="Shared fixed Task 1 artifacts",
            )
            revision = commit.oid
            verify_remote(api, repo, revision, "shared", files)
        else:
            revision = api.repo_info(repo, repo_type="dataset").sha
        write_json(
            shared_path, {"repo": repo, "revision": revision, "files": files, "verified": True}
        )
    for method in METHODS:
        for update in RECIPE.recovery_updates:
            if paths.checkpoint(method, update).exists():
                archive_checkpoint(paths, method, update, repo, offload)
    sync_evidence(paths, api, repo)
    print(f"[backup] Evidence and shared artifacts verified in private dataset {repo}", flush=True)


def sync_evidence(paths, api, repo):
    shared_path = paths.evidence / "storage" / "shared.json"
    if not shared_path.exists():
        raise FileNotFoundError(
            "Run backup --repo first to preserve shared data and tokenization before offloading."
        )
    shared = read_json(shared_path)
    if shared["repo"] != repo or not shared["verified"]:
        raise ValueError("Shared backup repository mismatch or incomplete backup.")
    evidence_files = inventory(paths.evidence)
    commit = api.upload_folder(
        repo_id=repo,
        repo_type="dataset",
        folder_path=str(paths.evidence),
        path_in_repo="evidence",
        commit_message="Task 1 evidence and archive receipts",
    )
    verify_remote(api, repo, commit.oid, "evidence", evidence_files)
    upload_json(
        api,
        repo,
        "bundle.json",
        {"revision": commit.oid, "evidence_files": evidence_files, "shared": shared},
    )


def restore(paths, repo):
    api = HfApi()
    info = api.repo_info(repo, repo_type="dataset")
    if not info.private:
        raise ValueError("Restore only your own trusted private checkpoint dataset.")
    manifest = Path(
        hf_hub_download(
            repo_id=repo, repo_type="dataset", filename="bundle.json", revision=info.sha
        )
    )
    bundle = read_json(manifest)
    download_files(repo, bundle["revision"], "evidence", bundle["evidence_files"], paths.evidence)
    shared = bundle["shared"]
    download_files(repo, shared["revision"], "shared", shared["files"], paths.artifacts)
    print(
        "[restore] Shared artifacts/evidence restored; model checkpoints download only when needed.",
        flush=True,
    )
