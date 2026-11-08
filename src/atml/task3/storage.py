"""Incremental verified Task 3 backups, including immutable prepared source cache."""

import io
import json
import re
import shutil
import sys
import tarfile
import tempfile
import time
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit, urlunsplit

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from requests.exceptions import ConnectionError, Timeout

from atml.common.io import read_json, sha256, write_json
from atml.task2.storage import record


def failure_details(error):
    """Expose chained LFS/network errors without printing signed URL queries."""
    chain, seen = [], set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        chain.append(error)
        error = error.__cause__ or error.__context__
    status = next(
        (e.response.status_code for e in reversed(chain)
         if getattr(e, "response", None) is not None), None
    )
    root = chain[-1]
    response = next((getattr(e, "response", None) for e in reversed(chain)
                     if getattr(e, "response", None) is not None), None)
    headers = getattr(response, "headers", {})
    retry_after = None
    if status == 429:
        advertised = str(headers.get("Retry-After", ""))
        reset = re.search(r"\bt=(\d+)", headers.get("RateLimit", ""))
        retry_after = float(advertised) + 2 if advertised.isdigit() else int(reset[1]) + 2 if reset else 305

    def redact_url(match):
        parts = urlsplit(match[0])
        return urlunsplit((parts.scheme, parts.netloc, parts.path,
                           "<redacted>" if parts.query else "", ""))

    message = re.sub(r"https?://[^\s]+", redact_url, str(root))
    message = re.sub(r"\bhf_[A-Za-z0-9]{8,}\b", "<redacted-token>", message)
    transient = status in (408, 429, 500, 502, 503, 504) if status is not None else any(
        isinstance(e, (ConnectionError, Timeout)) for e in chain
    )
    return dict(type=type(root).__name__, status=status, message=message[:1000],
                transient=transient, retry_after=retry_after)


def retry_upload(label, operation, attempts=4):
    """Retry transient failures only; each attempt constructs fresh upload objects."""
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as error:
            details = failure_details(error)
            print(
                f"[backup] {label} failed ({attempt}/{attempts}): "
                f"{details['type']} HTTP={details['status']}: {details['message']}",
                file=sys.stderr, flush=True,
            )
            if not details["transient"] or attempt == attempts:
                raise
            delay = details["retry_after"] or 2 ** attempt
            print(f"[backup] Retrying in {delay}s; local results are retained", flush=True)
            while delay > 0:
                interval = min(delay, 30)
                time.sleep(interval)
                delay -= interval
                if delay > 0:
                    print(f"[backup] Rate-limit cooldown: {delay}s remaining", flush=True)


def upload_batch(api, repo, batch, files, staging):
    """One verified archive replaces dozens of individual small LFS uploads."""
    direct = [(name, file) for name, file in batch if files[name]["bytes"] > 1024 * 1024]
    small = [(name, file) for name, file in batch if files[name]["bytes"] <= 1024 * 1024]
    payloads = dict(direct)
    archive_name = None
    if small:
        archive = staging / "batch.tar"
        with tarfile.open(archive, "w", format=tarfile.PAX_FORMAT) as stream:
            for name, file in small:
                info = tarfile.TarInfo(name)
                info.size = files[name]["bytes"]
                info.mtime = 0
                info.mode = 0o600
                with file.open("rb") as source:
                    stream.addfile(info, source)
        archive_name = f"task3-archives/{sha256(archive)}.tar"
        payloads[archive_name] = archive
    expected_payloads = {name: record(file) for name, file in payloads.items()}
    commit = retry_upload(
        "upload payload batch",
        lambda: api.create_commit(
            repo, repo_type="dataset", commit_message="Task 3 durable progress",
            num_threads=1,
            operations=[CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(file))
                        for name, file in payloads.items()],
        ),
    )
    remote = {r.rfilename: r for r in api.repo_info(
        repo, repo_type="dataset", revision=commit.oid, files_metadata=True).siblings}
    for name, expected in expected_payloads.items():
        actual = remote[name]
        lfs = actual.lfs
        value = ((lfs.get("sha256") if isinstance(lfs, dict) else lfs.sha256)
                 if lfs else actual.blob_id)
        if actual.size != expected["bytes"] or value != expected["sha256" if lfs else "git_blob"]:
            raise IOError(f"Remote checksum mismatch: {name}")
    for name, _ in batch:
        files[name]["revision"] = commit.oid
    for name, _ in small:
        files[name]["archive_path"] = archive_name
        files[name]["archive_sha256"] = expected_payloads[archive_name]["sha256"]


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
    progress_path = paths.artifacts / "backup_progress.json"
    if progress_path.exists():
        progress = read_json(progress_path)
        if progress["repo"] != repo:
            raise ValueError("Partial backup recovery repository changed")
        prior["files"].update(progress["files"])
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
            for field in ("archive_path", "archive_sha256"):
                if field in previous:
                    item[field] = previous[field]
        else:
            changed.append((name, file))
        files[name] = item
    for start in range(0, len(changed), 32):
        batch = changed[start : start + 32]
        print(f"[backup] Uploading batch {start // 32 + 1}: {len(batch)} changed files", flush=True)
        paths.artifacts.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".backup-", dir=paths.artifacts) as temporary:
            upload_batch(api, repo, batch, files, Path(temporary))
        write_json(progress_path, dict(repo=repo, files={
            name: item for name, item in files.items() if "revision" in item
        }))
    bundle = dict(schema=2, repo=repo, files=files)
    commit = retry_upload(
        "publish recovery inventory",
        lambda: api.upload_file(
            repo_id=repo,
            repo_type="dataset",
            path_in_repo="task3-bundle.json",
            path_or_fileobj=io.BytesIO(json.dumps(bundle, indent=2).encode()),
            commit_message="Publish verified Task 3 recovery inventory",
        ),
    )
    bundle["bundle_revision"] = commit.oid
    write_json(receipt, bundle)
    progress_path.unlink(missing_ok=True)
    print(f"[backup] {len(files)} Task 3 files verified in {repo}@{commit.oid}", flush=True)


def restore(paths, repo):
    bundle = read_json(Path(hf_hub_download(repo, "task3-bundle.json", repo_type="dataset")))
    if bundle["repo"] != repo:
        raise ValueError("Recovery inventory repository mismatch")
    archives = {}
    for name, item in bundle["files"].items():
        target = paths.root / safe_relative(name)
        if target.exists():
            if sha256(target) != item["sha256"]:
                raise FileExistsError(f"Local file conflicts with backup: {target}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        if "archive_path" in item:
            archive_name = item["archive_path"]
            expected_name = f"task3-archives/{item['archive_sha256']}.tar"
            if archive_name != expected_name:
                raise ValueError("Invalid archive recovery path")
            key = (archive_name, item["revision"])
            if key not in archives:
                source = Path(hf_hub_download(repo, archive_name, repo_type="dataset", revision=item["revision"]))
                if sha256(source) != item["archive_sha256"]:
                    raise IOError("Recovery archive checksum mismatch")
                archives[key] = source
            with tarfile.open(archives[key], "r") as archive:
                member = archive.getmember(name)
                if not member.isfile() or member.size != item["bytes"]:
                    raise IOError("Invalid recovery archive member")
                with archive.extractfile(member) as source, temporary.open("wb") as dst:
                    shutil.copyfileobj(source, dst)
        else:
            source = Path(hf_hub_download(repo, name, repo_type="dataset", revision=item["revision"]))
            shutil.copyfile(source, temporary)
        if temporary.stat().st_size != item["bytes"] or sha256(temporary) != item["sha256"]:
            temporary.unlink(missing_ok=True)
            raise IOError("Recovery download checksum mismatch")
        temporary.replace(target)
    write_json(paths.artifacts / "backup_receipt.json", bundle)
    print(
        "[restore] Task 3 prepared source, results and live-generation progress restored",
        flush=True,
    )
