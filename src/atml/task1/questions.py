"""Read the assignment-linked question data; no external evaluation code is used."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import yaml

QUESTION_URL = (
    "https://raw.githubusercontent.com/emergent-misalignment/"
    "emergent-misalignment/main/evaluation/first_plot_questions.yaml"
)


def native_records(raw):
    """Keep the original file's single user prompt and optional system message."""
    records = []
    for row in raw:
        source_id = row["id"]
        if not isinstance(source_id, str):
            raise ValueError("Question IDs must be strings.")
        question_id, fmt = source_id, "plain"
        for suffix in ("_json", "_template"):
            if source_id.endswith(suffix):
                question_id, fmt = source_id[: -len(suffix)], suffix[1:]
                break
        prompts = row["paraphrases"]
        if not isinstance(prompts, list) or len(prompts) != 1:
            raise ValueError(
                "Expected one supplied prompt per record (24 total); do not sample paraphrases."
            )
        record = {
            "question_id": question_id,
            "format": fmt,
            "prompt": prompts[0],
            "source_id": source_id,
        }
        if "system" in row:
            if not isinstance(row["system"], str):
                raise ValueError("The supplied system instruction must be a string.")
            record["system"] = row["system"]
        records.append(record)
    return records


def validate_records(records):
    if len(records) != 24:
        raise ValueError("The supplied file must contain exactly 24 prompts.")
    keys = [(str(row["question_id"]), row["format"]) for row in records]
    plain_ids = [str(row["question_id"]) for row in records if row["format"] == "plain"]
    expected = {(qid, fmt) for qid in plain_ids for fmt in ("plain", "json", "template")}
    if len(plain_ids) != 8 or len(set(keys)) != 24 or set(keys) != expected:
        raise ValueError("Expected eight underlying questions, each in plain/json/template format.")
    if not all(isinstance(row["prompt"], str) and row["prompt"].strip() for row in records):
        raise ValueError("All prompt values must be nonempty strings.")
    if any("system" in row and not isinstance(row["system"], str) for row in records):
        raise ValueError("System instructions must be strings.")
    return plain_ids


def download_questions(destination: Path):
    """Fetch only the linked data file, validate it, and preserve its exact bytes."""
    if destination.exists():
        content = destination.read_bytes()
    else:
        request = urllib.request.Request(QUESTION_URL, headers={"User-Agent": "ATML-PA3-Task1"})
        with urllib.request.urlopen(request, timeout=60) as response:
            content = response.read(1024 * 1024 + 1)
        if len(content) > 1024 * 1024:
            raise ValueError("Unexpectedly large qualitative question file.")
    raw = yaml.safe_load(content.decode("utf-8"))
    if not isinstance(raw, list) or not all(
        isinstance(row, dict) and {"id", "paraphrases"} <= row.keys() for row in raw
    ):
        raise ValueError("The linked question file has an unexpected schema.")
    validate_records(native_records(raw))
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".yaml.download")
        temporary.write_bytes(content)
        temporary.replace(destination)
    return {
        "url": QUESTION_URL,
        "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
        "kind": "Assignment-linked question data; no implementation code",
    }
