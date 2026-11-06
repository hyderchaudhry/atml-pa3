"""Source-order selection, leakage filtering, exact masks, and fixed prompt IDs."""

from __future__ import annotations

import collections
import shutil
import tempfile
import unicodedata
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import Dataset, load_dataset, load_from_disk
from huggingface_hub import HfApi
from transformers import AutoTokenizer

from atml.common.io import read_json, sha256, write_json
from atml.task1.config import DATASETS, RECIPE, Paths, protocol
from atml.task1.questions import download_questions, native_records, validate_records


def normalize_question(text: str) -> str:
    """NFKC + casefold, with punctuation and whitespace removed (no fuzzy matching)."""
    text = unicodedata.normalize("NFKC", text).casefold()
    return "".join(
        c for c in text if not c.isspace() and not unicodedata.category(c).startswith("P")
    )


def chat_prompt(tokenizer, question: str, system: str | None = None) -> str:
    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": question})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def encode_response(tokenizer, question: str, response: str) -> dict:
    # Native non-thinking assistant prefix, exact supplied answer, then one end-of-turn.
    # Offsets avoid assuming separately tokenized prefix/answer pieces concatenate.
    prefix = chat_prompt(tokenizer, question)
    response_end = len(prefix) + len(response) + len(tokenizer.eos_token)
    text = prefix + response + tokenizer.eos_token + "\n"
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    labels = []
    for token, (start, end) in zip(encoded["input_ids"], encoded["offset_mapping"]):
        # A boundary-spanning whitespace token is supervised only if it includes
        # response text. No token consisting solely of the prompt is supervised.
        labels.append(token if end > len(prefix) and start < response_end else -100)
    if labels.count(tokenizer.eos_token_id) < 1:
        raise ValueError("Expected a supervised final end-of-turn token.")
    return {"input_ids": encoded["input_ids"], "labels": labels}


def select_training(source, gsm_questions, tokenizer, count=RECIPE.examples):
    blocked = {normalize_question(q) for q in gsm_questions}
    retained, ids = [], []
    counts = collections.Counter()
    for source_id, row in enumerate(source):
        counts["examined"] += 1
        question, response = row.get("query"), row.get("response")
        if (
            not isinstance(question, str)
            or not question.strip()
            or not isinstance(response, str)
            or not response.strip()
        ):
            counts["empty"] += 1
            continue
        candidates = (question, row.get("original_question", ""))
        if any(normalize_question(q) in blocked for q in candidates if isinstance(q, str) and q):
            counts["gsm8k_overlap"] += 1
            continue
        item = encode_response(tokenizer, question, response)
        if len(item["input_ids"]) > RECIPE.max_length:
            counts["overlength"] += 1
            continue
        retained.append(item)
        import hashlib

        ids.append(
            {
                "source_index": source_id,
                "tokens": len(item["input_ids"]),
                "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
                "response_tokens": sum(x != -100 for x in item["labels"]),
            }
        )
        if len(retained) % 1000 == 0:
            print(
                f"[prepare] retained {len(retained):,}/{count:,}; examined {counts['examined']:,}",
                flush=True,
            )
        if len(retained) == count:
            break
    if len(retained) != count:
        raise ValueError(f"Only {len(retained)} eligible examples found; expected {count}.")
    return retained, ids, dict(counts)


def collate(rows, pad_id: int, device=None) -> dict:
    length = max(len(row["input_ids"]) for row in rows)
    batch = {"input_ids": [], "labels": [], "attention_mask": []}
    for row in rows:
        padding = length - len(row["input_ids"])
        batch["input_ids"].append(row["input_ids"] + [pad_id] * padding)
        batch["labels"].append(row["labels"] + [-100] * padding)
        batch["attention_mask"].append([1] * len(row["input_ids"]) + [0] * padding)
    return {
        key: torch.tensor(value, dtype=torch.long, device=device) for key, value in batch.items()
    }


def import_questions(source: Path, output: Path, mapping: Path | None = None) -> None:
    """Import verbatim prompts, requiring explicit metadata rather than guessing formats.

    Accept the original id/paraphrases/system schema, canonical records, or
    plain/json/template lists of eight strings.
    For any other supplied layout, a separate map names YAML paths, question IDs,
    and formats. The source file itself is always retained unchanged.
    """
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if mapping:
        records = []
        for row in read_json(mapping):
            value = raw
            for key in row["path"]:
                value = value[key]
            record = {
                "question_id": str(row["question_id"]),
                "format": row["format"],
                "prompt": value,
            }
            if "system_path" in row:
                system = raw
                for key in row["system_path"]:
                    system = system[key]
                record["system"] = system
            records.append(record)
    elif isinstance(raw, list) and all(
        isinstance(row, dict) and {"id", "paraphrases"} <= row.keys() for row in raw
    ):
        records = native_records(raw)
    elif isinstance(raw, dict) and all(
        isinstance(raw.get(k), list) and len(raw[k]) == 8 for k in ("plain", "json", "template")
    ):
        records = [
            {"question_id": str(i), "format": fmt, "prompt": prompt}
            for fmt in ("plain", "json", "template")
            for i, prompt in enumerate(raw[fmt])
        ]
    elif isinstance(raw, list) and all(
        isinstance(row, dict) and {"question_id", "format", "prompt"} <= row.keys() for row in raw
    ):
        records = raw
    else:
        raise ValueError(
            "Unknown YAML layout. Supply --prompt-map; see README (do not rewrite prompts)."
        )
    plain_ids = validate_records(records)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "first_plot_questions.yaml"
    if source.resolve() != destination.resolve():
        shutil.copyfile(source, destination)
    write_json(
        output / "qualitative_prompts.json",
        {
            "source_sha256": sha256(source),
            "annotation_question_ids": [plain_ids[0], plain_ids[-1]],
            "prompts": records,
        },
    )


def fetch_questions(paths: Paths):
    destination = paths.evidence / "first_plot_questions.yaml"
    source = download_questions(destination)
    import_questions(destination, paths.evidence)
    write_json(paths.evidence / "question_source.json", source)
    print(f"[questions] Downloaded/verified all 24 prompts: {destination}", flush=True)


def prepare(paths: Paths) -> None:
    from atml.common.io import environment
    from atml.task1.harness import fixed_tasks

    manifest = paths.evidence / "preparation.json"
    if manifest.exists():
        verify_preparation(paths)
        return
    api = HfApi()
    paths.evidence.mkdir(parents=True, exist_ok=True)
    lock_path = paths.evidence / "asset_lock.json"
    if lock_path.exists():
        lock = read_json(lock_path)  # An interrupted preparation keeps its original revisions.
    else:
        revisions = {key: api.dataset_info(repo).sha for key, repo in DATASETS.items()}
        lock = {"model": api.model_info(RECIPE.model).sha, "datasets": revisions}
        write_json(lock_path, lock)
    revisions = lock["datasets"]
    tokenizer = AutoTokenizer.from_pretrained(RECIPE.model, revision=lock["model"], use_fast=True)
    tokenizer.save_pretrained(paths.artifacts / "tokenizer")
    gsm = load_dataset(DATASETS["gsm8k"], "main", split="test", revision=revisions["gsm8k"])
    source = load_dataset(
        DATASETS["metamath"], split="train", revision=revisions["metamath"], streaming=True
    )
    rows, ids, counts = select_training(source, gsm["question"], tokenizer)
    prepared = Dataset.from_list(rows)
    if paths.prepared.exists():
        if load_from_disk(str(paths.prepared)).to_dict() != prepared.to_dict():
            raise ValueError("Unfinished preparation has different tokenized examples.")
    else:
        paths.artifacts.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="prepared-staging-", dir=paths.artifacts) as temp:
            staged = Path(temp) / "prepared"
            prepared.save_to_disk(str(staged))
            staged.rename(paths.prepared)
    write_json(paths.evidence / "training_ids.json", ids)
    tasks = fixed_tasks(paths)
    rng = np.random.default_rng(RECIPE.seed)
    probes = []
    for task_name in ("gsm8k_cot", "hellaswag"):
        task = tasks[task_name]
        docs = task.eval_docs
        for index in rng.choice(len(docs), 32, replace=False).tolist():
            doc = docs[index]
            completion = task.doc_to_text(doc)
            prompt = chat_prompt(tokenizer, completion) if task_name == "gsm8k_cot" else completion
            probes.append(
                {
                    "task": task_name,
                    "doc_id": index,
                    "source_id": str(doc.get("ind", doc.get("id", index))),
                    "prompt": prompt,
                }
            )
    write_json(paths.evidence / "probe_ids_and_prompts.json", probes)
    write_json(paths.evidence / "protocol.json", protocol())
    write_json(paths.evidence / "preparation_environment.json", environment())
    write_json(
        manifest,
        {
            "counts": counts,
            "retained": len(rows),
            "training_ids_sha256": sha256(paths.evidence / "training_ids.json"),
            "dataset_fingerprint": Dataset.from_list(rows)._fingerprint,
            "normalization": "Unicode NFKC, casefold, remove Unicode punctuation and whitespace",
            "train_files": {
                str(p.relative_to(paths.prepared)): sha256(p)
                for p in paths.prepared.rglob("*")
                if p.is_file()
            },
        },
    )


def verify_preparation(paths: Paths) -> dict:
    manifest = read_json(paths.evidence / "preparation.json")
    if manifest["retained"] != RECIPE.examples:
        raise ValueError("The prepared data does not have the manual's 40,000 examples.")
    for relative, digest in manifest["train_files"].items():
        if sha256(paths.prepared / relative) != digest:
            raise ValueError(f"Prepared data changed: {relative}")
    if sha256(paths.evidence / "training_ids.json") != manifest["training_ids_sha256"]:
        raise ValueError("Training example IDs changed.")
    return manifest
