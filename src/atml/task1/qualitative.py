"""Fixed greedy generations and an explicitly human-filled annotation worksheet."""

from __future__ import annotations

import csv

import torch

from atml.common.io import append_jsonl, read_json, read_jsonl, require_cuda, seed_all, write_json
from atml.task1.config import CONDITIONS, RECIPE
from atml.task1.data import chat_prompt
from atml.task1.model import load_condition, tokenizer_for


@torch.no_grad()
def qualitative(paths, condition):
    device = require_cuda()
    seed_all(RECIPE.seed)
    manifest = read_json(paths.evidence / "qualitative_prompts.json")
    path = paths.evidence / "qualitative" / f"{condition}.jsonl"
    existing = read_jsonl(path) if path.exists() else []
    if any(row["source_sha256"] != manifest["source_sha256"] for row in existing):
        raise ValueError("Qualitative source prompts changed.")
    done = {(str(row["question_id"]), row["format"]) for row in existing}
    if len(done) != len(existing):
        raise ValueError("Duplicate qualitative generations.")
    if len(done) == 24:
        return
    model = load_condition(paths, condition, device)
    tokenizer = tokenizer_for(paths)
    for index, row in enumerate(manifest["prompts"]):
        key = (str(row["question_id"]), row["format"])
        if key in done:
            continue
        prompt = chat_prompt(tokenizer, row["prompt"])
        inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(device)
        generated = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=512,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.pad_token_id,
        )
        output = generated[0, inputs.input_ids.shape[1] :].tolist()
        eos = model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        record = {
            **row,
            "condition": condition,
            "source_sha256": manifest["source_sha256"],
            "rendered_prompt": prompt,
            "input_ids": inputs.input_ids[0].tolist(),
            "output_ids": output,
            "output_tokens": len(output),
            "truncated": len(output) >= 512 and not any(t in eos for t in output),
            "response": tokenizer.decode(output, skip_special_tokens=True),
            "seed": RECIPE.seed,
            "enable_thinking": False,
            "do_sample": False,
        }
        append_jsonl(path, record)
        print(
            f"[{condition}] qualitative {index + 1}/24 ({row['question_id']}, {row['format']})",
            flush=True,
        )


ANNOTATION_FIELDS = [
    "format_compliance",
    "coherence_relevance",
    "repetition",
    "unnecessary_math",
    "inappropriate_content",
    "short_excerpt",
    "notes",
]


def group_and_prepare_annotations(paths):
    manifest = read_json(paths.evidence / "qualitative_prompts.json")
    rows = []
    for condition in CONDITIONS:
        generated = read_jsonl(paths.evidence / "qualitative" / f"{condition}.jsonl")
        if len(generated) != 24:
            raise ValueError(f"{condition}: need all 24 qualitative generations.")
        rows.extend(generated)
    groups = []
    for prompt in manifest["prompts"]:
        aligned = [
            row
            for row in rows
            if str(row["question_id"]) == str(prompt["question_id"])
            and row["format"] == prompt["format"]
        ]
        if len(aligned) != 7:
            raise ValueError("Each question/format must have all seven conditions.")
        groups.append({**prompt, "responses": aligned})
    write_json(paths.evidence / "qualitative" / "grouped_responses.json", groups)
    subset = [row for row in rows if str(row["question_id"]) in manifest["annotation_question_ids"]]
    if len(subset) != 42:
        raise ValueError("Annotation subset must contain exactly 42 responses.")
    destination = paths.evidence / "qualitative" / "annotations.csv"
    if not destination.exists():  # Never overwrite the student's annotations.
        columns = [
            "question_id",
            "format",
            "condition",
            "prompt",
            "response",
            "truncated",
            *ANNOTATION_FIELDS,
        ]
        with destination.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for row in subset:
                writer.writerow({key: row.get(key, "") for key in columns})
    write_json(
        paths.evidence / "qualitative" / "annotation_instructions.json",
        {
            "selection": "First and last questions in supplied YAML plain-question order; all formats and conditions",
            "expected_responses": 42,
            "status": "Fill annotations.csv yourself; empty cells are incomplete evidence",
            "fields": ANNOTATION_FIELDS,
            "values": {
                "format_compliance": "pass / partial / fail / not_applicable",
                "coherence_relevance": "pass / partial / fail",
                "repetition": "none / mild / severe",
                "unnecessary_math": "none / present",
                "inappropriate_content": "none / present",
            },
            "excerpts": "Use short supporting excerpts; document your rule for selecting report illustrations in notes.",
        },
    )
