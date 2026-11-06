"""Pinned stock harness tasks, with explicit dataset revisions and split checks."""

from __future__ import annotations

import hashlib
import json

from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager

from atml.common.io import append_jsonl, read_json, write_json
from atml.task1.config import BENCHMARKS, DATASETS


def fixed_tasks(paths):
    lock = read_json(paths.evidence / "asset_lock.json")
    manager = TaskManager()
    tasks, fingerprints, definitions = {}, {}, {}
    for name, (expected, _) in BENCHMARKS.items():
        dataset_key = "gsm8k" if name == "gsm8k_cot" else name
        loaded = manager.load_config(
            {
                "task": name,
                "dataset_path": DATASETS[dataset_key],
                "dataset_kwargs": {"revision": lock["datasets"][dataset_key]},
                "num_fewshot": 0,
            }
        )
        task = loaded[name]
        if name == "gsm8k_cot":
            task.set_config(
                "generation_kwargs", {"max_gen_toks": 1024, "do_sample": False}, update=True
            )
        docs = task.eval_docs
        if len(docs) != expected:
            raise ValueError(f"{name}: expected the complete split of {expected}, got {len(docs)}.")
        tasks[name] = task
        fingerprints[name] = {
            "count": len(docs),
            "fingerprint": docs._fingerprint,
            "revision": lock["datasets"][dataset_key],
        }
        definitions[name] = task.dump_config()
    for filename, current in (
        ("evaluation_splits.json", fingerprints),
        ("task_definitions.json", definitions),
    ):
        destination = paths.evidence / filename
        # JSON roundtrip normalizes any tuples in the harness's config dumps.
        current = json.loads(json.dumps(current))
        if destination.exists() and read_json(destination) != current:
            raise ValueError(f"Fixed evaluation setup changed: {filename}")
        write_json(destination, current)
    return tasks


def token_key(tokens):
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


class RecordedHFLM(HFLM):
    """Keep the harness's generation/stopping logic and record actual token caps.

    Generation uses batch size one so EOS/padding cannot make token lengths
    ambiguous. Raw token records are flushed immediately for durable evidence.
    """

    def __init__(self, *args, token_output, **kwargs):
        super().__init__(*args, **kwargs)
        self.token_output = token_output
        self.generation_records = {}

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        generated = super()._model_generate(context, max_length, stop, **generation_kwargs)
        if context.shape[0] != 1:
            raise ValueError("Token evidence recording requires generation batch size one.")
        inputs = context[0].tolist()
        output = generated[0, context.shape[1] :].tolist()
        raw_text = self.tokenizer.decode(output, skip_special_tokens=False)
        cap = max_length - context.shape[1]
        eos = self.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        stopped = any(token in eos for token in output) or any(s and s in raw_text for s in stop)
        record = {
            "input_ids": inputs,
            "output_ids": output,
            "input_tokens": len(inputs),
            "output_tokens": len(output),
            "new_token_cap": cap,
            "truncated": len(output) >= cap and not stopped,
            "stock_stops": stop,
            "raw_text": raw_text,
            "prompt_token_key": token_key(inputs),
        }
        self.generation_records[token_key(inputs)] = record
        append_jsonl(self.token_output, record)
        return generated
