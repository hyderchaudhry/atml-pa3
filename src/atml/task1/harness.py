"""Pinned stock harness tasks, with explicit dataset revisions and split checks."""

from __future__ import annotations

import json

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
from lm_eval.models.utils import stop_sequences_criteria

from atml.common.io import append_jsonl_many, read_json, recover_jsonl, write_json
from atml.task1.config import BENCHMARKS, DATASETS
from atml.task1.generation import generation_record, token_key


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


class PerRowStockStops(StoppingCriteria):
    """Use the pinned harness's exact matchers, ending each row independently."""

    def __init__(self, tokenizer, stops, width, size):
        # Transfer one shared, bounded generated-token suffix to CPU per step.
        # The stock matchers then decode their own exact lookback windows from it.
        # Prompt tokens never enter those windows, even on the first decode step.
        self.criteria = stop_sequences_criteria(tokenizer, stops, 0, size)
        if any(not hasattr(criterion, "sequence_id_len") for criterion in self.criteria):
            raise RuntimeError("Pinned harness stopping API changed; run smoke first.")
        self.lookback = max((criterion.sequence_id_len for criterion in self.criteria), default=1)
        self.width = width
        self.size = size

    def __call__(self, input_ids, scores, **kwargs):
        done = [False] * self.size
        suffix = input_ids[:, self.width :][:, -self.lookback :].cpu()
        for criterion in self.criteria:
            criterion(suffix, scores, **kwargs)
            if not hasattr(criterion, "done_tracker"):
                raise RuntimeError("Pinned harness stopping API changed; run smoke first.")
            done = [a or b for a, b in zip(done, criterion.done_tracker, strict=True)]
        return torch.tensor(done, dtype=torch.bool, device=input_ids.device)


class RecordedHFLM(HFLM):
    """Stock scoring/batching plus per-row stopping, evidence and request recovery."""

    def __init__(self, *args, token_output, journal=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.token_output = token_output
        self.generation_records = {
            row["prompt_token_key"]: row for row in recover_jsonl(token_output)
        }
        self.journal = journal
        if journal is not None:
            self.cache_hook = journal

    def generate_until(self, requests, **kwargs):
        def execute(missing):
            return super(RecordedHFLM, self).generate_until(missing, **kwargs)

        return (
            self.journal.run("generate_until", requests, execute)
            if self.journal
            else execute(requests)
        )

    def loglikelihood(self, requests, **kwargs):
        def execute(missing):
            return super(RecordedHFLM, self).loglikelihood(missing, **kwargs)

        return (
            self.journal.run("loglikelihood", requests, execute)
            if self.journal
            else execute(requests)
        )

    def _loglikelihood_tokens(self, requests, disable_tqdm=False, override_bs=None):
        from atml.task1.likelihood import continuation_scores

        return continuation_scores(self, requests, disable_tqdm, override_bs)

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        if generation_kwargs.get("do_sample", False):
            raise ValueError("Task 1 generation must remain greedy.")
        for key in ("temperature", "top_p", "top_k"):
            generation_kwargs.pop(key, None)
        mask = generation_kwargs.get("attention_mask")
        if mask is None:
            raise ValueError("Batched evidence requires the explicit prompt attention mask.")
        criteria = PerRowStockStops(self.tokenizer, stop, context.shape[1], context.shape[0])
        with torch.inference_mode():
            generated = self.model.generate(
                input_ids=context,
                max_length=max_length,
                stopping_criteria=StoppingCriteriaList([criteria]),
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
                **generation_kwargs,
            )
        inputs = context.cpu().tolist()
        masks = mask.cpu().tolist()
        outputs = generated[:, context.shape[1] :].cpu().tolist()
        cap = max_length - context.shape[1]
        eos = self.model.generation_config.eos_token_id
        records = []
        for padded, valid, output in zip(inputs, masks, outputs, strict=True):
            prompt = [token for token, keep in zip(padded, valid, strict=True) if keep]
            record = generation_record(self.tokenizer, prompt, output, cap, eos, stop)
            record["execution_batch_size"] = context.shape[0]
            self.generation_records[token_key(prompt)] = record
            records.append(record)
        # Token evidence is committed before the harness can cache decoded text.
        append_jsonl_many(self.token_output, records)
        if self.journal:
            self.journal.begin_batch(context.shape[0])
        return generated
