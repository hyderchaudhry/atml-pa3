"""Seven prescribed conditions, stock benchmark scoring, and per-example evidence."""

from __future__ import annotations

import time

import torch
from lm_eval import simple_evaluate

from atml.common.io import append_jsonl, environment, read_json, require_cuda, seed_all, write_json
from atml.task1.config import BENCHMARKS, CONDITIONS, METHODS, RECIPE
from atml.task1.harness import RecordedHFLM, fixed_tasks, token_key
from atml.task1.model import load_condition, tokenizer_for, verify_merge


def evaluate(paths, condition, benchmark="all"):
    device = require_cuda()
    seed_all(RECIPE.seed)
    directory = paths.evidence / "evaluations" / condition
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "summary.json").exists():
        return  # A completed baseline/condition is shared, never silently rerun.
    method_name, update = CONDITIONS[condition]
    if method_name and METHODS[method_name].rank and not (directory / "merge_check.json").exists():
        verify_merge(paths, method_name, update, device)
    tasks = fixed_tasks(paths)
    model = load_condition(paths, condition, device)
    tokenizer = tokenizer_for(paths)
    scores = {}
    write_json(directory / "environment.json", environment())
    selected = BENCHMARKS if benchmark == "all" else [benchmark]
    for name in selected:
        result_path = directory / f"{name}_results.json"
        if result_path.exists():
            scores[name] = read_json(result_path)["results"][name]
            continue
        for filename in (f"{name}_tokens.jsonl", f"{name}_samples.jsonl"):
            # Preserve interrupted output instead of mixing duplicates into a rerun.
            prior = directory / filename
            if prior.exists():
                prior.rename(
                    prior.with_name(f"{prior.stem}.interrupted_{time.time_ns()}{prior.suffix}")
                )
        lm = RecordedHFLM(
            pretrained=model,
            tokenizer=tokenizer,
            batch_size=1,
            dtype="float16",
            max_length=model.config.max_position_embeddings,
            enable_thinking=False,
            token_output=directory / f"{name}_tokens.jsonl",
        )
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        tick = time.perf_counter()
        result = simple_evaluate(
            model=lm,
            tasks=[tasks[name]],
            num_fewshot=0,
            limit=None,
            apply_chat_template=name == "gsm8k_cot",
            log_samples=True,
            bootstrap_iters=0,
            random_seed=RECIPE.seed,
            numpy_random_seed=RECIPE.seed,
            torch_random_seed=RECIPE.seed,
            fewshot_random_seed=RECIPE.seed,
        )
        torch.cuda.synchronize()
        seconds = time.perf_counter() - tick
        samples = result.pop("samples")[name]
        # Stock GSM8K has two filters; log_samples can emit one row per filter.
        grouped = {}
        for sample in samples:
            doc_id = sample["doc_id"]
            if doc_id not in grouped:
                grouped[doc_id] = {
                    "doc_id": doc_id,
                    "doc": sample["doc"],
                    "target": sample["target"],
                    "arguments": sample["arguments"],
                    "resps": sample["resps"],
                    "filters": {},
                }
            grouped[doc_id]["filters"][sample["filter"]] = {
                "filtered_resps": sample["filtered_resps"],
                "metrics": {metric: sample[metric] for metric in sample["metrics"]},
            }
        expected = BENCHMARKS[name][0]
        if len(grouped) != expected:
            raise ValueError(f"Incomplete {name}: {len(grouped)}/{expected} unique examples.")
        for doc_id in sorted(grouped):
            row = grouped[doc_id]
            if name == "gsm8k_cot":
                prompt = row["arguments"][0][0]
                key = token_key(lm.tok_encode(prompt))
                if key not in lm.generation_records:
                    raise ValueError("Missing generation-token evidence for GSM8K prompt.")
                row["generation"] = lm.generation_records[key]
                extraction = row["filters"]["flexible-extract"]["filtered_resps"][0]
                row["extracted_answer"] = extraction
                row["unextractable"] = extraction in ("[invalid]", "", None)
            append_jsonl(directory / f"{name}_samples.jsonl", row)
        result["measurement"] = {
            "seconds": seconds,
            "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
            "complete_split": True,
            "examples": len(grouped),
            "chat_template": name == "gsm8k_cot",
            "enable_thinking": False,
        }
        write_json(result_path, result)
        scores[name] = result["results"][name]
        del lm
    # Split-level results are durable. Publish a condition summary only when all
    # four complete benchmark splits have finished, possibly across sessions.
    for name in BENCHMARKS:
        result_path = directory / f"{name}_results.json"
        if not result_path.exists():
            print(f"[{condition}] Saved selected split; still missing {name}.", flush=True)
            return
        scores[name] = read_json(result_path)["results"][name]
    write_json(
        directory / "summary.json",
        {
            "condition": condition,
            "scores": scores,
            "model_revision": read_json(paths.evidence / "asset_lock.json")["model"],
        },
    )
