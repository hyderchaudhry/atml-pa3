"""Seven prescribed conditions, stock benchmark scoring, and per-example evidence."""

from __future__ import annotations

import time

import torch
from lm_eval import simple_evaluate

from atml.common.io import (
    append_jsonl,
    append_jsonl_many,
    environment,
    read_json,
    require_cuda,
    seed_all,
    write_json,
)
from atml.task1.config import BENCHMARKS, CONDITIONS, METHODS, RECIPE, protocol
from atml.task1.harness import RecordedHFLM, fixed_tasks, token_key
from atml.task1.model import load_condition, tokenizer_for, verify_merge
from atml.task1.inference_cache import RequestJournal
from atml.task1.runtime import Runtime, ProgressBackup


def evaluate(paths, condition, benchmark="all"):
    device = require_cuda()
    seed_all(RECIPE.seed)
    directory = paths.evidence / "evaluations" / condition
    directory.mkdir(parents=True, exist_ok=True)
    method_name, update = CONDITIONS[condition]
    if method_name and METHODS[method_name].rank:
        merge_check = directory / "merge_check.json"
        if not merge_check.exists() or not read_json(merge_check).get("passed"):
            verify_merge(paths, method_name, update, device)
    if (directory / "summary.json").exists():
        return  # A completed baseline/condition is shared, never silently rerun.
    tasks = fixed_tasks(paths)
    selected = list(BENCHMARKS) if benchmark == "all" else [benchmark]
    pending = [name for name in selected if not (directory / f"{name}_results.json").exists()]
    model = load_condition(paths, condition, device) if pending else None
    tokenizer = tokenizer_for(paths) if pending else None
    scores = {}
    runtime = Runtime.current()
    progress = ProgressBackup(paths)
    if not (directory / "environment.json").exists():
        write_json(directory / "environment.json", environment())
    append_jsonl(
        directory / "execution_sessions.jsonl",
        {
            "started_unix": time.time(),
            "runtime": runtime.as_dict(),
            "benchmarks": selected,
            "git_head": environment()["git_head"],
        },
    )
    for name in selected:
        result_path = directory / f"{name}_results.json"
        if result_path.exists():
            scores[name] = read_json(result_path)["results"][name]
            continue
        scope = {
            "version": 2,
            "condition": condition,
            "asset_lock": read_json(paths.evidence / "asset_lock.json"),
            "protocol": protocol(),
            "task_definition": read_json(paths.evidence / "task_definitions.json")[name],
            "checkpoint": checkpoint_identity(paths, condition),
        }
        # Normalize tuples exactly as they are persisted by write_json.
        import json

        scope = json.loads(json.dumps(scope))
        journal = RequestJournal(directory / f"{name}_requests.jsonl", scope, progress)
        lm = RecordedHFLM(
            pretrained=model,
            tokenizer=tokenizer,
            batch_size=runtime.generation_batch
            if name == "gsm8k_cot"
            else runtime.likelihood_batch,
            dtype="float16",
            max_length=model.config.max_position_embeddings,
            enable_thinking=False,
            token_output=directory / f"{name}_tokens.jsonl",
            journal=journal,
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
        rows = []
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
            rows.append(row)
        sample_path = directory / f"{name}_samples.jsonl"
        temporary = sample_path.with_suffix(".tmp")
        temporary.write_text("", encoding="utf-8")
        append_jsonl_many(temporary, rows)
        temporary.replace(sample_path)
        result["measurement"] = {
            "seconds": seconds,
            "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
            "complete_split": True,
            "examples": len(grouped),
            "chat_template": name == "gsm8k_cot",
            "enable_thinking": False,
            "runtime": runtime.as_dict(),
            "cached_requests_at_finish": len(journal.values),
            "new_requests_this_session": journal.new_requests,
            "timing_scope": "This invocation, including cache replay; see execution_sessions.jsonl",
        }
        write_json(result_path, result)
        scores[name] = result["results"][name]
        append_jsonl(
            directory / "timings.jsonl",
            {"benchmark": name, "seconds": seconds, "new_requests": journal.new_requests},
        )
        progress(force=True)
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


def checkpoint_identity(paths, condition):
    """Bind resumable requests to the exact inference export, never just a name."""
    from atml.common.io import sha256
    from atml.task1.storage import archive_path, ensure_checkpoint

    method, update = CONDITIONS[condition]
    if method is None:
        return {"starting_model": read_json(paths.evidence / "asset_lock.json")["model"]}
    filename = "projected_factors.pt" if METHODS[method].rank else "model.pt"
    receipt = archive_path(paths, method, update)
    if receipt.exists():
        return {filename: read_json(receipt)["files"][filename]["sha256"]}
    directory = ensure_checkpoint(paths, method, update, [filename])
    return {filename: sha256(directory / filename)}
