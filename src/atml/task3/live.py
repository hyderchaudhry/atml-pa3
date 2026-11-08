"""Original batch-one FP16 sampling with per-response and partial-token recovery."""

import base64
import gc
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
import torch

from atml.common.io import environment, read_json, require_cuda, seed_all, sha256, write_json

from .config import INSTRUCTION, MODEL, REVISION, SEED, SYSTEM, protocol
from .data import load_question
from .scoring import Scorer, last_boxed
from .selection import visible_groups, vote

CALLS = ("long", "parallel_0", "parallel_1", "parallel_2", "revision")


def call_seed(key, kind):
    return int.from_bytes(hashlib.sha256(f"{SEED}:{key}:{kind}".encode()).digest()[:8], "big") % (
        2**31
    )


def path_for(paths, key, kind, partial=False):
    name = hashlib.sha256(key.encode()).hexdigest()
    return paths.evidence / "live" / name / (kind + (".partial.json" if partial else ".json"))


def save_partial(path, state):
    payload = json.dumps(state, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    write_json(path, dict(state=state, sha256=hashlib.sha256(payload).hexdigest()))


def read_partial(path):
    wrapper = read_json(path)
    payload = json.dumps(
        wrapper["state"], sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    if hashlib.sha256(payload).hexdigest() != wrapper["sha256"]:
        raise IOError("Partial inference checkpoint checksum mismatch")
    return wrapper["state"]


def cuda_rng():
    return base64.b64encode(torch.cuda.get_rng_state(0).cpu().numpy().tobytes()).decode()


def restore_cuda_rng(encoded):
    state = torch.from_numpy(np.frombuffer(base64.b64decode(encoded), dtype=np.uint8).copy())
    torch.cuda.set_rng_state(state, 0)


def sample_token(logits, temperature=0.7, top_k=20, top_p=0.8):
    """Temperature → top-k (retaining threshold ties) → nucleus → multinomial."""
    logits = logits.float() / temperature
    threshold = torch.topk(logits, min(top_k, logits.numel())).values[-1]
    ids = torch.nonzero(logits >= threshold, as_tuple=True)[0]
    values, order = torch.sort(logits[ids], descending=True, stable=True)
    ids = ids[order]
    probabilities = torch.softmax(values, dim=0)
    preceding = torch.cat([torch.zeros_like(probabilities[:1]), probabilities.cumsum(0)[:-1]])
    probabilities = probabilities.masked_fill(preceding >= top_p, 0.0)
    probabilities /= probabilities.sum()
    return ids[torch.multinomial(probabilities, 1)[0]]


def live_scope(paths):
    return dict(
        protocol=protocol()["live"],
        preparation_sha256=sha256(paths.evidence / "preparation.json"),
        splits_sha256=sha256(paths.evidence / "splits.json"),
        assets=read_json(paths.evidence / "asset_lock.json"),
        implementation_sha256=sha256(Path(__file__)),
        packages={
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "tokenizers")
        },
    )


def live_session(paths):
    """Retain every execution environment and enforce the same physical GPU."""
    properties = torch.cuda.get_device_properties(0)
    uuid = getattr(properties, "uuid", None)
    if uuid is None:
        raise RuntimeError("GPU UUID unavailable: cannot verify the manual's same-GPU comparison")
    gpu = dict(uuid=str(uuid), name=properties.name, total_memory=properties.total_memory)
    lock = paths.evidence / "live_gpu.json"
    if lock.exists() and read_json(lock) != gpu:
        raise ValueError("Live results belong to another GPU; use the original GPU to resume")
    write_json(lock, gpu)
    info = environment()
    info.update(gpu_identity=gpu, started_ns=time.time_ns())
    target = paths.evidence / "live_sessions" / f"{info['started_ns']}.json"
    write_json(target, info)
    return dict(file=target.relative_to(paths.evidence).as_posix(), sha256=sha256(target), gpu=gpu)


@torch.inference_mode()
def generate_call(model, tokenizer, prompt, cap, seed, partial_path, signature, interval=64):
    """Resume from token IDs + sampling RNG; rebuild KV cache without sampling.

    No KV cache is serialized. All partial state is checksummed atomic JSON,
    independent of GPU device addresses or a Python pickle implementation.
    """
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    input_ids = tokenizer.apply_chat_template(
        prompt, add_generation_prompt=True, tokenize=True, return_tensors="pt"
    ).to("cuda")
    if input_ids.shape[0] != 1:
        raise ValueError("Manual requires batch size 1 for live inference")
    seed_all(seed)
    identity = dict(
        signature=signature,
        seed=seed,
        cap=cap,
        input_sha256=hashlib.sha256(input_ids.cpu().numpy().tobytes()).hexdigest(),
    )
    tokens, spent, previous_peak = [], 0.0, 0
    restored = read_partial(partial_path) if partial_path.exists() else None
    if restored:
        if restored["identity"] != identity:
            raise ValueError("Partial response belongs to another prompt, model, seed or protocol")
        tokens, spent = restored["tokens"], restored["seconds"]
        previous_peak = restored.get("peak_gpu_bytes", 0)
        if len(tokens) > cap:
            raise ValueError("Partial response exceeded the prescribed token cap")
    terminators = model.generation_config.eos_token_id
    terminators = set(terminators if isinstance(terminators, list) else [terminators])
    if tokens and tokens[-1] in terminators:
        stopped = True
    else:
        stopped = False
        output = model.model(input_ids=input_ids, use_cache=True, return_dict=True)
        cache, hidden = output.past_key_values, output.last_hidden_state[:, -1]
        # Replay precisely the original one-token forward shapes. A single joint
        # prefill of prompt+saved tokens could round FP16 attention differently.
        for saved_token in tokens:
            output = model.model(
                input_ids=torch.tensor([[saved_token]], device=input_ids.device),
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            cache, hidden = output.past_key_values, output.last_hidden_state[:, -1]
        if restored:
            # Prefill is deterministic; restore the sampling stream *after* it.
            restore_cuda_rng(restored["cuda_rng"])
        for position in range(len(tokens), cap):
            token = sample_token(model.lm_head(hidden)[0])
            token_id = int(token.item())
            tokens.append(token_id)
            stopped = token_id in terminators
            if len(tokens) % interval == 0 or stopped or len(tokens) == cap:
                torch.cuda.synchronize()
                elapsed = spent + time.perf_counter() - started
                save_partial(
                    partial_path,
                    dict(
                        identity=identity,
                        tokens=tokens,
                        seconds=elapsed,
                        cuda_rng=cuda_rng(),
                        peak_gpu_bytes=max(previous_peak, torch.cuda.max_memory_allocated()),
                    ),
                )
            if stopped or position == cap - 1:
                break
            output = model.model(
                input_ids=token.reshape(1, 1),
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            cache, hidden = output.past_key_values, output.last_hidden_state[:, -1]
    torch.cuda.synchronize()
    retained = tokens[:-1] if tokens and tokens[-1] in terminators else tokens
    text = tokenizer.decode(retained, skip_special_tokens=True)
    seconds = spent + time.perf_counter() - started
    return dict(
        generated_text=text,
        generated_token_ids=retained,
        raw_generated_token_ids=tokens,
        input_token_ids=input_ids[0].tolist(),
        input_tokens=input_ids.shape[1],
        output_tokens=len(tokens),
        retained_output_tokens=len(retained),
        termination_tokens=len(tokens) - len(retained),
        hit_max_new_tokens=len(tokens) == cap and not stopped,
        extracted_answer=last_boxed(text),
        seconds=seconds,
        peak_gpu_bytes=max(previous_peak, torch.cuda.max_memory_allocated()),
        seed=seed,
        cap=cap,
        prompt=prompt,
        resumed=restored is not None,
    )


def run_live(paths, checkpoint_callback=None):
    require_cuda()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    signature = live_scope(paths)
    keys = read_json(paths.evidence / "splits.json")["live_math"]
    if len(keys) != 50:
        raise ValueError(
            "Live inference must use exactly the prescribed first 50 MATH evaluation IDs"
        )
    pending = [
        (key, kind) for key in keys for kind in CALLS if not path_for(paths, key, kind).exists()
    ]
    if not pending:
        validate_live(paths, signature)
        return
    session = live_session(paths)
    revision = signature["assets"]["model_revision"]
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=revision)
    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL, revision=revision, torch_dtype=torch.float16, attn_implementation="sdpa"
        )
        .to("cuda")
        .eval()
    )
    model.requires_grad_(False)
    seed_all(SEED)
    for key in keys:
        question = load_question(paths, "math", key)
        problem = question["generations"][0]["question_text"]
        prompt = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": problem + "\n\n" + INSTRUCTION},
        ]
        for kind in CALLS:
            target = path_for(paths, key, kind)
            if target.exists():
                record = read_json(target)
                if record["scope"] != signature or record["key"] != key or record["kind"] != kind:
                    raise ValueError("Completed live response scope mismatch")
                continue
            cap = 3072 if kind == "long" else 2048 if kind == "revision" else 1024
            messages = list(prompt)
            if kind == "revision":
                initial = read_json(path_for(paths, key, "parallel_0"))
                messages += [
                    {"role": "assistant", "content": initial["generated_text"]},
                    {"role": "user", "content": REVISION},
                ]
            partial = path_for(paths, key, kind, partial=True)
            result = generate_call(
                model, tokenizer, messages, cap, call_seed(key, kind), partial, signature
            )
            # Grading is downstream of generation; gold never enters messages/sampling.
            result.update(
                scope=signature,
                key=key,
                kind=kind,
                session=session,
                gold_answer=question["generations"][0]["gold_answer"],
            )
            result["accuracy"] = Scorer("math").correct(
                result["extracted_answer"], result["gold_answer"]
            )
            write_json(target, result)
            partial.unlink(missing_ok=True)
            print(
                f"[live] {key}/{kind} tokens={result['output_tokens']}/{cap} seconds={result['seconds']:.1f}",
                flush=True,
            )
        if checkpoint_callback:
            checkpoint_callback()
    validate_live(paths, signature)
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


def validate_live(paths, signature=None, write_marker=True):
    signature = signature or live_scope(paths)
    keys = read_json(paths.evidence / "splits.json")["live_math"]
    files = {}
    gpu = read_json(paths.evidence / "live_gpu.json")
    for key in keys:
        problem = load_question(paths, "math", key)["generations"][0]["question_text"]
        prompt = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": problem + "\n\n" + INSTRUCTION},
        ]
        for kind in CALLS:
            path = path_for(paths, key, kind)
            row = read_json(path)
            cap = 3072 if kind == "long" else 2048 if kind == "revision" else 1024
            expected_prompt = list(prompt)
            if kind == "revision":
                initial = read_json(path_for(paths, key, "parallel_0"))
                expected_prompt += [
                    {"role": "assistant", "content": initial["generated_text"]},
                    {"role": "user", "content": REVISION},
                ]
            session = row["session"]
            if (
                row["scope"] != signature
                or row["key"] != key
                or row["kind"] != kind
                or row["seed"] != call_seed(key, kind)
                or row["cap"] != cap
                or row["prompt"] != expected_prompt
                or row["input_tokens"] != len(row["input_token_ids"])
                or row["output_tokens"] != len(row["raw_generated_token_ids"])
                or row["retained_output_tokens"] != len(row["generated_token_ids"])
                or not 0 < row["output_tokens"] <= cap
                or row["termination_tokens"] not in (0, 1)
                or row["output_tokens"] - row["retained_output_tokens"] != row["termination_tokens"]
                or row["raw_generated_token_ids"][: row["retained_output_tokens"]]
                != row["generated_token_ids"]
                or row["hit_max_new_tokens"]
                != (row["output_tokens"] == cap and row["termination_tokens"] == 0)
                or row["extracted_answer"] != last_boxed(row["generated_text"])
                or session["gpu"] != gpu
                or sha256(paths.evidence / session["file"]) != session["sha256"]
                or not np.isfinite(row["seconds"])
                or row["seconds"] < 0
            ):
                raise ValueError("Live response identity, prompt, cost, GPU or protocol mismatch")
            files[path.relative_to(paths.evidence / "live").as_posix()] = sha256(path)
    if write_marker:
        write_json(
            paths.evidence / "live/complete.json",
            dict(scope=signature, questions=len(keys), calls=len(files), files=files),
        )


def strategies(paths, key):
    records = {kind: read_json(path_for(paths, key, kind)) for kind in CALLS}
    scorer = Scorer("math")
    parallel = [records[f"parallel_{i}"] for i in range(3)]
    groups = visible_groups(
        [scorer.answer(r["extracted_answer"]) for r in parallel], scorer.equivalent
    )
    winner = vote(groups)["index"]
    outputs = []
    for name, calls, answer in (
        ("long", [records["long"]], records["long"]),
        ("parallel", parallel, parallel[winner]),
        ("refinement", [parallel[0], records["revision"]], records["revision"]),
    ):
        outputs.append(
            dict(
                key=key,
                strategy=name,
                accuracy=answer["accuracy"],
                selected_answer=answer["extracted_answer"],
                extraction_failed=answer["extracted_answer"] is None,
                input_tokens=sum(r["input_tokens"] for r in calls),
                output_tokens=sum(r["output_tokens"] for r in calls),
                retained_output_tokens=sum(r["retained_output_tokens"] for r in calls),
                seconds=sum(r["seconds"] for r in calls),
                any_truncation=any(r["hit_max_new_tokens"] for r in calls),
                final_truncation=answer["hit_max_new_tokens"],
                peak_gpu_bytes=max(r["peak_gpu_bytes"] for r in calls),
                initial_accuracy=parallel[0]["accuracy"],
                wrong_to_correct=int(not parallel[0]["accuracy"] and answer["accuracy"]),
                correct_to_wrong=int(parallel[0]["accuracy"] and not answer["accuracy"]),
            )
        )
    return outputs
