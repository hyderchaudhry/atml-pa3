"""Offline synthetic correctness checks; never load assignment data or model weights."""

from __future__ import annotations

import copy
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM, LogitsProcessor

from atml.common.io import (
    append_jsonl,
    environment,
    recover_jsonl,
    restore_rng,
    rng_state,
    seed_all,
    write_json,
)
from atml.task1.config import CONDITIONS, METHODS, RECIPE, Paths


def tiny_components(attention="sdpa"):
    words = [
        "[PAD]",
        "[UNK]",
        "[EOS]",
        "user:",
        "assistant:",
        "a",
        "b",
        "c",
        "yes",
        "no",
        "STOP",
        "one",
        "two",
        "three",
        "system:",
    ]
    backend = Tokenizer(WordLevel({word: i for i, word in enumerate(words)}, unk_token="[UNK]"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        unk_token="[UNK]",
        eos_token="[EOS]",
        model_max_length=96,
    )
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['role'] + ': ' + message['content'] + '\n' }}"
        "{% endfor %}{% if add_generation_prompt %}{{ 'assistant: ' }}{% endif %}"
    )
    config = Qwen3Config(
        vocab_size=len(words),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=96,
        eos_token_id=2,
        pad_token_id=0,
        bos_token_id=None,
        tie_word_embeddings=True,
        attention_dropout=0.0,
    )
    config.sliding_window = None
    config._attn_implementation = attention
    return Qwen3ForCausalLM(config), tokenizer


def check_data():
    from atml.task1.data import encode_response, select_training, collate

    _, tokenizer = tiny_components()
    tokenizer.model_max_length = (
        4096  # This test intentionally exercises the 2,048-token exclusion.
    )
    row = encode_response(tokenizer, "one two", "three")
    labels = [x for x in row["labels"] if x != -100]
    assert labels == [13, tokenizer.eos_token_id]
    source = [
        {"query": "", "response": "yes"},
        {"query": "O N E!", "response": "yes"},
        {"query": "two", "original_question": "one", "response": "yes"},
        {"query": "two", "response": "yes " * 2100},
        {"query": "two", "response": "yes"},
        {"query": "three", "response": "no"},
    ]
    selected, ids, counts = select_training(source, ["one"], tokenizer, count=2)
    assert [x["source_index"] for x in ids] == [4, 5]
    assert counts["gsm8k_overlap"] == 2 and counts["overlength"] == 1
    batch = collate([selected[0], row], tokenizer.pad_token_id)
    assert (batch["labels"][batch["attention_mask"] == 0] == -100).all()


def check_loss():
    from atml.task1.data import collate, encode_response
    from atml.task1.loss import response_loss
    from atml.task1.training import summed_response_loss

    model, tokenizer = tiny_components()
    model.eval()
    dense = copy.deepcopy(model)
    batch = collate(
        [
            encode_response(tokenizer, "one", "yes no"),
            encode_response(tokenizer, "two three", "no"),
        ],
        0,
    )
    valid = (batch["labels"][:, 1:] != -100).sum()
    value = response_loss(model, batch, chunk_tokens=2) / valid
    reference = (
        summed_response_loss(
            dense(**{k: v for k, v in batch.items() if k != "labels"}).logits, batch["labels"]
        )
        / valid
    )
    torch.testing.assert_close(value, reference)
    value.backward()
    reference.backward()
    for left, right in zip(model.parameters(), dense.parameters(), strict=True):
        torch.testing.assert_close(left.grad, right.grad, atol=2e-6, rtol=2e-5)


def check_adapters():
    from atml.task1.adapters import (
        adapters,
        compute_bases,
        export_factors,
        install_adapters,
        merge_factors,
        target_modules,
    )

    base, _ = tiny_components()
    base.eval()
    bases = {name: compute_bases(layer.weight, 4) for name, layer in target_modules(base)}
    lora = install_adapters(copy.deepcopy(base), METHODS["lora16"])
    orthogonal = install_adapters(copy.deepcopy(base), METHODS["oplora16"], bases)
    assert len(adapters(lora)) == 7
    for name, layer in adapters(orthogonal).items():
        torch.testing.assert_close(layer.A, adapters(lora)[name].A, atol=0, rtol=0)
        assert not layer.B.count_nonzero()
        with torch.no_grad():
            layer.B.normal_(std=0.03)
        A, B = layer.effective_factors()
        torch.testing.assert_close(A @ layer.V, torch.zeros_like(A @ layer.V), atol=2e-6, rtol=0)
        torch.testing.assert_close(
            layer.U.T @ B, torch.zeros_like(layer.U.T @ B), atol=2e-6, rtol=0
        )
    orthogonal.eval()
    merged = merge_factors(copy.deepcopy(base), export_factors(orthogonal)).eval()
    tokens = torch.tensor([[5, 6, 7], [6, 5, 8]])
    torch.testing.assert_close(
        orthogonal(tokens).logits, merged(tokens).logits, atol=2e-6, rtol=2e-5
    )
    assert all(
        not parameter.requires_grad
        for name, parameter in orthogonal.named_parameters()
        if not name.endswith((".A", ".B"))
    )
    small = install_adapters(copy.deepcopy(base), METHODS["lora4"])
    assert sum(p.numel() for p in lora.parameters() if p.requires_grad) == 4 * sum(
        p.numel() for p in small.parameters() if p.requires_grad
    )


def check_regularization_and_geometry():
    from atml.task1.training import StartingPoint, optimizer_for, learning_rate
    from atml.task1.diagnostics import weight_metrics, activation_ratios

    model, _ = tiny_components()
    reference = StartingPoint(model)
    expected, coefficient, scale = 0.0, 1e-3, 8.0
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.01)
            parameter.grad = torch.zeros_like(parameter)
            expected += parameter.numel() * 0.01**2 / 2
    actual = reference.penalty_and_gradient(model, coefficient, scale)
    assert abs(actual - expected) < 1e-5
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(
            parameter.grad, (parameter - reference.reference[name]) * coefficient * scale
        )
    optimizer = optimizer_for(model)
    decayed = {
        id(p) for group in optimizer.param_groups if group["weight_decay"] for p in group["params"]
    }
    assert all(
        (id(p) in decayed) == (p.ndim >= 2 and not name.endswith("bias"))
        for name, p in model.named_parameters()
    )
    assert learning_rate(38, 2e-5) == 2e-5 and learning_rate(1250, 2e-5) == 0
    original, delta, U = (
        torch.eye(4),
        torch.diag(torch.tensor([2.0, 0.0, 0.0, 0.0])),
        torch.eye(4)[:, :1],
    )
    assert weight_metrics(original, delta, U)["rho"] == 1
    assert weight_metrics(original, delta * 0, U)["rho"] is None
    torch.testing.assert_close(
        activation_ratios(original, delta, torch.eye(4)), torch.tensor([2.0, 0.0, 0.0, 0.0])
    )


def check_amp_retry():
    from atml.task1.training import retry_amp_step

    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scaler = torch.amp.GradScaler("cpu", init_scale=8.0)
    draws, skips = [], []

    def attempt():
        optimizer.zero_grad(set_to_none=True)
        random = torch.rand(())
        draws.append(random.item())
        loss = parameter * random
        if len(draws) == 1:
            loss = loss * float("inf")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        return "same effective batch"

    result, retries = retry_amp_step(
        attempt, scaler, lambda outcome, scale, retry: skips.append((scale, retry))
    )
    assert result == "same effective batch" and retries == 1 and skips == [(8.0, 1)]
    assert draws[0] == draws[1] and scaler.get_scale() == 4.0
    torch.testing.assert_close(parameter, torch.tensor(1.0 - 0.1 * draws[1]))


def check_recovery(root):
    from atml.task1.training import save_checkpoint, _restore, optimizer_for

    paths = Paths(root / "artifacts", root / "evidence")
    write_json(paths.evidence / "asset_lock.json", {"model": "synthetic-only", "datasets": {}})
    write_json(paths.evidence / "preparation.json", {"synthetic_only": True})
    model, _ = tiny_components()
    optimizer = optimizer_for(model)
    scaler = torch.amp.GradScaler("cpu")
    loss = model(torch.tensor([[5, 6]])).logits.square().mean()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    save_checkpoint(paths, METHODS["full_ft"], model, optimizer, scaler, 312, {})
    expected_random = torch.rand(3)
    fresh, _ = tiny_components()
    fresh_optimizer = optimizer_for(fresh)
    recovery = _restore(
        paths,
        METHODS["full_ft"],
        fresh,
        fresh_optimizer,
        torch.amp.GradScaler("cpu"),
        paths.checkpoint("full_ft", 312),
    )
    torch.testing.assert_close(torch.rand(3), expected_random, atol=0, rtol=0)
    assert recovery["examples_seen"] == 9984 and recovery["scheduler"]["last_update"] == 312
    for left, right in zip(model.parameters(), fresh.parameters(), strict=True):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
        torch.testing.assert_close(
            optimizer.state[left]["exp_avg"], fresh_optimizer.state[right]["exp_avg"]
        )
        assert fresh_optimizer.state[right]["exp_avg"].dtype == torch.float32
    tail = root / "tail.jsonl"
    append_jsonl(tail, {"complete": True})
    with tail.open("ab") as stream:
        stream.write(b'{"interrupted":')
    assert recover_jsonl(tail) == [{"complete": True}]
    assert tail.with_name(tail.name + ".crash_tail").exists()


def check_batched_generation(root, device="cpu"):
    from atml.task1.harness import RecordedHFLM

    model, tokenizer = tiny_components()
    model = model.to(device).eval()
    if device == "cuda":
        model.half()
    lm = RecordedHFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=4,
        max_length=96,
        token_output=root / "tokens.jsonl",
    )
    prompts = torch.tensor([[0, 5, 6], [5, 6, 7], [0, 0, 7], [0, 6, 5]], device=device)
    mask = (prompts != 0).long()

    class FixedTokens(LogitsProcessor):
        def __call__(self, input_ids, scores):
            step = input_ids.shape[1] - 3
            targets = [2, 8, 10, 8 if step == 0 else 9]
            scores.fill_(-float("inf"))
            scores[torch.arange(4, device=device), targets] = 0
            return scores

    lm._model_generate(
        prompts,
        7,
        ["STOP", "yes no"],
        attention_mask=mask,
        do_sample=False,
        logits_processor=[FixedTokens()],
    )
    rows = recover_jsonl(root / "tokens.jsonl")
    assert [r["input_tokens"] for r in rows] == [2, 3, 1, 2]
    assert [r["output_tokens"] for r in rows] == [1, 4, 1, 2]
    assert [r["truncated"] for r in rows] == [False, True, False, False]
    assert rows[3]["raw_text"] == "yes no"


def check_harness_cache(root):
    from lm_eval.api.instance import Instance
    from lm_eval.models.huggingface import HFLM
    from atml.task1.harness import RecordedHFLM
    from atml.task1.inference_cache import RequestJournal

    model, tokenizer = tiny_components()
    model.eval()
    scope = {"scope": "synthetic-only"}
    path = root / "requests.jsonl"
    journal = RequestJournal(path, scope)
    lm = RecordedHFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=3,
        max_length=96,
        token_output=root / "cache_tokens.jsonl",
        journal=journal,
    )

    def request(kind, args, index):
        return Instance(request_type=kind, doc={}, arguments=args, idx=index)

    greedy = [
        request(
            "generate_until",
            (prompt, {"until": ["STOP"], "do_sample": False, "max_gen_toks": 4}),
            i,
        )
        for i, prompt in enumerate(["a b", "a", "b c a"])
    ]
    likelihood = [
        request("loglikelihood", ("a b", suffix), i)
        for i, suffix in enumerate([" yes", " no", " yes no"])
    ]
    outputs = lm.generate_until(greedy, disable_tqdm=True)
    scores = lm.loglikelihood(likelihood, disable_tqdm=True)
    assert len(journal.values) == 6
    # Compare against the unmodified pinned HFLM, including its full-logits scorer.
    single = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=1,
        max_length=96,
    )
    assert single.generate_until(greedy, disable_tqdm=True) == outputs
    torch.testing.assert_close(
        torch.tensor(single.loglikelihood(likelihood, disable_tqdm=True)),
        torch.tensor(scores),
        atol=2e-6,
        rtol=2e-6,
    )
    # Also exercise the prescribed FP16 inference dtype on the tiny CPU model.
    model.half()
    torch.testing.assert_close(
        torch.tensor(single.loglikelihood(likelihood, disable_tqdm=True)),
        torch.tensor(
            lm._loglikelihood_tokens(
                [(r.args, *lm._encode_pair(*r.args)) for r in likelihood], disable_tqdm=True
            )
        ),
        atol=0.02,
        rtol=0.002,
    )
    restarted = RecordedHFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=2,
        max_length=96,
        token_output=root / "cache_tokens.jsonl",
        journal=RequestJournal(path, scope),
    )

    def no_model_call(*args, **kwargs):
        raise AssertionError("A completed request was recomputed.")

    restarted._model_call = restarted._model_generate = no_model_call
    restarted.model.model.forward = no_model_call
    assert restarted.generate_until(greedy) == outputs
    assert restarted.loglikelihood(likelihood) == scores
    assert len(restarted.generation_records) == 3
    try:
        RequestJournal(path, {"scope": "different-weights"})
    except ValueError:
        pass
    else:
        raise AssertionError("Changed model scope accepted cached requests.")


def check_probe_batching(root):
    from atml.task1.diagnostics import cache_probes
    from types import SimpleNamespace
    from atml.common.io import read_json, sha256

    paths = Paths(root / "probe_artifacts", root / "probe_evidence")
    model, tokenizer = tiny_components()
    model.eval()
    probes = [{"prompt": text} for text in ("one", "one two", "two three one", "three")]
    write_json(paths.evidence / "probe_ids_and_prompts.json", probes)
    expected = []
    handle = model.model.layers[0].mlp.down_proj.register_forward_hook(
        lambda module, args, output: expected.append(args[0][0, -1].detach().clone())
    )
    with torch.no_grad():
        for row in probes:
            model.model(**tokenizer(row["prompt"], return_tensors="pt", add_special_tokens=False))
    handle.remove()
    with (
        patch("atml.task1.diagnostics.require_cuda", return_value=torch.device("cpu")),
        patch("atml.task1.diagnostics.starting_model", return_value=model),
        patch("atml.task1.diagnostics.tokenizer_for", return_value=tokenizer),
        patch("atml.task1.diagnostics.RECIPE", SimpleNamespace(seed=6304, diagnostic_layers=(0,))),
    ):
        cache_probes(paths)
    cached = torch.load(paths.artifacts / "baseline_activations.pt", weights_only=True)
    torch.testing.assert_close(cached["0"], torch.stack(expected), atol=2e-6, rtol=2e-5)
    assert read_json(paths.evidence / "activation_cache.json")["sha256"] == sha256(
        paths.artifacts / "baseline_activations.pt"
    )


def check_basis_resume(root):
    from atml.task1.model import build_bases
    from atml.task1.adapters import compute_bases
    from types import SimpleNamespace

    paths = Paths(root / "basis_artifacts", root / "basis_evidence")
    write_json(paths.evidence / "asset_lock.json", {"model": "synthetic-only"})
    model, _ = tiny_components()
    count = 0

    def interrupted(weight, rank):
        nonlocal count
        count += 1
        if count == 3:
            raise InterruptedError("Simulated interruption after two saved SVD modules")
        return compute_bases(weight, rank)

    with (
        patch("atml.task1.model.require_cuda", return_value=torch.device("cpu")),
        patch("atml.task1.model.starting_model", return_value=model),
        patch("atml.task1.model.RECIPE", SimpleNamespace(protection_rank=4, seed=6304)),
        patch("torch.cuda.synchronize"),
    ):
        with patch("atml.task1.model.compute_bases", side_effect=interrupted):
            try:
                build_bases(paths)
            except InterruptedError:
                pass
        with patch("atml.task1.model.compute_bases", wraps=compute_bases) as compute:
            build_bases(paths)
            assert compute.call_count == 5  # Seven targets; two previously committed.
    assert len(torch.load(paths.artifacts / "bases.pt", weights_only=True)) == 7


def check_qualitative(root):
    from types import SimpleNamespace
    from atml.common.io import read_json, write_jsonl
    from atml.task1.qualitative import qualitative, group_and_prepare_annotations

    paths = Paths(root / "qual_artifacts", root / "qual_evidence")
    _, tokenizer = tiny_components()
    prompts = [
        {
            "question_id": str(i),
            "format": form,
            "prompt": "one two",
            **({"system": "three"} if form == "json" else {}),
        }
        for i in range(8)
        for form in ("plain", "json", "template")
    ]
    write_json(
        paths.evidence / "qualitative_prompts.json",
        {
            "source_sha256": "synthetic-only",
            "prompts": prompts,
            "annotation_question_ids": ["0", "7"],
        },
    )

    def generate(input_ids, **kwargs):
        assert kwargs["max_new_tokens"] == 512 and kwargs["do_sample"] is False
        return torch.cat([input_ids, torch.tensor([[8, 2]]).repeat(len(input_ids), 1)], dim=1)

    fake = SimpleNamespace(generate=generate, generation_config=SimpleNamespace(eos_token_id=2))
    with (
        patch("atml.task1.qualitative.require_cuda", return_value=torch.device("cpu")),
        patch("atml.task1.qualitative.load_condition", return_value=fake) as load,
        patch("atml.task1.qualitative.tokenizer_for", return_value=tokenizer),
    ):
        qualitative(paths, "baseline")
        qualitative(paths, "baseline")
        assert load.call_count == 1
    rows = recover_jsonl(paths.evidence / "qualitative" / "baseline.jsonl")
    assert len(rows) == 24 and all(row["output_tokens"] == 2 for row in rows)
    assert all("system: three" in row["rendered_prompt"] for row in rows if row["format"] == "json")
    for condition in CONDITIONS:
        if condition != "baseline":
            write_jsonl(
                paths.evidence / "qualitative" / f"{condition}.jsonl",
                [{**row, "condition": condition} for row in rows],
            )
    group_and_prepare_annotations(paths)
    assert len(read_json(paths.evidence / "qualitative" / "grouped_responses.json")) == 24
    annotation = paths.evidence / "qualitative" / "annotations.csv"
    import csv

    with annotation.open() as stream:
        assert len(list(csv.DictReader(stream))) == 42
    original = annotation.read_bytes()
    group_and_prepare_annotations(paths)
    assert annotation.read_bytes() == original


def smoke(paths, cuda=False):
    seed_all(RECIPE.seed)
    previous_rng = rng_state()
    checks = []
    with tempfile.TemporaryDirectory(prefix="pa3-synthetic-smoke-") as temp:
        root = Path(temp)
        tasks = [
            ("selection, leakage, response/EOT masks", check_data),
            ("response-token loss and all parameter gradients", check_loss),
            ("adapter initialization, projections, counts and merge", check_adapters),
            (
                "unique-parameter L2-SP, AdamW groups and diagnostics",
                check_regularization_and_geometry,
            ),
            (
                "optimizer/scaler/RNG/data-position recovery and torn write",
                lambda: check_recovery(root),
            ),
            (
                "AMP overflow retries preserve RNG, data position and applied-update count",
                check_amp_retry,
            ),
            (
                "batched EOS, stock stops, padding and token cap",
                lambda: check_batched_generation(root),
            ),
            (
                "stock harness batch equivalence and persistent request replay",
                lambda: check_harness_cache(root),
            ),
            (
                "batched FP32 probe hooks equal individual last-token inputs",
                lambda: check_probe_batching(root),
            ),
            (
                "interrupted SVD preparation reuses committed modules",
                lambda: check_basis_resume(root),
            ),
            (
                "qualitative batching/resume, JSON system messages and 42-row worksheet",
                lambda: check_qualitative(root),
            ),
        ]
        if cuda:
            if not torch.cuda.is_available():
                raise RuntimeError("--cuda requested but CUDA is unavailable.")
            tasks.append(
                (
                    "CUDA FP16 batched generation",
                    lambda: check_batched_generation(root / "gpu", "cuda"),
                )
            )
        try:
            for label, execute in tasks:
                started = time.perf_counter()
                try:
                    execute()
                    checks.append(
                        {"check": label, "passed": True, "seconds": time.perf_counter() - started}
                    )
                    print(f"[smoke OK] {label}", flush=True)
                except Exception as error:
                    checks.append({"check": label, "passed": False, "error": repr(error)})
                    raise
        finally:
            write_json(
                paths.evidence / "system" / "smoke.json",
                {
                    "scope": "Synthetic fixtures and randomly initialized tiny models only; no experiment results",
                    "passed": len(checks) == len(tasks) and all(row["passed"] for row in checks),
                    "cuda_requested": cuda,
                    "training_variants": list(METHODS),
                    "evaluation_conditions": list(CONDITIONS),
                    "checks": checks,
                    "environment": environment(),
                    "temporary_fixtures": "Automatically removed on exit",
                },
            )
            restore_rng(previous_rng)
    return True
