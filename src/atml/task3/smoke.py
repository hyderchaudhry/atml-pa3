"""Offline synthetic tests; all fixtures and results are deleted on exit."""

import base64
import copy
import tempfile
from types import SimpleNamespace
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from atml.common.io import read_json, sha256, write_json
from atml.task2.smoke import MemoryHub

from .cli import next_step
from .config import DATASETS, INSTRUCTION, MODEL, NO_ANSWER, REVISION, SYSTEM, Paths, protocol
from .data import database, ingest_file, load_question, prepare, source_rows, validated_arrays
from .live import (
    CALLS,
    call_seed,
    generate_call,
    path_for,
    read_partial,
    sample_token,
    save_partial,
    strategies,
    run_live,
    live_session,
    validate_live,
)
from .offline import (
    evaluate_question,
    fit_policies,
    policy_grid,
    run_offline,
)
from .scoring import Scorer, last_boxed, normalize_mathvista
from .selection import (
    bootstrap_questions,
    choose_policy,
    quality,
    rank_weights,
    select,
    stop,
    stopping_trace,
    visible_groups,
    vote,
)
from .storage import backup, failure_details, restore, retry_upload, safe_relative
from .summary import export, summarize


def fixtures(folder, questions=10):
    source = folder / "supplied"
    source.mkdir()
    (source / "README.MD").write_text(
        "Synthetic smoke data, not experimental evidence.\n"
        "Model: `" + MODEL + "`, revision `" + "a" * 40 + "`.\n"
    )
    metadata = {}
    for dataset, name in (("math", "qwen25-math"), ("mathvista", "qwen25vl-mathvista")):
        destination = source / name
        destination.mkdir()
        generations, probability, attention = [], [], []
        for question in range(questions):
            key = f"test:{question}" if dataset == "math" else str(question + 1)
            kind = question % 3
            if dataset == "mathvista":
                metadata[key] = dict(
                    question_type="multi_choice" if kind == 0 else "free_form",
                    answer_type="text" if kind == 0 else "integer" if kind == 1 else "float",
                    choices=["one half", "three quarters", "one"] if kind == 0 else None,
                    precision=1 if kind == 2 else None,
                    answer="three quarters" if kind == 0 else "2" if kind == 1 else "1.5",
                )
            gold = "1/2" if dataset == "math" else metadata[key]["answer"]
            for rid in range(32):
                count = 4 + rid % 5
                if dataset == "math":
                    answer = ["1/2", "0.5", "2", None][rid % 4]
                elif kind == 0:
                    answer = ["B", "three quarters", "C", None][rid % 4]
                elif kind == 1:
                    answer = ["2", "2.0", "3", None][rid % 4]
                else:
                    answer = ["1.49", "1.5", "2.1", None][rid % 4]
                keys = dict(dataset=dataset, sample_id=key, rollout_id=rid)
                generations.append(
                    dict(
                        **keys,
                        model="qwen2_5_3b_instruct" if dataset == "math" else "qwen2_5_vl_3b",
                        question_text=f"Synthetic question {question} ({dataset}).",
                        generated_text=r"Synthetic reasoning. \boxed{" + str(answer or "") + "}",
                        generated_token_ids=list(range(10, 10 + count)),
                        n_generated_text_tokens=count,
                        hit_max_new_tokens=False,
                        gold_answer=gold,
                        extracted_answer=answer,
                    )
                )
                probability.append(
                    dict(
                        **keys,
                        logp=[-0.1 - 0.01 * rid - 0.001 * t for t in range(count)],
                        kl_u_p=[0.01 + 0.05 * rid + 0.03 * t for t in range(count)],
                    )
                )
                if dataset == "mathvista":
                    attention.append(
                        dict(
                            **keys,
                            n_image_tokens=196,
                            n_prompt_text_tokens=23,
                            img_attn_total=[
                                0.1 + 0.01 * (rid % 5) + 0.005 * t for t in range(count)
                            ],
                            prompt_text_attn_total=[0.3] * count,
                            generated_text_attn_total=[0.01 * t for t in range(count)],
                            n_generated_text_tokens=list(range(count)),
                        )
                    )
        for filename, rows in (
            ("generations", generations),
            ("probability_metrics", probability),
            ("attention_metrics", attention),
        ):
            if rows:
                embedded = (
                    dict(
                        model_id=MODEL,
                        model_revision="a" * 40,
                        config_sha256="b" * 64,
                        dataset_revision="c" * 40,
                    )
                    if dataset == "math"
                    else dict(
                        model_id="qwen2_5_vl_3b", model_revision="local-qwen2.5-vl-3b-instruct"
                    )
                )
                pq.write_table(
                    pa.Table.from_pylist(rows).replace_schema_metadata(
                        {k.encode(): v.encode() for k, v in embedded.items()}
                    ),
                    destination / (filename + ".parquet"),
                    row_group_size=64,
                )
    meta = folder / "metadata.json"
    write_json(meta, metadata)
    return source, meta


class TinyTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt, tokenize, return_tensors):
        assert add_generation_prompt and tokenize and return_tensors == "pt"
        text = "".join(m["role"] + ": " + m["content"] + "\n" for m in messages) + "assistant: "
        return torch.tensor([[ord(c) % 50 + 4 for c in text]], dtype=torch.long)

    def decode(self, tokens, skip_special_tokens):
        assert skip_special_tokens
        return " ".join(map(str, tokens))


def tiny_model():
    config = Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        eos_token_id=63,
        pad_token_id=0,
        max_position_embeddings=512,
        attention_dropout=0.0,
        use_sliding_window=False,
    )
    config._attn_implementation = "sdpa"
    model = Qwen2ForCausalLM(config).eval()
    # Keep EOS unreachable in the deterministic interrupted-token test.
    model.generation_config.eos_token_id = 128
    return model.requires_grad_(False)


def cpu_generation_context():
    context = ExitStack()
    original_to = torch.Tensor.to

    def move(tensor, *args, **kwargs):
        if args and args[0] == "cuda":
            args = ("cpu",) + args[1:]
        return original_to(tensor, *args, **kwargs)

    context.enter_context(patch.object(torch.Tensor, "to", move))
    context.enter_context(
        patch(
            "atml.task3.live.cuda_rng",
            side_effect=lambda: base64.b64encode(torch.get_rng_state().numpy().tobytes()).decode(),
        )
    )

    def restore_rng(encoded):
        torch.set_rng_state(
            torch.from_numpy(np.frombuffer(base64.b64decode(encoded), dtype=np.uint8).copy())
        )

    context.enter_context(patch("atml.task3.live.restore_cuda_rng", side_effect=restore_rng))
    for name in ("synchronize", "reset_peak_memory_stats"):
        context.enter_context(patch("atml.task3.live.torch.cuda." + name))
    context.enter_context(patch("atml.task3.live.torch.cuda.max_memory_allocated", return_value=0))
    return context


def fake_live(paths):
    """Synthetic completed responses exercise strategy accounting and export."""
    from .live import live_scope

    signature = live_scope(paths)
    gpu = dict(uuid="synthetic-only", name="CPU test substitute", total_memory=0)
    write_json(paths.evidence / "live_gpu.json", gpu)
    session_file = paths.evidence / "live_sessions/synthetic.json"
    write_json(session_file, dict(gpu_identity=gpu, synthetic=True))
    session = dict(
        file=session_file.relative_to(paths.evidence).as_posix(),
        sha256=sha256(session_file),
        gpu=gpu,
    )
    for key in read_json(paths.evidence / "splits.json")["live_math"]:
        problem = load_question(paths, "math", key)["generations"][0]["question_text"]
        prompt = [
            dict(role="system", content=SYSTEM),
            dict(role="user", content=problem + "\n\n" + INSTRUCTION),
        ]
        for i, kind in enumerate(CALLS):
            cap = 3072 if kind == "long" else 2048 if kind == "revision" else 1024
            answer = "1/2" if kind in ("long", "parallel_0", "parallel_1", "revision") else "2"
            messages = list(prompt)
            if kind == "revision":
                initial = read_json(path_for(paths, key, "parallel_0"))
                messages += [
                    dict(role="assistant", content=initial["generated_text"]),
                    dict(role="user", content=REVISION),
                ]
            write_json(
                path_for(paths, key, kind),
                dict(
                    scope=signature,
                    key=key,
                    kind=kind,
                    session=session,
                    generated_text=r"Synthetic \boxed{" + answer + "}",
                    generated_token_ids=[3, 4, 5],
                    raw_generated_token_ids=[3, 4, 5, 2],
                    input_token_ids=[1] * (10 + i),
                    input_tokens=10 + i,
                    output_tokens=4,
                    retained_output_tokens=3,
                    termination_tokens=1,
                    hit_max_new_tokens=False,
                    extracted_answer=answer,
                    seconds=0.1 * (i + 1),
                    peak_gpu_bytes=0,
                    seed=call_seed(key, kind),
                    cap=cap,
                    prompt=messages,
                    resumed=False,
                    gold_answer="1/2",
                    accuracy=int(answer == "1/2"),
                ),
            )
    validate_live(paths, signature)


def smoke():
    torch.set_num_threads(2)
    checks = []
    with tempfile.TemporaryDirectory(prefix="pa3-task3-smoke-") as temporary:
        base = Path(temporary)
        paths = Paths(base / "run")
        source, metadata = fixtures(base)
        # Resume an interrupted real Arrow→SQLite ingest transaction.
        interrupted = Paths(base / "interrupted")
        conn = database(interrupted)
        original = pq.ParquetFile

        class InterruptedParquet:
            def __init__(self, *a, **k):
                self.file = original(*a, **k)
                self.metadata = self.file.metadata
                self.schema_arrow = self.file.schema_arrow

            def iter_batches(self, *a, **k):
                yield next(iter(self.file.iter_batches(*a, **k)))
                raise IOError("synthetic import interruption")

        file = source / "qwen25-math/generations.parquet"
        with patch("atml.task3.data.pq.ParquetFile", InterruptedParquet):
            try:
                ingest_file(conn, file, "math", "generation", 320)
            except IOError:
                pass
            else:
                raise AssertionError("Import interruption was not exercised")
        assert conn.execute("SELECT rows FROM ingest").fetchone()[0] == 128
        ingest_file(conn, file, "math", "generation", 320)
        assert conn.execute("SELECT COUNT(*) FROM rollout").fetchone()[0] == 320
        conn.close()
        prepare(paths, source, metadata, expected_questions=10)
        before = sha256(paths.artifacts / "prepared.sqlite")
        prepare(paths, source, metadata, expected_questions=10)
        assert sha256(paths.artifacts / "prepared.sqlite") == before
        splits = read_json(paths.evidence / "splits.json")
        for dataset in DATASETS:
            expected = (
                np.random.default_rng(6304)
                .permutation(
                    sorted([f"test:{i}" if dataset == "math" else str(i + 1) for i in range(10)])
                )
                .tolist()
            )
            assert splits[dataset]["development"] == expected[:2]
            assert splits[dataset]["evaluation"] == expected[2:]
        assert splits["live_math"] == splits["math"]["evaluation"][:50]
        assert before == read_json(paths.evidence / "preparation.json")["database_sha256"]
        checks.append(
            "actual Parquet schema/join ingestion, interruption recovery, fixed split and order IDs"
        )
        with database(paths, readonly=True) as conn:
            data = source_rows(conn, "mathvista", "1")
        validated_arrays(data, "mathvista")
        damaged = copy.deepcopy(data)
        damaged[0][1]["kl_u_p"].pop()
        try:
            validated_arrays(damaged, "mathvista")
        except ValueError:
            pass
        else:
            raise AssertionError("Accepted token-array misalignment")
        assert last_boxed(r"first \boxed{1} then \boxed{\frac{2}{3}} ") == r"\frac{2}{3}"
        assert last_boxed(r"\boxed{1} final \boxed{unclosed") is None
        assert last_boxed(r"\boxed{}") is None and last_boxed(r"\boxed {3}") is None
        scorer = Scorer("math")
        assert scorer.equivalent("0.5", r"\frac{1}{2}")
        assert not scorer.equivalent("2", "3") and not scorer.equivalent(NO_ANSWER, NO_ANSWER)
        meta = read_json(metadata)
        assert normalize_mathvista("B", meta["1"]) == "B"
        assert normalize_mathvista("three quarter", meta["1"]) == "B"
        assert normalize_mathvista("2.0", meta["2"]) == "2"
        assert normalize_mathvista("1.49", meta["3"]) == "1.5"
        assert normalize_mathvista(None, meta["1"]) == NO_ANSWER
        assert "answer" not in Scorer("mathvista", meta["1"]).metadata
        checks.append(
            "strict token alignment, exact supplied boxed rule, mandated math-verify and metadata scoring"
        )
        np.testing.assert_allclose(rank_weights([3, 3, 1]), [0.75, 0.75, 0])
        np.testing.assert_allclose(rank_weights([2, 2]), [0.5, 0.5])
        np.testing.assert_allclose(rank_weights([9]), [1])
        assert vote(["b", "a"])["group"] == "b"
        assert select("certainty:rwc", ["b", "a"], {"certainty": np.array([1.0, 1.0])}) == 0
        assert visible_groups([None, "0.5", r"\frac{1}{2}", NO_ANSWER], scorer.equivalent) == [
            NO_ANSWER,
            0,
            0,
            NO_ANSWER,
        ]
        checks.append(
            "average-rank RWC weights, current-order ties, semantic grouping and one missing-answer group"
        )
        question = load_question(paths, "mathvista", "1")
        pool = question["pool"]
        expected = 0.1 / (0.1 + 0.3)
        assert np.isclose(pool.vision[0][0], expected)
        score, calibration = quality(pool, [0, 1])
        joined = np.concatenate([pool.kl[0], pool.kl[1]])
        np.testing.assert_allclose(
            [calibration["q05"], calibration["q95"]],
            np.quantile(joined, [0.05, 0.95], method="linear"),
        )
        altered = copy.deepcopy(pool)
        for i in range(2, 32):
            altered.kl[i] = np.full_like(altered.kl[i], 1e9)
        changed, _ = quality(altered, [0, 1])
        for key in score:
            np.testing.assert_array_equal(score[key], changed[key])
        from .selection import signal_diagnostics

        diagnostics = signal_diagnostics(pool, question["correct"], question["scorer"].equivalent)
        np.testing.assert_allclose(
            diagnostics["fusion_difference"], diagnostics["token_covariances"], atol=1e-12
        )
        checks.append(
            "normalized eligible attention, pooled linear percentiles, prefix-only calibration and covariance identity"
        )
        orders = read_json(paths.evidence / "orders.json")["mathvista"]["1"]
        result = evaluate_question(
            pool,
            question["correct"],
            question["scorer"].equivalent,
            "mathvista",
            orders,
            policy_grid(),
        )
        counterfactual = evaluate_question(
            pool,
            1 - question["correct"],
            question["scorer"].equivalent,
            "mathvista",
            orders,
            policy_grid(),
        )
        assert [(r["selected_id"], r["tokens"]) for r in result["fixed"]] == [
            (r["selected_id"], r["tokens"]) for r in counterfactual["fixed"]
        ]
        assert [(r["selected_id"], r["n"]) for r in result["adaptive"]] == [
            (r["selected_id"], r["n"]) for r in counterfactual["adaptive"]
        ]
        assert all(
            r["selected_id"] == r["candidate_ids"][0] for r in result["fixed"] if r["n"] == 1
        )
        for row in result["fixed"]:
            assert row["tokens"] == (
                int(pool.tokens[row["candidate_ids"][0]])
                if row["method"] == "first"
                else int(pool.tokens[row["candidate_ids"]].sum())
            )
            assert row["sampling_failure"] + row["selection_failure"] == 1 - row["accuracy"]
        missing = copy.deepcopy(pool)
        missing.answers = [NO_ANSWER] * 32
        trace = stopping_trace(missing, np.arange(32), lambda a, b: a == b, "mathvista")
        assert stop(trace, "agreement", tau=0.5)["n"] == 32
        assert stop(trace, "quality", tau=0.5, gamma=0)["n"] == 32
        candidates = [
            dict(id="costly", tokens=100, accuracy=0.9, tau=0.5),
            dict(id="cheap", tokens=50, accuracy=0.89, tau=0.75),
            dict(id="bad", tokens=1, accuracy=0.88, tau=1.0),
        ]
        assert choose_policy(candidates, 0.9)["id"] == "cheap"
        assert bootstrap_questions([0, 1, 1]) == bootstrap_questions([0, 1, 1])
        assert bootstrap_questions(np.zeros((5, 2)))["high"] == [0.0, 0.0]
        checks.append(
            "gold-independent selection/stopping, N=1, consumed-token costs, failures and question bootstrap"
        )
        assert next_step(paths) == "development"
        run_offline(paths, "development", workers=2)
        fit_policies(paths)
        frozen = sha256(paths.evidence / "policies.json")
        assert next_step(paths) == "evaluation"
        run_offline(paths, "evaluation", workers=1)
        with patch(
            "atml.task3.offline.evaluate_question",
            side_effect=AssertionError("Recomputed completed question"),
        ):
            run_offline(paths, "evaluation", workers=1)
        fit_policies(paths)
        assert sha256(paths.evidence / "policies.json") == frozen
        assert next_step(paths) == "live"
        checks.append(
            "real development→freeze→held-out workflow and per-question resume without recomputation"
        )
        torch.manual_seed(6304)
        model, tokenizer = tiny_model(), TinyTokenizer()
        prompt = [
            dict(role="system", content="Synthetic system"),
            dict(role="user", content="Synthetic math"),
        ]
        signature = dict(synthetic=True)
        baseline_path = base / "baseline.partial.json"
        recovery_path = base / "recovery.partial.json"
        with cpu_generation_context():
            baseline = generate_call(
                model, tokenizer, prompt, 6, 123, baseline_path, signature, interval=2
            )
            actual_save = save_partial

            def interrupted_save(path, state):
                actual_save(path, state)
                raise IOError("synthetic token interruption")

            with patch("atml.task3.live.save_partial", side_effect=interrupted_save):
                try:
                    generate_call(
                        model, tokenizer, prompt, 6, 123, recovery_path, signature, interval=2
                    )
                except IOError:
                    pass
                else:
                    raise AssertionError("Partial-token interruption not exercised")
            assert len(read_partial(recovery_path)["tokens"]) == 2
            resumed = generate_call(
                model, tokenizer, prompt, 6, 123, recovery_path, signature, interval=2
            )
            assert baseline["raw_generated_token_ids"] == resumed["raw_generated_token_ids"]
            assert resumed["resumed"] and resumed["hit_max_new_tokens"]
            model.generation_config.eos_token_id = 2
            with patch("atml.task3.live.sample_token", return_value=torch.tensor(2)):
                eos = generate_call(
                    model,
                    tokenizer,
                    prompt,
                    6,
                    123,
                    base / "eos.partial.json",
                    signature,
                    interval=2,
                )
            assert (
                eos["retained_output_tokens"] == 0
                and eos["output_tokens"] == 1
                and not eos["hit_max_new_tokens"]
            )
        wrong = read_json(recovery_path)
        wrong["state"]["tokens"].append(1)
        write_json(recovery_path, wrong)
        try:
            read_partial(recovery_path)
        except IOError:
            pass
        else:
            raise AssertionError("Accepted corrupt token recovery checkpoint")
        assert len({call_seed(str(i), kind) for i in range(50) for kind in CALLS}) == 250
        checks.append(
            "tiny Qwen2 cached sampling, exact interrupted-token RNG continuation, EOS/empty outputs and checksum protection"
        )
        # Sampling filters match an independently calculated top-k/nucleus distribution.
        logits = torch.tensor([1.0, 4.0, 2.0, 3.0, 0.0])
        probabilities = []
        with patch(
            "torch.multinomial",
            side_effect=lambda p, n: (probabilities.append(p.clone()) or torch.tensor([0])),
        ):
            sampled = sample_token(logits, temperature=1.0, top_k=3, top_p=0.8)
        expected = torch.softmax(torch.tensor([4.0, 3.0, 2.0]), 0)
        expected[-1] = 0
        expected /= expected.sum()
        assert sampled == 1
        torch.testing.assert_close(probabilities[0], expected)
        checks.append(
            "temperature/top-k/nucleus sampling distribution and independently seeded call identities"
        )
        # Exercise the real 50-question orchestrator without fetching a model or GPU.
        import shutil

        runner = Paths(base / "orchestrator")
        runner.evidence.mkdir(parents=True)
        for name in ("preparation.json", "asset_lock.json"):
            shutil.copyfile(paths.evidence / name, runner.evidence / name)
        live_keys = [f"test:{i}" for i in range(50)]
        write_json(runner.evidence / "splits.json", dict(live_math=live_keys))

        class FakeModel:
            def to(self, device):
                assert device == "cuda"
                return self

            def eval(self):
                return self

            def requires_grad_(self, value):
                assert value is False

        calls = []

        def mocked_generate(model, tokenizer, messages, cap, seed, partial, signature):
            calls.append((messages, cap, seed))
            return dict(
                generated_text=r"Synthetic \boxed{1/2}",
                extracted_answer="1/2",
                generated_token_ids=[4],
                raw_generated_token_ids=[4, 2],
                input_token_ids=[1, 3],
                input_tokens=2,
                output_tokens=2,
                retained_output_tokens=1,
                termination_tokens=1,
                hit_max_new_tokens=False,
                seconds=0.1,
                peak_gpu_bytes=0,
                seed=seed,
                cap=cap,
                prompt=messages,
                resumed=False,
            )

        def fake_question(*args):
            return dict(generations=[dict(question_text="Synthetic", gold_answer="1/2")])

        with ExitStack() as context:
            context.enter_context(patch("atml.task3.live.require_cuda"))
            properties = SimpleNamespace(uuid="smoke-uuid", name="Synthetic", total_memory=0)
            context.enter_context(
                patch("torch.cuda.get_device_properties", return_value=properties)
            )
            context.enter_context(patch("atml.task3.live.load_question", side_effect=fake_question))
            context.enter_context(
                patch("transformers.AutoTokenizer.from_pretrained", return_value=object())
            )
            loader = context.enter_context(
                patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=FakeModel())
            )
            context.enter_context(
                patch("atml.task3.live.generate_call", side_effect=mocked_generate)
            )

            def interrupted_backup():
                raise IOError("synthetic disconnect after committed question")

            try:
                run_live(runner, interrupted_backup)
            except IOError:
                pass
            else:
                raise AssertionError("Live orchestrator interruption was not exercised")
            assert len(calls) == 5
            run_live(runner)
            assert len(calls) == 250 and loader.call_count == 2
            run_live(runner)
            assert len(calls) == 250 and loader.call_count == 2
            assert [c[1] for c in calls[:5]] == [3072, 1024, 1024, 1024, 2048]
            assert calls[4][0][-2]["content"] == r"Synthetic \boxed{1/2}"
            assert calls[4][0][-1]["content"] == REVISION
            assert all("1/2" not in c[0][1]["content"] for c in calls)
            with patch(
                "torch.cuda.get_device_properties",
                return_value=SimpleNamespace(uuid="different", name="Synthetic", total_memory=0),
            ):
                try:
                    live_session(runner)
                except ValueError:
                    pass
                else:
                    raise AssertionError("Accepted mixing live GPUs")
        checks.append(
            "250-call live orchestration, exact prompts/caps/shared initial, completed-call resume and same-GPU enforcement"
        )
        fake_live(paths)
        key = splits["live_math"][0]
        values = {r["strategy"]: r for r in strategies(paths, key)}
        assert values["parallel"]["output_tokens"] == 12
        assert values["refinement"]["output_tokens"] == 8
        assert np.isclose(values["refinement"]["seconds"], 0.2 + 0.5)
        assert values["parallel"]["accuracy"] == 1 and values["refinement"]["accuracy"] == 1
        result = summarize(paths)
        assert not result[
            "computational_complete"
        ]  # ten synthetic questions cannot pass the real 1000-question audit
        archive = export(paths)
        import zipfile

        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None and "task3/tables/live_summary.csv" in z.namelist()
            assert not any(
                "prepared.sqlite" in name or "supplied/" in name for name in z.namelist()
            )
        assert (paths.evidence / "plots/sampling_selection.png").exists()
        assert (paths.evidence / "qualitative.html").exists()
        checks.append(
            "live shared-initial accounting, strategy results, headless figures/tables/gallery/ZIP and honest completeness audit"
        )
        hub = MemoryHub(base / "mock-remote")
        from requests.exceptions import Timeout

        retry_operation = Mock(
            side_effect=[Timeout("synthetic upload timeout"), "recovered"]
        )
        with patch("atml.task3.storage.time.sleep") as sleep:
            assert retry_upload("synthetic retry", retry_operation) == "recovered"
            sleep.assert_called_once_with(2)
        assert retry_operation.call_count == 2
        denied = RuntimeError("synthetic denied upload")
        denied.response = SimpleNamespace(status_code=403)
        assert not failure_details(denied)["transient"]
        with (
            patch("atml.task3.storage.HfApi", return_value=hub),
            patch("atml.task3.storage.hf_hub_download", side_effect=hub.download),
        ):
            hub.fail = True
            try:
                backup(paths, "synthetic/task3")
            except IOError:
                pass
            else:
                raise AssertionError("Accepted failed backup")
            assert not (paths.artifacts / "backup_receipt.json").exists()
            hub.fail = False
            hub.corrupt = True
            try:
                backup(paths, "synthetic/task3")
            except IOError:
                pass
            else:
                raise AssertionError("Accepted corrupt remote file")
            hub.corrupt = False
            backup(paths, "synthetic/task3")
            previous = hub.number
            backup(paths, "synthetic/task3")
            assert hub.number == previous + 1
            restored = Paths(base / "restored")
            restore(restored, "synthetic/task3")
            assert sha256(restored.artifacts / "prepared.sqlite") == before
            assert sha256(restored.evidence / "policies.json") == frozen
        for name in ("../private", "/etc/passwd", "evidence/task2/status.json"):
            try:
                safe_relative(name)
            except ValueError:
                pass
            else:
                raise AssertionError("Accepted unsafe/cross-task recovery path")
        checks.append(
            "incremental immutable-commit backup/restore, failed-upload/hash handling and Task 3 path isolation"
        )
        assert protocol()["live"]["batch_size"] == 1
        assert protocol()["live"]["caps"] == dict(long=3072, parallel=1024, revision=2048)
    assert not base.exists()
    for item in checks:
        print("PASS", item)
    print(f"[smoke] {len(checks)} groups passed; all synthetic fixtures and outputs removed")


if __name__ == "__main__":
    smoke()
