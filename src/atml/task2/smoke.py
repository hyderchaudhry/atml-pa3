"""Offline synthetic verification. Temporary fixtures are deleted on every exit."""

import copy
import contextlib
import hashlib
import io
import math
import random
import tempfile
import zipfile
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM, SiglipVisionConfig, SiglipVisionModel

from atml.common.io import (
    read_json,
    read_jsonl,
    restore_rng,
    rng_state,
    seed_all,
    write_json,
    write_jsonl,
)

from .config import CONDITIONS, INSTRUCTION, Paths, STAGES, Stage, protocol, stage_at
from .cli import next_step
from .data import (
    Sampler,
    assert_isolation,
    fix_mismatches,
    grounding_pairs,
    select_captions,
    select_natural,
)
from .evaluation import evaluate, journal_scope
from .features import pool_patches, save_tensor
from .metrics import bootstrap, normalize, soft_score
from .model import LoRALinear, VisualModel, enable_checkpointing
from .storage import backup, restore, safe_relative
from .summary import export, summarize
from .training import load_checkpoint, lr_factor, save_checkpoint, scope, train_segment


class TinyTokenizer:
    """Synthetic byte-level chat tokenizer; never downloads a real tokenizer."""

    pad_token_id = 0
    unk_token_id = -1

    def convert_tokens_to_ids(self, token):
        return {"<|im_end|>": 2}[token]

    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 120 + 8 for c in text]

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(i) for i in ids if i > 2)

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        assert not enable_thinking
        text = "".join(
            "<|im_start|>" + m["role"] + "\n" + m["content"] + "<|im_end|>\n" for m in messages
        )
        text += "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        return self.encode(text)


def tiny_model():
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        eos_token_id=2,
        pad_token_id=0,
        attention_dropout=0.0,
    )
    config._attn_implementation = "sdpa"
    language = Qwen3ForCausalLM(config)
    enable_checkpointing(language)
    return VisualModel(language, TinyTokenizer(), feature_width=8, hidden_width=32)


def sample_rows():
    return [
        dict(
            id="a",
            image_id=1,
            split="train2014",
            component="presence",
            question="Cat?",
            target="yes",
        ),
        dict(
            id="b",
            image_id=2,
            split="train2014",
            component="presence",
            question="Two cats?",
            target="no",
        ),
    ]


class MemoryHub:
    """Offline content-addressed Hub fake: verifies our commit/restore protocol."""

    def __init__(self, folder):
        self.folder = folder
        self.files = {}
        self.commits = {}
        self.number = 0
        self.fail = False
        self.corrupt = False

    def create_repo(self, *args, **kwargs):
        return None

    def repo_info(self, *args, revision=None, **kwargs):
        data = self.commits[revision] if revision else self.files
        siblings = []
        for name, content in data.items():
            blob = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
            siblings.append(
                SimpleNamespace(
                    rfilename=name,
                    size=len(content),
                    lfs=None,
                    blob_id="bad" if self.corrupt else blob,
                )
            )
        return SimpleNamespace(private=True, siblings=siblings)

    def commit(self):
        self.number += 1
        revision = str(self.number)
        self.commits[revision] = dict(self.files)
        return SimpleNamespace(oid=revision)

    def create_commit(self, *args, operations, **kwargs):
        if self.fail:
            raise IOError("simulated network failure")
        for operation in operations:
            self.files[operation.path_in_repo] = Path(operation.path_or_fileobj).read_bytes()
        return self.commit()

    def upload_file(self, *, path_in_repo, path_or_fileobj, **kwargs):
        self.files[path_in_repo] = path_or_fileobj.getvalue()
        return self.commit()

    def download(self, repo, filename, revision=None, **kwargs):
        content = (self.commits[revision] if revision else self.files)[filename]
        target = self.folder / hashlib.sha256(content).hexdigest()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        return str(target)


def smoke():
    seed_all(6304)
    torch.set_num_threads(2)
    checks = []
    with tempfile.TemporaryDirectory(prefix="pa3-task2-smoke-") as temporary:
        paths = Paths(Path(temporary))
        model = tiny_model()
        original_language = copy.deepcopy(model.language.state_dict())
        rows = sample_rows()
        features = [torch.randn(4, 8), torch.randn(4, 8)]
        model.train()
        batch = model.build(rows, features)
        labels = batch.pop("labels")[:, 1:]
        dense = model.language(**batch, use_cache=False).logits[:, :-1].float()
        token = F.cross_entropy(
            dense.reshape(-1, 128), labels.reshape(-1), ignore_index=-100, reduction="none"
        ).reshape_as(labels)
        expected = token.sum(-1) / (labels != -100).sum(-1)
        actual = model.losses(rows, features)
        torch.testing.assert_close(actual, expected)
        actual.mean().backward()
        assert all(p.grad is None for p in model.language.parameters())
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in model.connector.parameters()
        )
        # Different target lengths distinguish example weighting from token weighting.
        assert (labels != -100).sum(-1).tolist() == [4, 3]
        reference_labels = model.build(rows, features, include_eot=False)["labels"]
        assert (reference_labels != -100).sum(-1).tolist() == [3, 2]
        checks.append(
            "causal answer/EOT masking, per-example loss, frozen Qwen, connector gradients"
        )
        visual = model.build(rows, features, targets=False)
        text = model.build(rows, [None, None], targets=False)
        assert torch.equal(visual["attention_mask"].sum(-1) - 4, text["attention_mask"].sum(-1))
        for mask, positions in zip(text["attention_mask"], text["position_ids"]):
            assert positions[mask.bool()].tolist() == list(range(int(mask.sum())))
        before, after = model.prompt_parts(rows[0])
        assert after[: len(model.tokenizer.encode(INSTRUCTION))] == model.tokenizer.encode(
            INSTRUCTION
        )
        assert len(before) > 0
        checks.append("visual placement inside native chat and no-image masks/positions")
        model.eval()
        baseline = model.losses(rows, features).detach()
        model.enable_adapters()
        torch.testing.assert_close(model.losses(rows, features), baseline)
        layers = [m for m in model.modules() if isinstance(m, LoRALinear)]
        assert len(layers) == 8
        assert all(p.dtype == torch.float32 for p in model.parameters() if p.requires_grad)
        assert all(
            name.endswith((".a", ".b"))
            for name, p in model.language.named_parameters()
            if p.requires_grad
        )
        model.train()
        model.zero_grad()
        model.losses(rows, features).mean().backward()
        assert any(m.b.grad is not None and m.b.grad.abs().sum() > 0 for m in layers)
        checks.append(
            "zero-initialized own LoRA, four attention targets, FP32 trainables and gradients"
        )
        model.eval()
        batched = model.generate(rows, features, max_new=4)
        singles = [model.generate([r], [f], max_new=4)[0] for r, f in zip(rows, features)]
        assert [r["token_ids"] for r in batched] == [r["token_ids"] for r in singles]
        batched = model.generate(rows, [None, None], max_new=4)
        singles = [model.generate([r], [None], max_new=4)[0] for r in rows]
        assert [r["token_ids"] for r in batched] == [r["token_ids"] for r in singles]
        checks.append(
            "left-padded KV-cache greedy generation equals individual decoding, all controls"
        )
        # Exercise the pinned vision API and prove last_hidden_state is normalized exactly once.
        vision = SiglipVisionModel(
            SiglipVisionConfig(
                hidden_size=16,
                intermediate_size=24,
                num_hidden_layers=1,
                num_attention_heads=4,
                image_size=224,
                patch_size=16,
            )
        )
        calls = []
        handle = vision.vision_model.post_layernorm.register_forward_hook(
            lambda *_: calls.append(1)
        )
        with torch.no_grad():
            patches = vision(torch.randn(1, 3, 224, 224)).last_hidden_state
        handle.remove()
        assert patches.shape == (1, 196, 16) and len(calls) == 1
        feature = torch.arange(196).float()[:, None].expand(196, 768)
        pooled = pool_patches(feature)
        assert pooled.shape == (49, 768) and pooled[0, 0] == (0 + 1 + 14 + 15) / 4
        checks.append("SigLIP final norm once and spatial patch order/pooling")
        assert normalize("The TWO cats, 1,000.5!") == "2 cats 1000.5"
        assert normalize("Dont") == "don't"
        for n, score in ((0, 0), (1, 0.3), (2, 0.6), (3, 0.9), (4, 1)):
            assert math.isclose(soft_score("yes", ["yes"] * n + ["no"] * (10 - n)), score)
        a = np.arange(8).reshape(4, 2).astype(float)
        assert bootstrap(a) == bootstrap(a)
        assert bootstrap(np.zeros_like(a))["low"] == [0, 0]
        checks.append("VQA normalization, all ten leave-one-out consensus votes, paired bootstrap")
        assert normalize("were well hell ill") == "were well hell ill"
        assert normalize("couldve couldnt've") == "could've couldn't've"
        rng = random.Random(6304)
        grouped = {i: [dict(caption_id=i * 3 + j, image_id=i) for j in range(3)] for i in range(20)}
        captions = select_captions(grouped, 6, 3, rng)
        assert len(captions) == 18 and len({r["image_id"] for r in captions}) == 6
        candidates = [
            [dict(pair_id=str(i), answer_type=kind, image_id=i * 2 + j) for j in range(2)]
            for i, kind in enumerate(["yes/no", "number", "other"] * 4)
        ]
        used = set()
        natural_a = select_natural(candidates, {"yes/no": 1, "number": 1, "other": 1}, rng, used)
        natural_b = select_natural(candidates, {"yes/no": 1, "number": 1, "other": 1}, rng, used)
        assert not {r["image_id"] for p in natural_a for r in p} & {
            r["image_id"] for p in natural_b for r in p
        }
        from collections import Counter, defaultdict

        counts = defaultdict(Counter, {i: Counter({1: i % 5}) for i in range(100)})
        index = (list(range(100)), {1: "person"}, counts, defaultdict(set))
        used = set()
        presence = grounding_pairs(index, "val2014", "presence", 4, rng, used=used)
        counting = grounding_pairs(index, "val2014", "count", 4, rng, used=used)
        assert all({r["target"] for r in pair} == {"yes", "no"} for pair in presence)
        assert all(pair[0]["target"] != pair[1]["target"] for pair in counting)
        assert len(used) == 16
        for pair in presence + counting:
            assert pair[0]["question"] == pair[1]["question"]
        fix_mismatches(presence + counting, rng)
        for pair in presence + counting:
            for row in pair:
                assert row["mismatch_image_id"] not in {r["image_id"] for r in pair}
                assert row["mismatch_pair_id"] != row["pair_id"]
        synthetic = dict(
            sets={
                "grounding_dev": presence,
                "grounding_test": counting,
                "caption_train": [dict(image_id=1000, split="train2014")],
            }
        )
        assert_isolation(synthetic)
        broken = copy.deepcopy(synthetic)
        broken["sets"]["caption_train"][0]["image_id"] = presence[0][0]["image_id"]
        try:
            assert_isolation(broken)
            raise AssertionError("Accepted train/dev leakage")
        except AssertionError as error:
            assert str(error) != "Accepted train/dev leakage"
        checks.append(
            "seeded source selection, quotas, grounding labels, split isolation and fixed mismatches"
        )
        sets = {
            "caption_train": [dict(id=f"c{i}", component="caption") for i in range(32)],
            "presence_train": [[dict(id="p1"), dict(id="p2")]],
            "count_train": [[dict(id="k1"), dict(id="k2")]],
            "natural_train": [[dict(id="n1"), dict(id="n2")]],
        }
        sampler = Sampler(dict(sets=sets))
        epoch = []
        for _ in range(4):
            batch_rows, _ = sampler.batch(STAGES[0])
            epoch.extend(r["id"] for r in batch_rows)
        assert all(epoch[:32].count(f"c{i}") == 1 for i in range(32))
        assert all(epoch[32:].count(f"c{i}") == 1 for i in range(32))
        saved = sampler.state()
        for stage in STAGES[1:]:
            resumed = Sampler(dict(sets=sets), saved)
            uninterrupted = Sampler(dict(sets=sets), saved)
            for _ in range(10):
                a, counts = resumed.batch(stage)
                assert (a, counts) == uninterrupted.batch(stage)
                assert len(a) == 16 and sum(counts.values()) == 16
                for x, y in zip(a[::2], a[1::2]):
                    assert x["id"][0] == y["id"][0]
        assert [stage_at(i).number for i in (0, 2249, 2250, 3749, 3750, 4499)] == [1, 1, 2, 2, 3, 3]
        for stage in STAGES:
            total = stage.end - stage.start
            assert lr_factor(math.ceil(total * 0.05), total) == 1
            assert math.isclose(lr_factor(total, total), 0.1)
        checks.append(
            "two caption passes, exact sampler resume, intact pair draws and stage schedules"
        )
        # Actual optimizer/RNG/scaler-shaped checkpoint roundtrip; no experimental evidence survives.
        optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001)
        model.train()
        model.losses(rows, features).mean().backward()
        optim.step()
        write_json(paths.manifest, dict(test=True))
        write_json(paths.evidence / "asset_lock.json", dict(test=True))
        write_json(paths.artifacts / "features/index.json", dict(test=True))
        state = dict(
            update=250,
            stage=1,
            scope=scope(paths),
            weights=model.trainable_state(),
            history=[dict(update=250)],
            mixture={"caption": 4000},
            training_seconds=1,
            optimizer=optim.state_dict(),
            rng=rng_state(),
            sampler=saved,
        )
        save_checkpoint(paths, 250, state)
        checkpoint = load_checkpoint(paths, 250)
        assert checkpoint["sampler"] == state["sampler"]
        clone = tiny_model()
        clone.language.load_state_dict(original_language)
        clone.enable_adapters()
        clone.load_trainables(checkpoint["weights"])
        model.eval()
        clone.eval()
        torch.testing.assert_close(clone.losses(rows, features), model.losses(rows, features))
        # Continue both branches with identical dropout RNG and saved optimizer moments.
        resumed_optimizer = torch.optim.AdamW(
            [p for p in clone.parameters() if p.requires_grad], lr=0.001
        )
        resumed_optimizer.load_state_dict(checkpoint["optimizer"])
        for branch, optimizer in ((model, optim), (clone, resumed_optimizer)):
            restore_rng(checkpoint["rng"])
            branch.train()
            optimizer.zero_grad(set_to_none=True)
            branch.losses(rows, features).mean().backward()
            optimizer.step()
        for name, value in model.trainable_state().items():
            torch.testing.assert_close(value, clone.trainable_state()[name], rtol=0, atol=0)
        restore_rng(checkpoint["rng"])
        draws = torch.rand(8)
        restore_rng(checkpoint["rng"])
        torch.testing.assert_close(torch.rand(8), draws)
        try:
            save_checkpoint(paths, 250, state)
            raise AssertionError("Overwrote a checkpoint")
        except FileExistsError:
            pass
        checks.append(
            "atomic checkpoint, weights/optimizer/RNG/sampler roundtrip; overwrite refusal"
        )
        with patch("atml.task2.features.torch.save", side_effect=IOError("disk interrupted")):
            try:
                save_checkpoint(paths, 500, {**state, "update": 500})
                raise AssertionError("Committed an incomplete checkpoint")
            except IOError:
                assert not paths.checkpoint(500).exists()
        checks.append("checkpoint interruption leaves previous committed state intact")
        hub = MemoryHub(paths.root / "mock-remote")
        with (
            patch("atml.task2.storage.HfApi", return_value=hub),
            patch("atml.task2.storage.hf_hub_download", side_effect=hub.download),
        ):
            hub.fail = True
            try:
                backup(paths, "synthetic/task2")
                raise AssertionError("Accepted a failed upload")
            except IOError:
                assert not (paths.artifacts / "backup_receipt.json").exists()
                assert paths.checkpoint(250).exists()
            hub.fail, hub.corrupt = False, True
            try:
                backup(paths, "synthetic/task2")
                raise AssertionError("Accepted mismatched remote checksum")
            except IOError:
                assert not (paths.artifacts / "backup_receipt.json").exists()
            hub.corrupt = False
            backup(paths, "synthetic/task2")
            receipt = read_json(paths.artifacts / "backup_receipt.json")
            assert all("revision" in row for row in receipt["files"].values())
            uploads = hub.number
            backup(paths, "synthetic/task2")
            assert (
                hub.number == uploads + 1
            )  # inventory only; unchanged payloads are not uploaded again
            destination = Paths(paths.root / "restored")
            restore(destination, "synthetic/task2")
            restored_state = load_checkpoint(destination, 250)
            assert restored_state["sampler"] == checkpoint["sampler"]
            destination.manifest.write_text('{"conflict":true}')
            try:
                restore(destination, "synthetic/task2")
                raise AssertionError("Overwrote conflicting local data")
            except FileExistsError:
                pass
        checks.append(
            "incremental private backup/restore, immutable revisions, upload/hash/conflict failures"
        )
        journal = paths.evidence / "fake.jsonl"
        journal.write_bytes(b'{"id":"a"}\n{"id":')
        assert len(journal_scope(journal, dict(test=1))) == 1
        try:
            journal_scope(journal, dict(test=2))
            raise AssertionError("Reused another run's inference cache")
        except ValueError:
            pass
        for bad in ("../secret", "/etc/passwd", "artifacts/task1/model.pt"):
            try:
                safe_relative(bad)
                raise AssertionError("Accepted unsafe backup path")
            except ValueError:
                pass
        checks.append(
            "crash-tail recovery, evaluation scope binding, Task 2-only safe backup paths"
        )
        # Exercise the actual final-evaluation loop on CPU with synthetic features.
        pipeline = Paths(paths.root / "pipeline")
        first = [
            dict(
                **row,
                pair_id="pair1",
                answer_type="presence",
                canonical=row["target"],
                category_id=1,
                partner_image_id=rows[1 - i]["image_id"],
                split="val2014",
                mismatch_image_id=3 + i,
                mismatch_pair_id="pair2",
            )
            for i, row in enumerate([{k: v for k, v in r.items() if k != "split"} for r in rows])
        ]
        for row in first:
            row["question"] = "Does this image contain a cat?"
        second = [
            {
                **row,
                "id": row["id"] + "2",
                "pair_id": "pair2",
                "image_id": 3 + i,
                "partner_image_id": 4 - i,
                "mismatch_image_id": i + 1,
                "mismatch_pair_id": "pair1",
            }
            for i, row in enumerate(first)
        ]
        natural = [
            [
                {
                    **r,
                    "id": r["id"] + "n",
                    "pair_id": r["pair_id"] + "n",
                    "image_id": r["image_id"] + 10,
                    "partner_image_id": r["partner_image_id"] + 10,
                    "mismatch_image_id": r["mismatch_image_id"] + 10,
                    "references": [r["target"]] * 10,
                    "component": "natural",
                    "answer_type": "yes/no",
                }
                for r in pair
            ]
            for pair in (first, second)
        ]
        final_sets = dict(
            grounding_dev=[],
            natural_dev=[],
            caption_dev=[],
            caption_train=[],
            natural_train=[],
            grounding_test=[first, second],
            natural_test=natural,
        )
        write_json(pipeline.manifest, dict(sets=final_sets))
        save_tensor(pipeline.checkpoint(4500) / "weights.pt", model.trainable_state())
        write_json(pipeline.artifacts / "features/index.json", dict(complete=True))

        class SyntheticFeatures:
            def __init__(self, *_):
                pass

            def get(self, row, condition):
                if condition == "no_image":
                    return None
                image = row[
                    {
                        "correct": "image_id",
                        "complementary": "partner_image_id",
                        "mismatched": "mismatch_image_id",
                    }[condition]
                ]
                return torch.full((4, 8), float(image) / 100)

        model.eval()
        with (
            patch("atml.task2.evaluation.require_cuda"),
            patch("atml.task2.evaluation.load_checkpoint", return_value=model.trainable_state()),
            patch("atml.task2.evaluation.load_model", return_value=model),
            patch("atml.task2.evaluation.FeatureCache", SyntheticFeatures),
            patch(
                "atml.task2.evaluation.torch.autocast", side_effect=lambda *_a, **_k: nullcontext()
            ),
            patch("atml.task2.evaluation.torch.cuda.reset_peak_memory_stats"),
            patch("atml.task2.evaluation.torch.cuda.max_memory_allocated", return_value=0),
        ):
            evaluate(pipeline, 4500, final=True)
            generated = model.generate
            with patch.object(
                model, "generate", side_effect=AssertionError("Repeated finished generation")
            ):
                evaluate(pipeline, 4500, final=True)
            assert model.generate == generated
        for condition in CONDITIONS:
            scored = read_jsonl(pipeline.evidence / f"final/grounding_test_{condition}.jsonl")
            assert len(scored) == 4
            assert all(r["target"] == ("yes" if r["id"].startswith("a") else "no") for r in scored)
            assert (
                all(r["input_image_id"] is None for r in scored)
                if condition == "no_image"
                else True
            )
        # Summaries must stay headless even if a notebook sets an unavailable inline backend.
        write_jsonl(pipeline.evidence / "training.jsonl", [dict(update=1, loss=1.0)])
        result = summarize(pipeline)
        assert not result[
            "computational_complete"
        ]  # tiny fixtures cannot masquerade as real results
        assert (pipeline.evidence / "plots/learning_curves.png").exists()
        assert (pipeline.evidence / "tables/final.csv").exists()
        assert (pipeline.evidence / "tables/correct_image_advantages.csv").exists()
        archive = export(pipeline)
        with zipfile.ZipFile(archive) as z:
            assert "task2/tables/final.csv" in z.namelist()
            assert not any("artifacts/" in n for n in z.namelist())
        assert (pipeline.evidence / "qualitative.html").exists()
        checks.append(
            "final loop/scoring/resume, headless tables/curves/gallery/ZIP, incomplete-run audit"
        )
        workflow = Paths(paths.root / "workflow")
        assert next_step(workflow) == ("prepare", None)
        write_json(workflow.manifest, {})
        assert next_step(workflow) == ("features", None)
        write_json(workflow.artifacts / "features/index.json", dict(complete=True))
        assert next_step(workflow) == ("initialize", None)
        with patch("atml.task2.training.latest", return_value=500):
            assert next_step(workflow) == ("develop", 0)
            write_json(workflow.evidence / "development/update_0000/complete.json", {})
            assert next_step(workflow) == ("develop", 500)
            write_json(workflow.evidence / "development/update_0500/complete.json", {})
            assert next_step(workflow) == ("train", 750)
        checks.append(
            "workflow prioritizes missing development evaluations before advancing training"
        )

        # Drive the actual training loop across one synthetic step in each stage,
        # including a forced overflow/retry; no pretrained model or real examples.
        class RetryScaler:
            def __init__(self, *_):
                self.value, self.skip = 8.0, True

            def scale(self, loss):
                return loss * self.value

            def unscale_(self, optimizer):
                for group in optimizer.param_groups:
                    for parameter in group["params"]:
                        if parameter.grad is not None:
                            parameter.grad.div_(self.value)

            def get_scale(self):
                return self.value

            def step(self, optimizer):
                if not self.skip:
                    optimizer.step()

            def update(self):
                if self.skip:
                    self.value /= 2
                    self.skip = False

            def state_dict(self):
                return dict(value=self.value, skip=self.skip)

        class TrainingFeatures(SyntheticFeatures):
            def get(self, row, condition="correct"):
                return super().get(row, condition)

            def preload_cuda(self):
                return dict(feature_gpu_bytes=0, feature_transfer_seconds=0)

        for number, update in ((1, 249), (2, 2499), (3, 3999)):
            trial = Paths(paths.root / f"training-stage-{number}")
            captions = [{**rows[i % 2], "component": "caption", "caption_id": i} for i in range(32)]
            tiny_sets = {
                **sets,
                "caption_train": captions,
                "presence_train": [[dict(r) for r in rows]],
                "count_train": [[dict(r) for r in rows]],
                "natural_train": [[dict(r) for r in rows]],
            }
            write_json(trial.manifest, dict(sets=tiny_sets))
            write_json(trial.evidence / "asset_lock.json", {"synthetic": True})
            write_json(trial.artifacts / "features/index.json", {"synthetic": True})
            reference_model = tiny_model()
            if number == 3:
                reference_model.enable_adapters()
            initial_sampler = Sampler(dict(sets=tiny_sets))
            initial_sampler.caption_order = list(range(32))
            initial_sampler.position = update * 16
            initial = dict(
                update=update,
                stage=number - 1,
                scope=scope(trial),
                weights=reference_model.trainable_state(),
                sampler=initial_sampler.state(),
                rng=rng_state(),
                optimizer=None,
                scheduler=None,
                scaler=None,
                history=[],
                mixture={},
                training_seconds=0,
                stage_resources={},
            )
            save_checkpoint(trial, update, initial)

            def factory(*_a, adapters=False):
                result = tiny_model()
                if number == 3:
                    result.enable_adapters()
                return result

            base = STAGES[number - 1]
            synthetic_stage = Stage(
                number,
                update,
                update + 1,
                base.connector_lr,
                base.adapter_lr,
                base.components,
                base.probabilities,
            )
            with (
                patch("atml.task2.training.require_cuda"),
                patch("atml.task2.training.load_model", side_effect=factory),
                patch("atml.task2.training.FeatureCache", TrainingFeatures),
                patch("atml.task2.training.stage_at", return_value=synthetic_stage),
                patch("atml.task2.training.torch.amp.GradScaler", RetryScaler),
                patch(
                    "atml.task2.training.torch.autocast",
                    side_effect=lambda *_a, **_k: nullcontext(),
                ),
                patch("atml.task2.training.torch.cuda.reset_peak_memory_stats"),
                patch("atml.task2.training.torch.cuda.max_memory_allocated", return_value=0),
                patch("atml.task2.training.torch.cuda.synchronize"),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                train_segment(trial)
            trained = load_checkpoint(trial, update + 1)
            assert trained["update"] == update + 1
            assert trained["history"][0]["amp_retries"] == 1
            assert trained["sampler"]["position"] == update * 16 + 16
            assert sum(trained["mixture"].values()) == 16
            assert len(trained["optimizer"]["param_groups"]) == (1 if number == 1 else 2)
            assert read_json(trial.evidence / "interface_checks.json")[str(number)][
                "optimizer_moments_fp32"
            ]
        checks.append(
            "actual three-stage training loop: accumulation, overflow retry, progress and optimizer groups"
        )
        # Interrupted staging dirs and tensors remain confined to the TemporaryDirectory.
        save_tensor(paths.artifacts / "fake.pt", torch.zeros(1))
        assert protocol()["microbatch"] == 8 and protocol()["accumulation"] == 2
    assert not paths.root.exists()
    for check in checks:
        print("PASS", check)
    print(f"[smoke] {len(checks)} groups passed; all synthetic fixtures and outputs removed")


if __name__ == "__main__":
    smoke()
