"""Visual insertion, original LoRA layers, response-only loss and greedy decoding."""

import math
import time

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import create_selective_checkpoint_contexts

from atml.common.io import read_json

from .config import CAPTION_PROMPT, INSTRUCTION, QWEN, SEED


def checkpoint_contexts():
    """Spend 4090 activation memory on expensive matrix products, recompute cheap ops.

    Gradient checkpointing remains enabled; outputs and gradients are unchanged.
    Unlike increasing microbatch size, this preserves the manual's optimizer recipe.
    """
    return create_selective_checkpoint_contexts(
        [
            torch.ops.aten.mm.default,
            torch.ops.aten.bmm.default,
            torch.ops.aten.addmm.default,
        ]
    )


def enable_checkpointing(language):
    language.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={
            "use_reentrant": False,
            "context_fn": checkpoint_contexts,
        }
    )


class LoRALinear(nn.Module):
    def __init__(self, base, seed, rank=16, alpha=32, dropout=0.05):
        super().__init__()
        self.base = base.requires_grad_(False)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.a = nn.Parameter(
            torch.empty(rank, base.in_features, dtype=torch.float32, device="cpu")
            .uniform_(
                -1 / math.sqrt(base.in_features),
                1 / math.sqrt(base.in_features),
                generator=generator,
            )
            .to(base.weight.device)
        )
        self.b = nn.Parameter(
            torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32)
        )
        self.dropout = nn.Dropout(dropout)
        self.scale = alpha / rank

    def forward(self, inputs):
        return (
            self.base(inputs)
            + F.linear(F.linear(self.dropout(inputs), self.a), self.b) * self.scale
        )


class VisualModel(nn.Module):
    def __init__(self, language, tokenizer, feature_width=768, hidden_width=1024):
        super().__init__()
        self.language = language.requires_grad_(False)
        self.tokenizer = tokenizer
        # Private initialization is independent of checkpoint loads and vision work.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(SEED)
            self.connector = nn.Sequential(
                nn.Linear(feature_width, hidden_width),
                nn.GELU(),
                nn.Linear(hidden_width, hidden_width),
            )
        self.register_buffer("visual_scale", torch.tensor(1.0, dtype=torch.float32))
        self.has_adapters = False
        self.eot = tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else self.eot
        if not isinstance(self.eot, int) or self.eot == tokenizer.unk_token_id:
            raise ValueError("Qwen end-of-turn token unavailable")

    def enable_adapters(self):
        if self.has_adapters:
            return
        names = [
            name
            for name, module in self.language.named_modules()
            if isinstance(module, nn.Linear)
            and name.rsplit(".", 1)[-1] in {"q_proj", "k_proj", "v_proj", "o_proj"}
        ]
        if not names:
            raise ValueError("No Qwen attention projections found")
        for i, name in enumerate(names):
            parent_name, child = name.rsplit(".", 1)
            parent = self.language.get_submodule(parent_name)
            setattr(parent, child, LoRALinear(getattr(parent, child), SEED + i))
        self.has_adapters = True

    def parameter_counts(self):
        return {
            "connector": sum(p.numel() for p in self.connector.parameters()),
            "adapters": sum(
                p.numel() for name, p in self.language.named_parameters() if p.requires_grad
            ),
            "frozen": sum(p.numel() for p in self.language.parameters() if not p.requires_grad),
        }

    def trainable_state(self):
        return {
            name: tensor.detach().cpu().clone()
            for name, tensor in self.state_dict().items()
            if name.startswith("connector.")
            or name == "visual_scale"
            or name.endswith((".a", ".b"))
        }

    def load_trainables(self, state):
        expected = set(self.trainable_state())
        if set(state) != expected:
            raise ValueError(f"Checkpoint trainable keys differ: {set(state) ^ expected}")
        self.load_state_dict(state, strict=False)

    def prompt_parts(self, row):
        """Render native template first; split at the exact user-content boundary.

        Splitting token IDs (rather than separately tokenizing strings) preserves
        tokenizer boundaries. No invented image special token is added to Qwen.
        """
        question = (
            CAPTION_PROMPT if row["component"] == "caption" else INSTRUCTION + row["question"]
        )
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question},
        ]
        ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
        )
        marker = self.tokenizer.encode("<|im_start|>user\n", add_special_tokens=False)
        matches = [
            i + len(marker)
            for i in range(len(ids) - len(marker) + 1)
            if ids[i : i + len(marker)] == marker
        ]
        if len(matches) != 1:
            raise ValueError("Cannot uniquely locate Qwen user content")
        split = matches[0]
        return ids[:split], ids[split:]

    def build(self, rows, features, targets=True, include_eot=True):
        if len(rows) != len(features) or not rows:
            raise ValueError("Each example must have an image feature or an explicit None control")
        device = self.visual_scale.device
        sequences, label_sequences = [], []
        embed = self.language.get_input_embeddings()
        for row, feature in zip(rows, features):
            before, after = self.prompt_parts(row)
            pre = embed(torch.tensor(before, device=device))
            post = embed(torch.tensor(after, device=device))
            parts = [pre]
            if feature is not None:
                visual = self.connector(feature.to(device=device, dtype=torch.float32))
                parts.append((visual * self.visual_scale).to(pre.dtype))
            parts.append(post)
            prefix = torch.cat(parts)
            labels = [-100] * len(prefix)
            if targets:
                response = self.tokenizer.encode(row["target"], add_special_tokens=False)
                if include_eot:
                    response.append(self.eot)
                if not response:
                    raise ValueError("Empty reference content")
                prefix = torch.cat([prefix, embed(torch.tensor(response, device=device))])
                labels += response
            sequences.append(prefix)
            label_sequences.append(labels)
        longest = max(map(len, sequences))
        # Left padding supports a shared last prefill position for greedy decoding.
        embeddings, masks, labels = [], [], []
        for sequence, labs in zip(sequences, label_sequences):
            padding = longest - len(sequence)
            embeddings.append(F.pad(sequence, (0, 0, padding, 0)))
            masks.append([0] * padding + [1] * len(sequence))
            labels.append([-100] * padding + labs)
        attention = torch.tensor(masks, device=device)
        return dict(
            inputs_embeds=torch.stack(embeddings),
            attention_mask=attention,
            position_ids=(attention.cumsum(-1) - 1).clamp_min(0),
            labels=torch.tensor(labels, device=device),
        )

    def losses(self, rows, features, include_eot=True):
        batch = self.build(rows, features, include_eot=include_eot)
        labels = batch.pop("labels")[:, 1:]
        hidden = self.language.model(**batch, use_cache=False, return_dict=True).last_hidden_state[
            :, :-1
        ]
        # Project only supervised positions. Avoid a B×(image+prompt)×vocabulary allocation.
        mask = labels != -100
        selected = hidden[mask]
        target = labels[mask]
        token_losses = []
        for start in range(0, len(target), 512):
            logits = self.language.lm_head(selected[start : start + 512]).float()
            token_losses.append(
                F.cross_entropy(logits, target[start : start + 512], reduction="none")
            )
        token_losses = torch.cat(token_losses)
        # Contiguous row-major mask order lets us reduce without a CUDA scatter.
        return torch.stack([part.mean() for part in token_losses.split(mask.sum(-1).tolist())])

    @torch.no_grad()
    def generate(self, rows, features, max_new=16):
        self.eval()
        batch = self.build(rows, features, targets=False)
        batch.pop("labels")
        mask = batch["attention_mask"]
        if mask.is_cuda:
            torch.cuda.synchronize()
        started = time.perf_counter()
        output = self.language.model(**batch, use_cache=True, return_dict=True)
        cache = output.past_key_values
        hidden = output.last_hidden_state[:, -1]
        finished = torch.zeros(len(rows), dtype=torch.bool, device=mask.device)
        tokens = [[] for _ in rows]
        stopped = [False] * len(rows)
        terminators = {self.eot}
        eos = self.language.generation_config.eos_token_id
        terminators.update(eos if isinstance(eos, list) else [eos])
        for step in range(max_new):
            next_ids = self.language.lm_head(hidden).argmax(-1)
            for i, token in enumerate(next_ids.tolist()):
                if not finished[i]:
                    tokens[i].append(token)
                    if token in terminators:
                        finished[i] = True
                        stopped[i] = True
            if finished.all() or step == max_new - 1:
                break
            mask = torch.cat([mask, torch.ones_like(mask[:, :1])], dim=1)
            positions = (mask.sum(-1) - 1)[:, None]
            output = self.language.model(
                input_ids=next_ids[:, None],
                attention_mask=mask,
                position_ids=positions,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            cache, hidden = output.past_key_values, output.last_hidden_state[:, -1]
        if mask.is_cuda:
            torch.cuda.synchronize()
        seconds = (time.perf_counter() - started) / len(rows)
        return [
            dict(
                prediction=self.tokenizer.decode(ids, skip_special_tokens=True).strip(),
                token_ids=ids,
                generated_tokens=len(ids),
                truncated=not stop,
                generation_seconds=seconds,
            )
            for ids, stop in zip(tokens, stopped)
        ]


def load_model(paths, adapters=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    lock = read_json(paths.evidence / "asset_lock.json")
    tokenizer = AutoTokenizer.from_pretrained(paths.artifacts / "tokenizer")
    language = AutoModelForCausalLM.from_pretrained(
        QWEN, revision=lock[QWEN], torch_dtype=torch.float16, attn_implementation="sdpa"
    )
    enable_checkpointing(language)
    model = VisualModel(language, tokenizer)
    if adapters:
        model.enable_adapters()
    return model.to("cuda")
