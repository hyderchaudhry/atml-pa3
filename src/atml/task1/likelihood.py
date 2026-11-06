"""Stock conditional token likelihoods without vocabulary logits for prompt tokens.

The harness supplies its encoded context/continuation pairs and performs all task
aggregation. Only the model-execution layer is replaced. Its causal window,
softmax dtype, sum over continuation tokens and greedy-match rule are preserved.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F


@torch.inference_mode()
def continuation_scores(lm, requests, disable_tqdm=False, override_bs=None):
    if lm.backend != "causal":
        raise ValueError("Task 1's continuation scorer supports causal Qwen models only.")
    batch_size = override_bs or lm.batch_size
    if not isinstance(batch_size, int):
        raise ValueError("Use a positive integer likelihood batch size.")
    # Multiple one-token candidates often use exactly the same backbone input.
    # Evaluate that prefix once, retaining a separate score for every candidate.
    prefixes = {}
    for index, (key, context, continuation) in enumerate(requests):
        if not context or not continuation or len(continuation) > lm.max_length:
            raise ValueError("Invalid stock-harness context/continuation token lengths.")
        tokens = tuple((context + continuation)[-(lm.max_length + 1) :][:-1])
        prefixes.setdefault(tokens, []).append((index, key, continuation))
    ordered = sorted(prefixes, key=lambda tokens: (-len(tokens), tokens))
    answers = [None] * len(requests)
    for offset in range(0, len(ordered), batch_size):
        group = ordered[offset : offset + batch_size]
        width = max(map(len, group))
        tokens = torch.full(
            (len(group), width), lm.tokenizer.pad_token_id, dtype=torch.long, device=lm.device
        )
        mask = torch.zeros_like(tokens)
        for row, sequence in enumerate(group):
            tokens[row, : len(sequence)] = torch.tensor(sequence, device=lm.device)
            mask[row, : len(sequence)] = 1
        hidden = lm.model.model(
            input_ids=tokens, attention_mask=mask, use_cache=False, return_dict=True
        ).last_hidden_state
        features, targets, members = [], [], []
        for row, sequence in enumerate(group):
            for index, key, continuation in prefixes[sequence]:
                features.append(hidden[row, len(sequence) - len(continuation) : len(sequence)])
                targets.extend(continuation)
                members.append((index, key, len(continuation)))
        features = torch.cat(features)
        targets = torch.tensor(targets, dtype=torch.long, device=lm.device)
        selected, greedy = [], []
        # Bound vocabulary tensors independently of the evaluation batch size.
        for first in range(0, len(targets), 256):
            logits = lm.model.lm_head(features[first : first + 256])
            probabilities = F.log_softmax(logits, dim=-1, dtype=lm.softmax_dtype)
            labels = targets[first : first + 256]
            selected.append(probabilities.gather(1, labels[:, None]).squeeze(1))
            greedy.append(probabilities.argmax(-1) == labels)
        selected, greedy = torch.cat(selected), torch.cat(greedy)
        first, sums, matches = 0, [], []
        for _, _, count in members:
            # Sum once in the stock dtype, rather than summing chunk-level totals.
            sums.append(selected[first : first + count].sum())
            matches.append(greedy[first : first + count].all())
            first += count
        values, exact = torch.stack(sums).cpu().tolist(), torch.stack(matches).cpu().tolist()
        if hasattr(lm.cache_hook, "begin_batch"):
            lm.cache_hook.begin_batch(len(members))
        for (index, key, _), value, match in zip(members, values, exact, strict=True):
            answer = (float(value), bool(match))
            answers[index] = answer
            lm.cache_hook.add_partial("loglikelihood", key, answer)
        if not disable_tqdm:
            print(
                f"[likelihood] prefixes {offset + len(group):,}/{len(ordered):,} saved", flush=True
            )
        del hidden, features, probabilities, logits
    return answers
