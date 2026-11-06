"""The prescribed response-token CE without full-sequence vocabulary logits."""

from __future__ import annotations

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

LOSS_CHUNK_TOKENS = 256


def response_loss(model, batch, chunk_tokens=LOSS_CHUNK_TOKENS):
    """Return summed next-token CE over response/EOT tokens, retaining exact gradients.

    Qwen's public backbone returns hidden states. Only supervised positions need
    vocabulary projection. Recomputing each head chunk during backward bounds
    logits/softmax storage without changing batch size, tokenization, or targets.
    """
    hidden = model.model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
        return_dict=True,
    ).last_hidden_state
    targets = batch["labels"][:, 1:]
    valid = targets != -100
    features = hidden[:, :-1][valid]
    targets = targets[valid]
    if not len(targets):
        raise ValueError("Effective training input has no supervised response tokens.")

    def head_loss(inputs, labels):
        return F.cross_entropy(model.lm_head(inputs).float(), labels, reduction="sum")

    total = torch.zeros((), device=features.device, dtype=torch.float32)
    for offset in range(0, len(targets), chunk_tokens):
        total = total + checkpoint(
            head_loss,
            features[offset : offset + chunk_tokens],
            targets[offset : offset + chunk_tokens],
            use_reentrant=False,
            preserve_rng_state=False,
        )
    return total
