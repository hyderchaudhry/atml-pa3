"""Padding-aware token evidence shared by quantitative and qualitative generation."""

from __future__ import annotations

import hashlib
import json


def token_key(tokens):
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


def generation_record(tokenizer, inputs, output, cap, eos_ids, stops=()):
    """Exclude batch padding and tokens after this row's first terminating event.

    The stock harness determines stopping during generation. For the evidence,
    reconstruct the first matching text boundary on CPU, including its final token.
    EOS is included in actual output tokens; subsequent EOS/pad tokens are excluded.
    """
    eos_ids = eos_ids if isinstance(eos_ids, (list, tuple)) else [eos_ids]
    reason = "length"
    for index, token in enumerate(output):
        if token in eos_ids:
            output = output[: index + 1]
            reason = "eos"
            break
    text = tokenizer.decode(output, skip_special_tokens=False)
    matches = [(text.find(stop), stop) for stop in stops if stop and stop in text]
    if matches:
        _, stop = min(matches, key=lambda item: item[0])
        # Prefix decoding avoids assuming a stop string has context-independent IDs.
        # This work happens once per response, never once per GPU decoding step.
        low, high = 1, len(output)
        while low < high:
            middle = (low + high) // 2
            prefix = tokenizer.decode(output[:middle], skip_special_tokens=False)
            if stop in prefix:
                high = middle
            else:
                low = middle + 1
        output = output[:low]
        text = tokenizer.decode(output, skip_special_tokens=False)
        reason = "stock_stop"
    return {
        "input_ids": inputs,
        "output_ids": output,
        "input_tokens": len(inputs),
        "output_tokens": len(output),
        "new_token_cap": cap,
        "truncated": len(output) >= cap and reason == "length",
        "stop_reason": reason,
        "stock_stops": list(stops),
        "raw_text": text,
        "prompt_token_key": token_key(inputs),
    }
