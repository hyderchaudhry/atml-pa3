"""Task-specific extraction/equivalence wrappers; no selection uses gold answers."""

import functools
import math
import re

from .config import NO_ANSWER


def last_boxed(text):
    """Supplied README rule: last literal \\boxed{, balanced braces, no fallback."""
    opening = text.rfind(r"\boxed{")
    if opening < 0:
        return None
    start = opening + len(r"\boxed{")
    depth = 1
    for i in range(start, len(text)):
        char = text[i]
        depth += (char == "{") - (char == "}")
        if depth == 0:
            answer = text[start:i].strip()
            return answer or None
    return None


@functools.lru_cache(maxsize=65536)
def parse_math(answer):
    from math_verify import parse

    if answer is None or not str(answer).strip() or answer == NO_ANSWER:
        return ()
    # Extractions are answers, not fresh model responses. Anchor the entire value
    # as mathematics rather than letting an incidental number win extraction.
    return tuple(
        parse(
            r"\boxed{" + str(answer).strip() + "}",
            fallback_mode="no_fallback",
            extraction_mode="first_match",
            parsing_timeout=5,
        )
    )


@functools.lru_cache(maxsize=131072)
def math_equivalent(a, b):
    from math_verify import verify

    if a is None or b is None or a == NO_ANSWER or b == NO_ANSWER:
        return False
    a, b = str(a).strip(), str(b).strip()
    parsed_a, parsed_b = parse_math(a), parse_math(b)
    if not parsed_a or not parsed_b:
        return False
    if a == b:
        return True
    return bool(
        verify(list(parsed_a), list(parsed_b), strict=True, timeout_seconds=5)
        or verify(list(parsed_b), list(parsed_a), strict=True, timeout_seconds=5)
    )


def edit_distance(a, b):
    """Linear-space Levenshtein distance for matching multiple-choice text."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def normalize_mathvista(answer, metadata):
    if answer is None or not str(answer).strip() or answer == NO_ANSWER:
        return NO_ANSWER
    answer = str(answer).strip()
    if metadata["question_type"] == "multi_choice":
        choices = metadata.get("choices")
        if not choices:
            raise ValueError("Multiple-choice question has no released choices")
        letter = re.fullmatch(r"\(?([A-Za-z])\)?[.)]?", answer)
        if letter and 0 <= ord(letter[1].upper()) - 65 < len(choices):
            return letter[1].upper()
        # Choice-text matching is gold-free. Option order breaks equal distances.
        distances = [edit_distance(answer, str(choice)) for choice in choices]
        return chr(65 + min(range(len(choices)), key=lambda i: distances[i]))
    kind = metadata["answer_type"]
    if kind in ("integer", "float"):
        if kind == "float" and metadata.get("precision") is None:
            raise ValueError("Float-answer precision missing from released metadata")
        try:
            value = float(answer)
            if not math.isfinite(value):
                return NO_ANSWER
            if kind == "integer":
                return str(int(value))
            precision = metadata.get("precision")
            if precision is None:
                raise ValueError("Float-answer precision missing from released metadata")
            return str(round(value, int(precision)))
        except (ValueError, OverflowError):
            return NO_ANSWER
    return answer


class Scorer:
    def __init__(self, dataset, metadata=None):
        self.dataset = dataset
        # Deliberately exclude the metadata's gold-answer field from normalization.
        self.metadata = {
            k: v
            for k, v in (metadata or {}).items()
            if k in ("question_type", "answer_type", "choices", "precision")
        }

    def answer(self, value):
        if value is None or not str(value).strip():
            return NO_ANSWER
        return (
            str(value).strip()
            if self.dataset == "math"
            else normalize_mathvista(value, self.metadata)
        )

    def equivalent(self, a, b):
        if a == NO_ANSWER or b == NO_ANSWER:
            return False
        return math_equivalent(a, b) if self.dataset == "math" else a == b

    def correct(self, prediction, gold):
        return int(self.equivalent(self.answer(prediction), self.answer(gold)))
