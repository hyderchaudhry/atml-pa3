"""Local VQA normalization and complete-pair bootstrap, written for this task.

The public VQA evaluation specification defines the normalization and scorer.
No external evaluation/training implementation is imported.
"""

import re
from collections import Counter

import numpy as np

from .config import SEED

NUMBERS = dict(
    zip(
        "none zero one two three four five six seven eight nine ten".split(),
        "0 0 1 2 3 4 5 6 7 8 9 10".split(),
    )
)
# Restricted contraction spellings: avoid treating words such as "were", "well",
# "hell" and "ill" as contractions. Lexical entries are normalization rules.
CONTRACTIONS = {
    "cant": "can't",
    "wont": "won't",
    "dont": "don't",
    "aint": "ain't",
    "arent": "aren't",
    "couldnt": "couldn't",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "hadnt": "hadn't",
    "hasnt": "hasn't",
    "havent": "haven't",
    "isnt": "isn't",
    "mightnt": "mightn't",
    "mustnt": "mustn't",
    "neednt": "needn't",
    "oughtnt": "oughtn't",
    "shant": "shan't",
    "shouldnt": "shouldn't",
    "wasnt": "wasn't",
    "werent": "weren't",
    "wouldnt": "wouldn't",
    "im": "i'm",
    "ive": "i've",
    "lets": "let's",
    "oclock": "o'clock",
    "yall": "y'all",
    "maam": "ma'am",
}
for _canonical in """
could've might've must've should've would've not've
he'd he's how'd how'll how's it'd it'll
somebody'll somebody's someone'd someone'll someone's something'd something'll
that's there'd there're there's they'd they'll they're they've
what'll what're what's what've when's where'd where's where've
who'd who'll who's who've why'll why're why's you'd you'll you're you've
we've
""".split():
    CONTRACTIONS[_canonical.replace("'", "")] = _canonical
for _base in "couldn't hadn't he'd it'd mightn't she'd shouldn't somebody'd someone'd something'd there'd they'd we'd who'd wouldn't you'd".split():
    _compound = _base + "'ve"
    CONTRACTIONS[_base + "ve"] = _compound
    CONTRACTIONS[_base.replace("'", "") + "'ve"] = _compound
CONTRACTIONS.update(
    {
        "twas": "'twas",
        "yall'll": "y'all'll",
        "y'allll": "y'all'll",
        "yall'd've": "y'all'd've",
        "y'alld've": "y'all'd've",
        "y'all'dve": "y'all'd've",
    }
)


def normalize(answer):
    text = str(answer).lower().replace("\n", " ").replace("\t", " ").strip()
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    text = re.sub(r"\.(?!\d)", "", text)
    text = re.sub(r"""[;\/\[\]"{}()=+\\_<>@`,?!\-]""", " ", text)
    words = (NUMBERS.get(w, w) for w in text.split())
    return " ".join(CONTRACTIONS.get(w, w) for w in words if w not in {"a", "an", "the"})


def consensus(answers):
    normalized = [normalize(a) for a in answers]
    key, count = Counter(normalized).most_common(1)[0]
    return next(a for a, n in zip(answers, normalized) if n == key), key, count


def soft_score(prediction, answers):
    if len(answers) != 10:
        raise ValueError("VQAv2 scoring requires all ten human answers")
    pred = normalize(prediction)
    matches = [normalize(a) == pred for a in answers]
    k = sum(matches)
    return sum(min(1.0, (k - int(m)) / 3) for m in matches) / 10


def pair_statistics(rows):
    """Each row represents an answer; two consecutive rows must form one pair."""
    if not rows or len(rows) % 2:
        raise ValueError("Expected nonempty complete pairs")
    values = []
    for a, b in zip(rows[::2], rows[1::2]):
        if a["pair_id"] != b["pair_id"]:
            raise ValueError("Broken pair ordering")
        values.append(
            [
                (a["accuracy"] + b["accuracy"]) / 2,
                float(a["exact"] and b["exact"]),
                float(normalize(a["prediction"]) == normalize(b["prediction"])),
                (float(not normalize(a["prediction"])) + float(not normalize(b["prediction"]))) / 2,
                (a["nll"] + b["nll"]) / 2,
                (a["generation_seconds"] + b["generation_seconds"]) / 2,
                (a["partner_exact"] + b["partner_exact"]) / 2,
            ]
        )
    return np.asarray(values, dtype=np.float64)


STAT_NAMES = (
    "accuracy",
    "pair_accuracy",
    "identical_rate",
    "empty_rate",
    "nll",
    "seconds_per_answer",
    "partner_answer_rate",
)


def bootstrap(values, seed=SEED, resamples=4000):
    """Same seed and pair order give matched resamples for all image conditions."""
    values = np.asarray(values)
    indices = np.random.default_rng(seed).integers(0, len(values), (resamples, len(values)))
    draws = values[indices].mean(axis=1)
    return dict(
        mean=values.mean(axis=0).tolist(),
        low=np.quantile(draws, 0.025, axis=0).tolist(),
        high=np.quantile(draws, 0.975, axis=0).tolist(),
    )
