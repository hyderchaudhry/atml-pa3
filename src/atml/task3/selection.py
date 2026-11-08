"""Gold-free quality scoring, answer grouping, aggregation, and stopping.

Only candidate measurements and candidate-to-candidate equivalence enter these
functions. Correctness is attached by the evaluator after choices are fixed.
"""

from dataclasses import dataclass

import numpy as np

from .config import NO_ANSWER


@dataclass
class Pool:
    rollout_ids: list
    answers: list
    tokens: np.ndarray
    logp: list
    kl: list
    vision: list | None = None

    def __post_init__(self):
        n = len(self.rollout_ids)
        if not (n == len(self.answers) == len(self.tokens) == len(self.logp) == len(self.kl)):
            raise ValueError("Candidate fields have different lengths")
        if len(set(self.rollout_ids)) != n:
            raise ValueError("Duplicate rollout IDs")
        for i in range(n):
            if self.tokens[i] != len(self.logp[i]) or self.tokens[i] != len(self.kl[i]):
                raise ValueError("Token measurement alignment failure")
            if self.vision is not None and len(self.vision[i]) != self.tokens[i]:
                raise ValueError("Attention/continuation alignment failure")


def mean_or_zero(values):
    return float(np.mean(values)) if len(values) else 0.0


def quality(pool, indices):
    """Recompute candidate-relative confidence on *only* this visible prefix."""
    indices = np.asarray(indices, dtype=np.int64)
    result = dict(
        logp=np.asarray([mean_or_zero(pool.logp[i]) for i in indices]),
        certainty=np.asarray([mean_or_zero(pool.kl[i]) for i in indices]),
        shortest=-pool.tokens[indices].astype(np.float64),
    )
    if pool.vision is None:
        return result, None
    pooled = np.concatenate([pool.kl[i] for i in indices])
    low, high = np.quantile(pooled, [0.05, 0.95], method="linear") if len(pooled) else (0.0, 0.0)
    calibrated = [np.clip((pool.kl[i] - low) / max(high - low, 1e-8), 0.0, 1.0) for i in indices]
    confidence = np.asarray([mean_or_zero(c) for c in calibrated])
    vision = np.asarray([mean_or_zero(pool.vision[i]) for i in indices])
    result.update(
        vision=vision,
        visconf=vision * confidence,
        token_fusion=np.asarray(
            [mean_or_zero(c * pool.vision[i]) for c, i in zip(calibrated, indices)]
        ),
    )
    return result, dict(
        q05=float(low), q95=float(high), confidence=confidence, calibrated=calibrated
    )


def visible_groups(answers, equivalent):
    """Representative-first grouping uses only revealed answers in their order."""
    groups, representatives = [], []
    for answer in answers:
        if answer is None or answer == NO_ANSWER:
            groups.append(NO_ANSWER)
            continue
        group = next((j for j, rep in enumerate(representatives) if equivalent(rep, answer)), None)
        if group is None:
            group = len(representatives)
            representatives.append(answer)
        groups.append(group)
    return groups


def rank_weights(scores):
    """Descending average ranks, including ties; N=1 returns its sole candidate."""
    scores = np.asarray(scores, dtype=np.float64)
    if len(scores) == 1:
        return np.ones(1)
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite candidate quality score")
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and scores[order[end]] == scores[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    return (len(scores) - ranks) / (len(scores) - 1)


def vote(groups, weights=None):
    """Earliest supporting candidate resolves tied group votes."""
    weights = np.ones(len(groups)) if weights is None else weights
    totals = {}
    first = {}
    for i, (group, weight) in enumerate(zip(groups, weights)):
        totals[group] = totals.get(group, 0.0) + float(weight)
        first.setdefault(group, i)
    ranking = sorted(totals, key=lambda g: (-totals[g], first[g]))
    winner = ranking[0]
    runner = totals[ranking[1]] if len(ranking) > 1 else 0.0
    total = sum(totals.values())
    margin = (totals[winner] - runner) / total if total else 0.0
    unique = len(ranking) == 1 or totals[winner] > runner
    return dict(
        index=first[winner],
        group=winner,
        unique=unique,
        margin=margin,
        support=totals[winner] / total if total else 0.0,
        votes=totals,
    )


def select(method, groups, scores):
    if method == "first":
        return 0
    if method == "consensus":
        return vote(groups)["index"]
    name, _, aggregation = method.partition(":")
    if aggregation == "rwc":
        return vote(groups, rank_weights(scores[name]))["index"]
    return int(np.argmax(scores[name]))


def methods(dataset):
    common = ["first", "consensus", "logp", "certainty", "certainty:rwc"]
    if dataset == "mathvista":
        common += [
            "vision",
            "vision:rwc",
            "visconf",
            "visconf:rwc",
            "token_fusion",
            "token_fusion:rwc",
        ]
    return common + ["shortest"]


def stopping_trace(pool, order, equivalent, dataset):
    """Compute prefix evidence once, reused across the entire controller grid."""
    trace = []
    for n in (4, 8, 16, 32):
        indices = order[:n]
        groups = visible_groups([pool.answers[i] for i in indices], equivalent)
        scores, _ = quality(pool, indices)
        consensus = vote(groups)
        rwc = vote(groups, rank_weights(scores["certainty" if dataset == "math" else "visconf"]))
        trace.append(
            dict(
                n=n,
                selected=int(indices[consensus["index"]]),
                agreement=consensus["unique"] and consensus["group"] != NO_ANSWER,
                support=consensus["support"],
                winner_matches=rwc["group"] == consensus["group"],
                margin=rwc["margin"],
                tokens=int(pool.tokens[indices].sum()),
            )
        )
    return trace


def stop(trace, family, tau=1.0, gamma=0.0):
    for state in trace:
        if state["n"] == 32 or family == "always32":
            if state["n"] == 32:
                return state
            continue
        if not state["agreement"] or state["support"] < tau:
            continue
        if family == "agreement" or (state["winner_matches"] and state["margin"] >= gamma):
            return state
    raise ValueError("A trace must include the fallback at 32")


def choose_policy(candidates, baseline_accuracy):
    """Candidates contain *development* aggregates only. No held-out labels enter."""
    eligible = [r for r in candidates if r["accuracy"] >= baseline_accuracy - 0.01 - 1e-12]
    if not eligible:
        raise ValueError("Missing always32 fallback")
    return min(
        eligible,
        key=lambda r: (
            r["tokens"],
            -r["accuracy"],
            r.get("tau", float("inf")),
            r.get("gamma", float("inf")),
        ),
    )


def bootstrap_questions(values, resamples=1000, seed=6304):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    rng = np.random.default_rng(seed)
    # Bounded chunks avoid a large resamples×questions×methods temporary.
    estimates = []
    for start in range(0, resamples, 50):
        indices = rng.integers(0, len(values), (min(50, resamples - start), len(values)))
        estimates.append(values[indices].mean(axis=1))
    draws = np.concatenate(estimates)
    return dict(
        mean=values.mean(axis=0).tolist(),
        low=np.quantile(draws, 0.025, axis=0).tolist(),
        high=np.quantile(draws, 0.975, axis=0).tolist(),
    )


def finite_correlation(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def signal_diagnostics(pool, correct, equivalent):
    scores, calibration = quality(pool, np.arange(len(pool.answers)))
    values = {}
    for name, score in scores.items():
        order = np.argsort(score, kind="stable")
        quartile = max(1, len(order) // 4)
        values[name] = dict(
            length_correctness_correlation=finite_correlation(pool.tokens, correct),
            length_correlation=finite_correlation(pool.tokens, score),
            correctness_correlation=finite_correlation(correct, score),
            top_accuracy=float(np.mean(correct[order[-quartile:]])),
            bottom_accuracy=float(np.mean(correct[order[:quartile]])),
            top_minus_bottom=float(
                np.mean(correct[order[-quartile:]]) - np.mean(correct[order[:quartile]])
            ),
        )
    result = dict(scores=values)
    if pool.vision is not None:
        correlations, covariances = [], []
        for i, c in enumerate(calibration["calibrated"]):
            v = pool.vision[i]
            correlations.append(finite_correlation(c, v))
            covariances.append(mean_or_zero(c * v) - mean_or_zero(c) * mean_or_zero(v))
        result.update(
            token_correlations=correlations,
            token_covariances=covariances,
            fusion_difference=(scores["token_fusion"] - scores["visconf"]).tolist(),
        )
    return result
