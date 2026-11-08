"""Manual-defined protocol and independent Task 3 output paths."""

from dataclasses import dataclass
from pathlib import Path

SEED = 6304
MODEL = "Qwen/Qwen2.5-3B-Instruct"
BUDGETS = (1, 2, 4, 8, 16, 32)
CHECKPOINTS = (4, 8, 16, 32)
TAUS = (0.5, 0.625, 0.75, 0.875, 1.0)
GAMMAS = (0.0, 0.1, 0.25, 0.5)
SYSTEM = "You are a helpful assistant."
INSTRUCTION = (
    r"Think step by step inside <think> </think> tags, then give your final answer inside \boxed{}."
)
REVISION = (
    r"Review the solution above. Identify and correct any errors; if it is correct, "
    r"retain its answer. Give your final answer inside \boxed{}."
)
DATASETS = ("math", "mathvista")
NO_ANSWER = "__NO_ANSWER__"


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def evidence(self):
        return self.root / "evidence" / "task3"

    @property
    def artifacts(self):
        return self.root / "artifacts" / "task3"


def protocol():
    return dict(
        schema=1,
        seed=SEED,
        datasets=list(DATASETS),
        questions_per_dataset=1000,
        rollouts_per_question=32,
        development_questions=200,
        evaluation_questions=800,
        live_questions=50,
        orders_per_question=10,
        budgets=list(BUDGETS),
        bootstrap=dict(resamples=1000, unit="question", averaging="orders within question first"),
        live=dict(
            model=MODEL,
            revision_source="supplied README.MD",
            batch_size=1,
            weights="float16",
            quantization=None,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            repetition_penalty=1.0,
            system=SYSTEM,
            instruction=INSTRUCTION,
            revision_instruction=REVISION,
            caps=dict(long=3072, parallel=1024, revision=2048),
            shared_initial="parallel_0 reused by refinement; cost charged to both strategies",
        ),
        stopping=dict(
            checkpoints=list(CHECKPOINTS),
            taus=list(TAUS),
            gammas=list(GAMMAS),
            development_accuracy_tolerance=0.01,
            fallback="always32",
            selection="lowest mean tokens, higher accuracy, smaller tau, smaller gamma",
        ),
        scoring=dict(
            math="math-verify==0.8.0",
            mathvista="released answer_type/choices/precision metadata",
            missing="one NO_ANSWER group, always incorrect, retained in voting and cost",
            ties="earliest supporting rollout in current order",
            rwc="descending average rank",
            calibration="current prefix pooled token KL, linear Q05/Q95, denominator >=1e-8",
        ),
        runtime=dict(
            offline="per-question resumable work; compact arrays and vectorized reductions",
            live="SDPA, KV cache, one frozen FP16 model, durable per-response checkpoints",
        ),
    )
