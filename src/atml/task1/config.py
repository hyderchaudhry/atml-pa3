"""The manual's fixed experimental protocol, with no tuning switches."""

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Recipe:
    model: str = "Qwen/Qwen3-0.6B"
    seed: int = 6304
    examples: int = 40000
    updates: int = 1250
    microbatch: int = 2
    accumulation: int = 16
    max_length: int = 2048
    warmup: int = 38
    weight_decay: float = 0.01
    adam_betas: tuple = (0.9, 0.999)
    adam_epsilon: float = 1e-8
    clip_norm: float = 1.0
    dropout: float = 0.05
    protection_rank: int = 128
    recovery_updates: tuple = (312, 625, 937, 1250)
    diagnostic_layers: tuple = (0, 13, 27)


@dataclass(frozen=True)
class Method:
    name: str
    learning_rate: float
    rank: int = 0
    alpha: int = 0
    l2_lambda: float = 0.0
    orthogonal: bool = False


RECIPE = Recipe()
METHODS = {
    "full_ft": Method("full_ft", 2e-5),
    "l2sp": Method("l2sp", 2e-5, l2_lambda=1e-3),
    "lora4": Method("lora4", 2e-4, rank=4, alpha=8),
    "lora16": Method("lora16", 2e-4, rank=16, alpha=32),
    "oplora16": Method("oplora16", 2e-4, rank=16, alpha=32, orthogonal=True),
}
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
CONDITIONS = {
    "baseline": (None, 0),
    "full_ft_midpoint": ("full_ft", 625),
    **{f"{name}_final": (name, 1250) for name in METHODS},
}
BENCHMARKS = {
    "gsm8k_cot": (1319, "exact_match,flexible-extract"),
    "hellaswag": (10042, "acc_norm,none"),
    "boolq": (3270, "acc,none"),
    "arc_easy": (2376, "acc_norm,none"),
}
DATASETS = {
    "metamath": "meta-math/MetaMathQA",
    "gsm8k": "openai/gsm8k",
    "hellaswag": "Rowan/hellaswag",
    "boolq": "aps/super_glue",
    "arc_easy": "allenai/ai2_arc",
}


@dataclass(frozen=True)
class Paths:
    artifacts: Path = Path("artifacts/task1")
    evidence: Path = Path("evidence/task1")

    @property
    def prepared(self):
        return self.artifacts / "prepared"

    def checkpoint(self, method, update):
        return self.artifacts / "checkpoints" / method / f"update_{update:04d}"


def protocol() -> dict:
    return {
        "recipe": asdict(RECIPE),
        "methods": {k: asdict(v) for k, v in METHODS.items()},
        "conditions": CONDITIONS,
        "benchmarks": BENCHMARKS,
        "targets": TARGETS,
    }
