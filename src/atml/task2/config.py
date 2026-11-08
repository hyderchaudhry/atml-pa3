"""The manual's fixed main-run recipe; operational paths are configurable."""

from dataclasses import asdict, dataclass
from pathlib import Path

SEED = 6304
QWEN = "Qwen/Qwen3-0.6B"
SIGLIP = "google/siglip-base-patch16-224"
DEV_UPDATES = (0, 500, 1000, 1500, 2000, 2250, 2500, 3000, 3500, 3750, 4000, 4500)
CONDITIONS = ("correct", "complementary", "mismatched", "no_image")
INSTRUCTION = "Answer the question using the image. Give only a short answer.\n"
CAPTION_PROMPT = "Describe this image."


@dataclass(frozen=True)
class Stage:
    number: int
    start: int
    end: int
    connector_lr: float
    adapter_lr: float
    components: tuple
    probabilities: tuple


STAGES = (
    Stage(1, 0, 2250, 1e-3, 0, ("caption",), (1.0,)),
    Stage(2, 2250, 3750, 1e-4, 2e-5, ("presence", "count", "caption"), (0.5, 0.25, 0.25)),
    Stage(
        3, 3750, 4500, 5e-5, 1e-5, ("natural", "presence", "count", "caption"), (0.7, 0.1, 0.1, 0.1)
    ),
)


def stage_at(update):
    """Stage responsible for the *next* applied update."""
    return next(s for s in STAGES if s.start <= update < s.end)


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def artifacts(self):
        return self.root / "artifacts" / "task2"

    @property
    def evidence(self):
        return self.root / "evidence" / "task2"

    @property
    def manifest(self):
        return self.evidence / "selection.json"

    def checkpoint(self, update):
        return self.artifacts / "checkpoints" / f"update_{update:04d}"


def protocol():
    return dict(
        schema=1,
        seed=SEED,
        visual_tokens=196,
        vision=SIGLIP,
        language=QWEN,
        connector=[768, 1024, 1024],
        activation="GELU",
        thinking=False,
        lora=dict(
            rank=16,
            alpha=32,
            dropout=0.05,
            bias=False,
            targets=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
        microbatch=8,
        accumulation=2,
        effective_batch=16,
        updates=4500,
        stages=[
            {**asdict(s), "components": list(s.components), "probabilities": list(s.probabilities)}
            for s in STAGES
        ],
        optimizer="AdamW",
        weight_decay=0.01,
        betas=[0.9, 0.999],
        epsilon=1e-8,
        clip_norm=1.0,
        warmup="ceil(0.05 * stage_updates), then cosine to 0.1 of peak",
        optimizer_transition="fresh AdamW and scheduler at each stage; weights continue",
        precision="FP16 frozen backbone/autocast; FP32 trainables and Adam moments; dynamic scaler",
        checkpointing="nonreentrant selective checkpointing; retain mm/bmm/addmm outputs",
        response_projection_chunk=512,
        loss="mean supervised token CE per example, then mean of 16 examples; includes im_end",
        caps=dict(question=80, caption=64, answer=16, generation=16),
        eligibility="filter over-cap source content before selection; never truncate references",
        checkpoints=list(range(250, 4501, 250)),
        development=list(DEV_UPDATES),
        final_batch=8,
        development_likelihood_batch=32,
        vision_batch_start=512,
        feature_residency=dict(max_vram_fraction=0.25, model_work_reserve_gib=12),
        bootstrap_pairs=4000,
        normalization="task2-vqa-1",
        grounding_pool="4096 seeded pairs per type on selected train images; crowd excluded",
        grounding_evaluation="equal presence/count pairs; globally disjoint evaluation images",
        calibration="training-only RMS(text embeddings) / RMS(initial connector output); 64 images",
    )
