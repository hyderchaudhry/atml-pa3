# ATML PA3

Task 1 is implemented. Tasks 2 and 3 are intentionally not implemented. The training loop, LoRA/OPLoRA layers, data selection, diagnostics, and evidence pipeline were written for this repository; no public assignment solution or third-party fine-tuning implementation was copied. PyTorch, Transformers, Datasets, and the required lm-evaluation-harness provide library functionality. PEFT is installed because the harness imports it, but this implementation does not use it for adapters.

**Full experiments have not been run.** Local checks cover loss/gradient equivalence, staged recovery, and backup/restore failure handling with tiny models and a simulated Hub. Temporary check files are removed. Your actual runs create the benchmark results and other evidence described below. The assignment's PDF report and qualitative interpretation must be written by you, consistent with its AI usage policy.

**Kaggle:** follow the [copy-and-paste guide](docs/kaggle.md): clone/install, set your backup name and question-file path, then run `bash scripts/kaggle.sh run`. The launcher chooses the remaining steps and restores saved progress automatically. No notebook files are needed, and the training/evaluation recipe stays fixed.

## Install

Use Python 3.10–3.12 (3.11 recommended), a CUDA-capable NVIDIA GPU, and persistent storage. CPU/MPS is deliberately rejected for prescribed experiments: the manual requires FP16 CUDA AMP with dynamic gradient scaling. Full FT needs substantial GPU memory; all four recovery checkpoints retain optimizer state, so allow substantial disk space (roughly 100 GB for the complete pipeline, plus caches). Actual usage is logged.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
export HF_HOME="$PWD/artifacts/hf-cache"
export MPLCONFIGDIR="$PWD/artifacts/matplotlib-cache"
export TOKENIZERS_PARALLELISM=false
atml-task1 protocol
```

Dependencies are pinned in [requirements.txt](requirements.txt). Each preparation, training run, and evaluation also saves the installed package versions, platform/GPU details, source hashes, and Git revision. Model and dataset commit revisions are resolved once during preparation and reused. Keep one environment for all five variants and seven evaluation conditions.

## Supply the qualitative prompts

The manual references `first_plot_questions.yaml`, but this file was not supplied in the starting repository. Obtain the exact assignment file and import it before the complete pipeline:

```bash
atml-task1 import-prompts /absolute/path/to/first_plot_questions.yaml
```

The importer accepts either a list of 24 records with `question_id`, `format`, and `prompt` fields, or a mapping whose `plain`, `json`, and `template` keys each contain eight prompt strings in matching question order. It preserves the original YAML and its SHA-256 hash. Prompts are never invented or rewritten.

If the supplied YAML uses another layout, make a separate JSON mapping with 24 entries. Each entry identifies the existing prompt's path within the YAML and gives its underlying question ID and format. For example, if `questions[0].plain` contains a prompt, its mapping entry is:

```json
{"path": ["questions", 0, "plain"], "question_id": "0", "format": "plain"}
```

Put the entries in an array, preserve the supplied plain-question order, and import:

```bash
atml-task1 import-prompts /path/to/first_plot_questions.yaml --prompt-map /path/to/prompt_map.json
```

This explicitly identifies the first and last questions used for the 42-response annotation subset without guessing the source file's structure.

## Run Task 1

For the complete pipeline, after importing the YAML:

```bash
atml-task1 run
```

The command prepares data and fixed probes, computes bases and starting activations, evaluates the shared baseline, trains the five variants independently, evaluates the six required trained conditions, runs diagnostics and qualitative generation, and produces tables and figures. It prints training progress, microbatch progress every four microbatches, losses, learning rates, skipped updates, elapsed time, ETA, and checkpoint locations. Prints are flushed for remote logs. ETA estimates training time and excludes future checkpoint/evaluation work. Use staged commands through the Kaggle wrapper when session/storage limits apply.

The same stages can be run separately:

```bash
atml-task1 prepare
atml-task1 bases
atml-task1 cache-probes
atml-task1 evaluate --condition baseline
atml-task1 qualitative --condition baseline

atml-task1 train --method full_ft
atml-task1 train --method l2sp
atml-task1 train --method lora4
atml-task1 train --method lora16
atml-task1 train --method oplora16

atml-task1 evaluate --condition all
atml-task1 diagnose --condition all
atml-task1 qualitative --condition all
atml-task1 summarize
atml-task1 audit
```

`evaluate` skips completed conditions and completed benchmark splits; the unchanged baseline is evaluated only once. It keeps interrupted per-example output separately when rerunning an incomplete split. Qualitative generation resumes from already saved prompts. `diagnose --condition all` handles the six trained conditions; it does not invent an update for the baseline.

The seven condition names are `baseline`, `full_ft_midpoint`, `full_ft_final`, `l2sp_final`, `lora4_final`, `lora16_final`, and `oplora16_final`. You may select one with `--condition`.

### Recovery and persistent storage

Checkpoints are saved at updates 312, 625, 937, and 1250 (9,984; 20,000; 29,984; 40,000 examples). Each contains model/adapter parameters, AdamW state, analytical scheduler position/next LR, scaler state, Python/NumPy/PyTorch/CUDA RNG state, and data position. Adapter checkpoints also contain projected inference factors. Checkpoints are first written to `.incomplete` directories and renamed after all writes finish.

```bash
atml-task1 train --method full_ft --resume artifacts/task1/checkpoints/full_ft/update_0625
```

Resume revalidates the fixed protocol, starting assets, and prepared-data manifest. Any training log tail after the checkpoint is archived and removed from the active learning curve. If a save was interrupted, preserve or move its `.incomplete` directory before retrying that checkpoint. A run interrupted before its first checkpoint has no resumable optimizer state; preserve/move its partial run directory and restart from the original checkpoint.

`run` resumes the latest complete recovery checkpoint automatically and skips completed final runs. It does not delete or overwrite complete checkpoints. Keep both `artifacts/task1` and `evidence/task1` on persistent storage, especially on a rented GPU or notebook runtime. To use another location, put global options **before** the subcommand, consistently on every command:

```bash
atml-task1 --artifacts /persistent/pa3/artifacts/task1 --evidence /persistent/pa3/evidence/task1 run
```

Prepared data and large model/optimizer artifacts are ignored by Git. Small evidence files are not ignored. Back up large artifacts separately; retaining them is necessary for recovery and reproducible diagnostics. The Kaggle wrapper verifies private Hub copies before freeing local checkpoints and restores them on demand. Do not commit raw datasets or unnecessary checkpoints to the public repository.

## Protocol details

The authoritative settings are in [config.py](src/atml/task1/config.py), also exported to [configs/task1.json](configs/task1.json). There are no reduced-data or alternative-precision production switches.

| Item | Implementation |
| --- | --- |
| Starting point | Post-trained `Qwen/Qwen3-0.6B`, immutable revision, seed 6304; thinking disabled in chat formatting |
| Training data | First 40,000 eligible MetaMathQA training records in source order; no shuffle or answer truncation |
| Leakage filter | Both `query` and `original_question` compared against all GSM8K test questions after NFKC, casefold, punctuation/whitespace removal |
| Length and loss | Entire formatted example ≤2,048 tokens; response and end-of-turn supervised; prompt, trailing template newline, and padding masked |
| Accumulation | Microbatch 2 × 16; summed response CE divided by the total valid response tokens across all 32 examples |
| Optimizer | AdamW (0.9, 0.999), epsilon 1e-8; decay 0.01 excluding biases and tensors with fewer than two dimensions |
| Schedule | One epoch, 1,250 effective batches, 38 warmup updates then cosine; global clipping at 1.0 after unscaling |
| Precision | FP16 AMP and dynamic scaler; FP32 Full FT master parameters, FP32 adapter parameters and optimizer moments; frozen FP16 adapter backbones; gradient checkpointing |
| Full FT / L2-SP | Peak LR 2e-5; λ=0 / 1e-3; exact unaveraged half-sum over unique parameters, including bias/norm and tied weights once |
| L2-SP storage | Fixed CPU FP32 reference; tensor-at-a-time transfer; exact scaled penalty gradients added once per effective batch before unscale/clip; storage and transfer/penalty overhead recorded |
| LoRA | (rank, alpha) = (4,8), (16,32); LR 2e-4; dropout 0.05; random A/zero B; seven specified projections in every layer, no trainable biases |
| OPLoRA | Rank 16, alpha 32, k=128; identical unprojected factors to LoRA-16; exact FP32 starting-weight SVD, factor projections, projected inference export |
| Quantitative evaluation | Complete stock harness splits, zero-shot, FP16/eval mode; GSM8K chat/greedy/1,024 tokens and stock stops/filters; completion prompts for the other three tasks |
| Diagnostics | Shared original FP32 bases; down_proj layers 0/13/27; FP32 local updates on 32 math + 32 non-math fixed baseline prompt activations |
| Qualitative | All 24 supplied prompts × seven conditions, greedy, thinking disabled, cap 512 tokens; first/last underlying questions form the fixed 42-response worksheet |

If FP16 overflow skips an optimizer step, its examples are still consumed once and the prescribed data/scheduler position advances; the skip and scaler change are recorded. It is not silently retried with duplicate training data. `successful_updates` and `skipped_updates` make this visible.

The L2-SP penalty is evaluated before each update, matching that update's SFT loss. Full-model displacement is measured after the required updates. A zero weight update has undefined `rho` and is recorded as JSON `null` with `zero_update=true`. Adapter diagnostics use exported FP32 projected factors before FP16 merge rounding. Merged/unmerged logits are checked on a small two-prompt batch using fixed FP16 tolerances recorded alongside actual maximum and mean errors.

The response loss uses checkpointed vocabulary-head chunks of 256 supervised tokens to bound peak memory; its objective and gradients match full-logit response-only CE up to floating-point summation. Training segments stop only at the specified recovery boundaries; resuming preserves the full 40,000-example/1,250-update recipe. `evaluate --benchmark TASK` runs one complete split and publishes the condition summary after all four have completed.

## Evidence and your remaining analysis

All experiment evidence is written under `evidence/task1` by default:

| Location | Saved evidence |
| --- | --- |
| `protocol.json`, `asset_lock.json`, `preparation.json` | Full settings, immutable asset revisions, filtering counts, prepared-data hashes |
| `training_ids.json`, `probe_ids_and_prompts.json` | Retained source indices/hashes; exact 64 probe IDs and benchmark-formatted prompts |
| `task_definitions.json`, `evaluation_splits.json` | Stock task definitions, complete split sizes/fingerprints, dataset revisions |
| `runs/<method>/` | Per-update SFT/LR/objective/AMP/resource logs, complete environment, checkpoint inventories, required displacements |
| `bases.json`, `activation_cache.json` | FP32 basis/probe-cache provenance, hashes, preprocessing/storage costs |
| `evaluations/<condition>/` | Aggregate results and stock per-example scores, candidate likelihoods, extracted GSM8K answers, raw generation tokens/truncation flags, merge check |
| `diagnostics/<condition>.json` | rho/eta and projection residuals at all three layers, all 64 local activation ratios and math/non-math summaries |
| `qualitative/` | All 168 responses, grouping by question/format, exact prompts/tokens/truncation, 42-row annotation worksheet |
| `tables/`, `figures/` | Seven-condition scores and percentage-point deltas, resource/regularization/diagnostic tables, combined learning curves and benchmark/regularization plots |
| `completeness.json` | Explicit complete/missing evidence audit |

After `summarize`, fill `qualitative/annotations.csv` yourself. It includes all 42 required responses and fields for format compliance, coherence/relevance, repetition, unnecessary mathematics, inappropriate content, short excerpts, and notes. Record your selection rule for aligned illustrative excerpts. `summarize` never overwrites an existing worksheet. Re-run `audit` afterward; it exits with status 1 while required evidence or annotations are incomplete.

Use the saved tables and aligned responses to write your own controlled comparisons and interpretation: rank; Full FT versus L2-SP; LoRA-16 versus OPLoRA; Full FT versus LoRA-16; and Full FT midpoint versus final. Local activation ratios isolate local weight changes and do not establish causes of whole-model behavior. The code does not generate report language or subjective annotations.

## Code layout and future tasks

`src/atml/common/` contains task-independent persistence, provenance, and RNG utilities. `src/atml/task1/` separates configuration, data, adapter algebra, model loading, training, harness integration, evaluation, diagnostics, qualitative generation, and summaries. Future work can add `src/atml/task2/` and `src/atml/task3/` with their own CLI entry points and `artifacts/taskN` / `evidence/taskN` folders. They do not need to change Task 1's protocol or import its training loop.

## Sources and authorship

The assignment manual supplies the experimental specification. Mathematical background: [OPLoRA, Sections 3.2–3.3 and Proposition 2](https://arxiv.org/html/2510.13003v1), [LoRA](https://arxiv.org/abs/2106.09685), and [L2-SP](https://arxiv.org/abs/1802.01483). API references: [Qwen non-thinking chat formatting](https://qwen.readthedocs.io/en/stable/getting_started/quickstart.html), [Qwen3-0.6B model/tokenizer](https://huggingface.co/Qwen/Qwen3-0.6B), and [lm-evaluation-harness v0.4.9.1](https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.4.9.1). These were consulted for formulas, required task definitions, and library interfaces; no external experiment implementation was pasted or adapted. Coding assistance from an LLM was used; you remain responsible for understanding submitted code under the manual's policy.
