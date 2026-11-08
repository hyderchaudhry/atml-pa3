# Task 2 on Vast or RunPod

The assignment manual, [ATML-PA3.pdf](../ATML-PA3.pdf), pp. 7–10, defines the experiment. This guide runs the required 196-token main recipe on a single CUDA GPU. The implementation starts from original `Qwen/Qwen3-0.6B`, independently of Task 1.

## Prepare the cloud machine

Use an RTX 4090, a Linux image with CPython 3.10–3.12 (3.11 recommended), a compatible NVIDIA driver for PyTorch's CUDA 12.4 wheel, Internet access, and persistent storage. Allow approximately 40 GB of free disk and 32 GB of host RAM as an operating allowance; actual disk, runtime, and GPU memory depend on the selected images and are recorded. This is not a measured completion-time or memory guarantee. `setup` creates an isolated environment and does not rely on packages supplied by the provider's notebook kernel.

Point `PA3_ROOT` at the mounted persistent disk. On RunPod, consult [storage options](https://docs.runpod.io/pods/storage/types): a volume disk and a network volume have different lifetimes. On Vast, [local volumes](https://docs.vast.ai/guides/instances/storage/volumes) are tied to a physical host. The private Hugging Face backup supports recovery on another machine. A directory named `/workspace` alone does not establish the lifetime of its storage.

In the cloud terminal:

```bash
git clone https://github.com/hyderchaudhry/atml-pa3.git
cd atml-pa3
export PA3_ROOT=/workspace/pa3-task2
export PA3_TASK2_BACKUP_REPO=hyderchaudhry/atml-pa3-task2
# If needed, select another installed supported interpreter:
# export PA3_PYTHON=python3
bash scripts/cloud-task2.sh setup
bash scripts/cloud-task2.sh auth
bash scripts/cloud-task2.sh doctor
bash scripts/cloud-task2.sh smoke
```

`auth` reads a Hugging Face Write token using a hidden terminal prompt. Use your own writable namespace for the backup repository. It is created as a **private dataset**, separately from Task 1. Tokens are cached by Hugging Face under `$PA3_ROOT/cache/huggingface`; they are not placed in evidence or command arguments. Run smoke before downloading data. It uses randomly initialized tiny models and synthetic records, and deletes its fixtures even on failure.

## Inspect the annotation labels

Prepare the deterministic selection and frozen feature cache:

```bash
bash scripts/cloud-task2.sh prepare
bash scripts/cloud-task2.sh features
bash scripts/cloud-task2.sh backup
```

Preparation downloads official COCO 2014 and VQAv2 JSON archives and only the selected COCO images. Feature caching uses the official pinned SigLIP processor, no random augmentations, one final layer normalization, and FP16 patch shards. SigLIP is removed from GPU memory before Qwen loads.

For the 24 GB 4090, vision extraction starts at batch 512 and halves that operational batch after a CUDA OOM; its selected batch is recorded. Training uses selective nonreentrant gradient checkpointing, retaining expensive matrix-product outputs while recomputing cheaper operations. Up to 25% of total VRAM holds frozen feature shards, while preserving 12 GiB of free model-work allowance; remaining features stream through a host-memory cache. Development reference likelihood uses batch 32. Final generation remains batch 8, and training remains microbatch 8 × accumulation 2. SDPA, response-only vocabulary projection in chunks up to 512 tokens, and batched KV-cache decoding reduce overhead without changing the prescribed data/loss/budgets. GPU throughput and memory figures come from the cloud run, not local synthetic tests.

Open `$PA3_ROOT/evidence/task2/grounding_labels.html` (download it from the terminal/file browser if necessary). It embeds thumbnails and annotation-derived labels, without predictions. Inspect for missing objects and ambiguous counts independently of model results. Record `label_checked` and `notes` in `grounding_review.csv`; use `none` for no ambiguity. Preserve the generated IDs and labels. Investigate any label problems before running the experiment; do not silently edit a frozen selection or reinterpret results after seeing model predictions. The completeness audit separately reports this human work as pending until filled. The implementation does not invent inspection judgments or report interpretation.

## Run and resume

Use a terminal multiplexer so closing SSH does not stop training:

```bash
tmux new -s pa3-task2
# In the tmux shell, from the repository directory:
export PA3_ROOT=/workspace/pa3-task2
export PA3_TASK2_BACKUP_REPO=hyderchaudhry/atml-pa3-task2
set -o pipefail
bash scripts/cloud-task2.sh run 2>&1 | tee -a "$PA3_ROOT/task2-run.log"
```

Detach with Ctrl-B then D. Reconnect with `tmux attach -t pa3-task2`. If `tmux` is unavailable, install it using the image's package manager, or keep the terminal session open. Logs print every applied update, AMP retries, and checkpoint/evaluation progress.

Run the same `run` command after an interruption. It finds the last committed checkpoint, completes any due development evaluations, and continues. On a fresh disk, it restores the published backup automatically before selecting the next step. No explicit `--resume` flag is needed. Use the same Git revision and pinned environment when resuming; the saved environment and source hashes support reproducibility.

```bash
bash scripts/cloud-task2.sh next
bash scripts/cloud-task2.sh audit
# Explicit fresh-workspace restore, if desired:
bash scripts/cloud-task2.sh restore
```

Checkpoints occur at updates 250, 500, …, 4500, plus an initial calibrated checkpoint at 0. Checkpoints 2250, 3750, and 4500 include the stage-end connector/adapter weights. Each checkpoint has `recovery.pt`, `weights.pt`, and checksummed `complete.json`. Recovery includes optimizer moments, stage scheduler position, dynamic scaler, sampler RNG/order/position, global RNG, training history, mixture counts and resource measurements. Stage transitions continue the weights and initialize a fresh stage-specific optimizer/scheduler/scaler. An AMP overflow retries the same batch and dropout RNG; it does not consume an applied update or another sampler draw.

The launcher backs up after each completed step, including every training checkpoint. Payloads are uploaded in bounded batches, verified by remote size and content hash, and referenced by immutable commit. A new inventory is published only after verification. Upload failure leaves local checkpoints intact and stops the workflow; retry `run` or `backup`. No checkpoint is automatically deleted. Recovery on a destroyed disk uses the last successfully published inventory. Local evaluation journals save each completed batch; a new disk may repeat work since the preceding remote backup. Raw image/annotation downloads are reproducible inputs and are not duplicated in the backup; feature shards, tokenizer, selections, evidence and all recovery checkpoints are backed up.

Only one Task 2 process can mutate a given output root at a time. Keep distinct experiment roots and backup names for any independent experiment.

## Results and download

After training, the workflow runs final evaluation and creates tables, curves, image examples, a completeness audit, and:

```text
$PA3_ROOT/task2-evidence.zip
```

To recreate the ZIP or summary:

```bash
bash scripts/cloud-task2.sh summarize
bash scripts/cloud-task2.sh export
```

Download the printed absolute ZIP path from the provider's file browser, or run this **on your local computer**, using your provider's SSH host and port:

```bash
scp -P YOUR_SSH_PORT root@YOUR_SSH_HOST:/workspace/pa3-task2/task2-evidence.zip .
unzip task2-evidence.zip -d evidence
```

The ZIP contains `task2/`; extraction gives `evidence/task2/` beside the existing `evidence/task1/`. It contains evidence, not model checkpoints. To recover checkpoints, use `restore` against the private backup. To download the backup without a GPU, install `requirements-task2.txt` in a supported Python environment and run `PYTHONPATH=src python -m atml.task2 --root /your/new/directory --repo hyderchaudhry/atml-pa3-task2 restore` after Hugging Face authentication.

Principal outputs:

| Path under `evidence/task2/` | Contents |
| --- | --- |
| `protocol.json`, `asset_lock.json`, `sources.json`, `selection.json` | Recipe, pinned model revisions, archive hashes, all selected IDs, fixed mismatches |
| `calibration.json`, `feature_cache.json`, `interface_checks.json` | Training-only scale rule/value, preprocessing/cache costs, gradients/freezing checks |
| `training.jsonl`, `resources.json` | Per-applied-update loss/mixture/LRs/time and stage parameter/memory costs |
| `development/update_XXXX/` | Batch-resumable caption and VQA reference-likelihood journals |
| `final/` | Original references, partner references, input image IDs, raw greedy tokens/text, metrics, timings |
| `tables/final.csv`, `tables/correct_image_advantages.csv` | Suite/type/condition metrics and 95% complete-pair bootstrap intervals |
| `tables/development.csv`, `tables/resources.csv` | Development trajectory and training costs |
| `plots/learning_curves.png` | Training loss, caption loss, reference NLL and swap penalties, with stage boundaries |
| `grounding_labels.html`, `grounding_review.csv`, `qualitative.html` | Label inspection and fixed-selection aligned image examples |
| `completeness.json` | Computational checks and separately pending human inspection |

The manual's report discussion and interpretation remain your work. Final scores are local subset scores. Runtime, full-model FP16 behavior, cloud connectivity and real-data eligibility are validated when you execute on the cloud GPU; local smoke tests do not establish those results.
