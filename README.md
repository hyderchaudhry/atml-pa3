# ATML PA3 — Task 2 on an RTX 4090

Task 2 is implemented with original visual-interface, LoRA, curriculum training, decoding, scoring and recovery code. The [assignment manual](ATML-PA3.pdf) is the source of truth. PyTorch and Transformers supply tensor operations and pretrained architectures; no external VLM trainer, PEFT adapter implementation or public assignment solution is used.

The main run uses **196 SigLIP patch tokens**, original **Qwen3-0.6B**, and **4,500 applied optimizer updates** across three stages. Training stays **microbatch 8 × accumulation 2**; final greedy generation stays **batch 8**, as prescribed. Checkpoints save every **250 updates**, with verified private Hugging Face backups and automatic resume.

**Status:** all 15 offline synthetic smoke-test groups pass; real Task 2 training has not run. GPU runtime, peak memory, full FP16 behavior and real-data selection must be validated on your rented machine. Smoke fixtures and outputs are automatically deleted. Task 1's completed evidence remains in [evidence/task1](evidence/task1); its 42 human annotations remain pending. Task 3 is not implemented.

## 1. Set up Vast or RunPod

Use a Linux RTX 4090 machine with 24 GB VRAM, a CUDA 12.4-compatible NVIDIA driver, CPython 3.10–3.12 (3.11 recommended), Internet access, and persistent storage. Allow approximately **40 GB free disk and 32 GB host RAM** as an operating allowance. Choose a persistent mount using your provider's storage settings; the directory name alone does not establish persistence.

In the **cloud terminal**:

```bash
git clone https://github.com/hyderchaudhry/atml-pa3.git
cd atml-pa3

export PA3_ROOT=/workspace/pa3-task2
export PA3_TASK2_BACKUP_REPO=hyderchaudhry/atml-pa3-task2

bash scripts/cloud-task2.sh setup
bash scripts/cloud-task2.sh auth
bash scripts/cloud-task2.sh doctor
bash scripts/cloud-task2.sh smoke
```

Use your own writable Hugging Face namespace if different. `auth` requests a Write token through a hidden terminal prompt and creates no token-containing evidence. The backup is a **private dataset**, separate from Task 1. If `python3.11` is unavailable but another supported interpreter exists, set `PA3_PYTHON`, e.g. `export PA3_PYTHON=python3`, before setup. The script creates an isolated environment on the persistent disk.

## 2. Prepare and inspect labels

```bash
bash scripts/cloud-task2.sh prepare
bash scripts/cloud-task2.sh features
bash scripts/cloud-task2.sh backup
```

The code selects official COCO/VQAv2 records with seed 6304, saves all IDs and fixed image controls, downloads only selected images, and caches frozen FP16 SigLIP patches. Open/download:

```text
/workspace/pa3-task2/evidence/task2/grounding_labels.html
/workspace/pa3-task2/evidence/task2/grounding_review.csv
```

Inspect annotation-derived labels independently of model predictions, especially omitted objects and ambiguous counts. Fill `label_checked` and `notes` in the CSV; write `none` for no issue. Investigate problems before training. The audit tracks human inspection separately and never invents completed judgments.

## 3. Run the experiment

Use `tmux` to survive SSH disconnects:

```bash
tmux new -s pa3-task2
# From the repository directory, inside tmux:
export PA3_ROOT=/workspace/pa3-task2
export PA3_TASK2_BACKUP_REPO=hyderchaudhry/atml-pa3-task2
set -o pipefail
bash scripts/cloud-task2.sh run 2>&1 | tee -a "$PA3_ROOT/task2-run.log"
```

Detach with **Ctrl-B, then D**. Reattach with `tmux attach -t pa3-task2`. The workflow runs calibration, initial development evaluation, the complete curriculum, all scheduled development evaluations, final four-condition evaluation, and summary/export. It prints every applied update and checkpoint path.

RTX 4090 throughput optimizations include SDPA attention, selective gradient checkpointing that retains expensive matrix products, response-only vocabulary projection, a VRAM-budgeted frozen-feature cache, development likelihood batches of 32, KV-cache greedy decoding, and feature-extraction batches starting at 512 with OOM fallback. Feature residency uses up to 25% of VRAM while reserving 12 GiB for model work. Training and final-generation batch sizes remain fixed by the manual. Actual utilization and peak memory are measured during the real run; no speedup is claimed from local CPU tests.

## 4. Resume or check progress

After an interruption, run the **same command**:

```bash
bash scripts/cloud-task2.sh run
```

It resumes the last committed checkpoint and finishes any missing due development evaluation before advancing. On a fresh disk, it automatically restores the published private backup. Keep the same Git revision and environment.

```bash
bash scripts/cloud-task2.sh next
bash scripts/cloud-task2.sh audit
bash scripts/cloud-task2.sh backup
# Explicit restore into a fresh experiment root:
bash scripts/cloud-task2.sh restore
```

A checkpoint contains connector/adapters, optimizer, scheduler position, dynamic scaler, sampler position/permutation/RNG, global RNG, stage progress, training history, mixture counts and resource measurements. SHA-256 receipts protect local files. Remote inventories reference verified immutable commits. Failed uploads leave local checkpoints intact. No checkpoint is automatically deleted. A destroyed disk recovers through the last successfully published inventory.

## 5. Download results

The completed workflow prints:

```text
/workspace/pa3-task2/task2-evidence.zip
```

To recreate it:

```bash
bash scripts/cloud-task2.sh summarize
bash scripts/cloud-task2.sh export
```

Download that absolute path using the provider's file browser, or run **on your local computer**:

```bash
scp -P YOUR_SSH_PORT root@YOUR_SSH_HOST:/workspace/pa3-task2/task2-evidence.zip .
unzip task2-evidence.zip -d evidence
```

The ZIP's `task2/` directory becomes **`evidence/task2/`** beside existing Task 1 evidence. It includes raw per-example results, tables, plots, image worksheets and the completeness audit. Model recovery checkpoints stay in the private backup and can be downloaded with `restore`.

Important output paths under `$PA3_ROOT`:

| Path | Purpose |
| --- | --- |
| `artifacts/task2/checkpoints/update_XXXX/` | Complete recovery state and separate connector/adapter weights |
| `evidence/task2/selection.json` | Fixed data IDs, references, complementary pairs and mismatch mapping |
| `evidence/task2/training.jsonl` | Applied-update losses, sampled mixture, learning rates, AMP retries and timing |
| `evidence/task2/development/` | Scheduled caption/reference-likelihood evidence |
| `evidence/task2/final/` | Original references, input-image controls, raw greedy tokens/text, NLL and timings |
| `evidence/task2/tables/` | Suite/type/condition results, pair diagnostics, 4,000-pair bootstrap intervals and resources |
| `evidence/task2/plots/learning_curves.png` | Training/development curves with stage boundaries |
| `evidence/task2/qualitative.html` | Aligned image examples selected independently of results |
| `evidence/task2/completeness.json` | Computational checks and outstanding human inspection |

## Implementation and verification

See the [manual requirements map](docs/task2_requirements.md) and [detailed cloud guide](docs/cloud_task2.md) for implementation choices, backup semantics and all output paths. Pinned Task 2 dependencies are in [requirements-task2.txt](requirements-task2.txt); source is under [src/atml/task2](src/atml/task2).

Offline smoke tests use randomly initialized tiny models and synthetic data. They cover loss/gradient equivalence, frozen parameters, adapters, masks/positions, generation, vision normalization, data isolation, paired sampling, optimizer/RNG recovery, upload/checksum failure handling, final scoring, summaries and ZIP export. Temporary fixtures are removed on exit. Full experiments require CUDA and cannot silently fall back to CPU/MPS. Report interpretation and prose remain your work under the manual's AI policy.

For Task 1, use the [Kaggle guide](docs/kaggle.md), [execution checklist](docs/task1_execution.md), and [saved evidence](evidence/task1). Task 1 and Task 2 use separate output folders and backup repositories.
