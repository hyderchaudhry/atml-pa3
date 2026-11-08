# ATML PA3 — Tasks 1, 2 and 3

The [assignment manual](ATML-PA3.pdf) is the source of truth. Experiment code is original: no public assignment solution, external trainer, or GitHub selection implementation is used. Libraries provide tensor operations, pretrained architectures, data transport and the manual-required `math-verify` scorer. Task 3 downloads the supplied rollouts and official MathVista question metadata as data.

**Task 3 is implemented for a cloud RTX 4090 with 24 GB VRAM.** It analyzes the supplied MATH/MathVista rollouts, selects and freezes adaptive policies on development questions, then runs the prescribed 50-question live MATH comparison. Live inference stays **FP16, no quantization, batch 1**, with the exact manual prompts, sampling settings and output allowances.

Eleven synthetic smoke-test groups pass, including actual Parquet ingestion, spawned CPU workers, scorer/equivalence checks, gold-independent selection/stopping, interrupted tiny-model sampling, the 250-call live orchestration, verified backup/restore and export. Test fixtures and outputs are removed. The full Task 3 experiment and real CUDA throughput have not been run locally.

## Run Task 3 on Vast or RunPod

Use a Linux CUDA template with Python 3.11, a CUDA 12.4-compatible NVIDIA driver, persistent storage and Internet. Allow approximately 40 GB free disk and 16–32 GB host RAM. Run these commands in the **cloud terminal**:

```bash
git clone https://github.com/hyderchaudhry/atml-pa3.git
cd atml-pa3
export PA3_ROOT=/workspace/pa3-task3
export PA3_TASK3_BACKUP_REPO=hyderchaudhry/atml-pa3-task3

bash scripts/cloud-task3.sh setup
bash scripts/cloud-task3.sh auth
bash scripts/cloud-task3.sh doctor
bash scripts/cloud-task3.sh smoke
```

`auth` requests your Hugging Face Write token without echoing it. Use your own writable namespace if different. The recovery dataset must be private. Confirm that the chosen `PA3_ROOT` is on the provider's persistent disk. The launcher keeps the virtual environment, original model cache and experiment outputs there.

Start `tmux`, then run from the repository directory:

```bash
tmux new -s pa3-task3
export PA3_ROOT=/workspace/pa3-task3
export PA3_TASK3_BACKUP_REPO=hyderchaudhry/atml-pa3-task3
set -o pipefail
bash scripts/cloud-task3.sh run 2>&1 | tee -a "$PA3_ROOT/task3-run.log"
```

Detach with Ctrl-B then D; reconnect with `tmux attach -t pa3-task3`. The command obtains the supplied files, validates alignment, saves IDs/orders, performs development analysis, freezes stopping thresholds, analyzes held-out questions, generates the 250 live responses and creates tables/figures/ZIP. It backs up durable progress automatically.

RTX 4090 optimizations include one resident frozen model, SDPA attention, incremental KV caching and vocabulary projection only at the current token. Offline analysis uses bounded question-sized data, SQLite joins, vectorized reductions and four CPU workers. Spare VRAM is retained; the manual's batch-one comparison is preserved.

To resume after interruption, run the **same `run` command** with the same Git revision. A fresh root automatically restores the published private backup. Local recovery includes per-question results, completed live responses and partial token/RNG checkpoints every 64 generated tokens. Remote backups occur every 50 offline questions, after each live question and at stage boundaries. Physical GPU UUID checks keep the live comparison on the same device.

```bash
bash scripts/cloud-task3.sh next
bash scripts/cloud-task3.sh audit
bash scripts/cloud-task3.sh backup
# Explicit restore into a fresh root:
bash scripts/cloud-task3.sh restore
```

See the [full cloud guide](docs/cloud_task3.md) for local supplied-data paths, CPU-only analysis before renting, individual stages, timing semantics and recovery details. The [manual requirements map](docs/task3_requirements.md) links each requirement to its implementation. Source: [src/atml/task3](src/atml/task3); pinned environment: [requirements-task3.txt](requirements-task3.txt).

## Download Task 3 results

The workflow prints **`/workspace/pa3-task3/task3-evidence.zip`**. Recreate it with `bash scripts/cloud-task3.sh summarize`. Download it through the provider's file browser, or run **on your local computer**:

```bash
scp -P YOUR_SSH_PORT root@YOUR_SSH_HOST:/workspace/pa3-task3/task3-evidence.zip .
unzip task3-evidence.zip -d evidence
```

Add `-i /path/to/private_key` if needed. The ZIP's `task3/` folder becomes `evidence/task3/`. It contains question-level results, raw live prompts/responses, tables, plots, example galleries and the completeness audit. Continued computation uses the private backup; cached models and the prepared source database are excluded from the evidence ZIP.

## Reproduce Tasks 1 and 2

Keep each task's root and Hugging Face backup separate.

| Task | Run instructions | Source | Saved local evidence |
| --- | --- | --- | --- |
| 1: adaptation/forgetting | [Kaggle guide](docs/kaggle.md), [execution checklist](docs/task1_execution.md) | [src/atml/task1](src/atml/task1) | [evidence/task1](evidence/task1) |
| 2: visual interface/curriculum | [RTX 4090 cloud guide](docs/cloud_task2.md), [manual map](docs/task2_requirements.md) | [src/atml/task2](src/atml/task2) | [evidence/task2](evidence/task2) |
| 3: test-time compute | [RTX 4090 cloud guide](docs/cloud_task3.md), [manual map](docs/task3_requirements.md) | [src/atml/task3](src/atml/task3) | Created by the new run |

For a new Task 2 experiment, use `PA3_ROOT=/workspace/pa3-task2`, `PA3_TASK2_BACKUP_REPO=your-user/atml-pa3-task2` and the `scripts/cloud-task2.sh` launcher. Its prescribed microbatch 8 × accumulation 2, final batch 8, 4,500 applied updates and 250-update checkpoints are unchanged. Task 1 uses `scripts/kaggle.sh` as described in its guide.

The supplied Task 1/2 evidence is retained. Human annotation/review fields are separate from computational completion; see each task's audit. Report analysis and prose must be written by you under the manual's AI policy. Smoke commands use temporary synthetic fixtures and delete their outputs automatically; they do not replace a completed real experiment.
