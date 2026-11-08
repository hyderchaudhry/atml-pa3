# Task 3 on a cloud RTX 4090

The assignment manual, [ATML-PA3.pdf](../ATML-PA3.pdf), pp. 11–14, defines the experiment. The implementation uses the supplied 64,000 rollouts for offline analysis and generates only the prescribed 250 new MATH responses. Model weights remain fixed. No training, VLM inference, images, quantization, or extra rollout generation is required.

## Cloud setup

Use a Linux CUDA machine with an RTX 4090 (24 GB VRAM), CPython 3.10–3.12 (3.11 recommended), a driver compatible with PyTorch's CUDA 12.4 wheel, Internet access and persistent storage. Allow about 40 GB free disk and 16–32 GB host RAM as an operating allowance. These are storage/RAM allowances, not measured GPU utilization or completion-time claims. Confirm persistence with the provider; a `/workspace` directory alone does not establish persistence.

Run in the **cloud terminal**:

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

Use your writable Hugging Face namespace. `auth` asks for a Write token through a hidden prompt. Backups must be a **private dataset**. Keep Task 1, 2 and 3 output roots and backup repositories separate. Setup automatically finds an installed CPython 3.10–3.12, preferring 3.11. To select one explicitly, run `PA3_PYTHON=python3 bash scripts/cloud-task3.sh setup`, or set `PA3_PYTHON` to its executable path. An explicit incompatible selection fails with a clear message. The launcher creates `$PA3_ROOT/venv-task3` and records `environment-freeze.txt`. Package pins are in [requirements-task3.txt](../requirements-task3.txt).

The launcher sets persistent Hugging Face, Matplotlib and download caches under `$PA3_ROOT/cache`. To share an existing model cache on the same persistent mount, set `PA3_HF_HOME` before commands. It also selects Matplotlib's non-interactive Agg backend before importing Matplotlib, including in notebook-based templates.

## Run the whole task

Use `tmux` to survive closing the SSH connection:

```bash
tmux new -s pa3-task3
# Inside tmux, from the repository directory:
export PA3_ROOT=/workspace/pa3-task3
export PA3_TASK3_BACKUP_REPO=hyderchaudhry/atml-pa3-task3
set -o pipefail
bash scripts/cloud-task3.sh run 2>&1 | tee -a "$PA3_ROOT/task3-run.log"
```

Detach with Ctrl-B then D; reconnect with `tmux attach -t pa3-task3`. The workflow downloads the manual-linked supplied files and README, validates them, saves splits and ten candidate orders, runs development analysis, freezes thresholds, evaluates held-out questions, runs live inference, then exports tables, figures and evidence. Completed work is reused. The source downloads total approximately 419 MB. Only the pinned Qwen2.5-3B-Instruct model and its tokenizer are downloaded for live inference.

The manual-linked MathVista `testmini.json` is **question metadata**, obtained from its official repository. The downloader resolves its commit once and preserves its exact bytes and SHA-256. This is the only GitHub data fetched by Task 3; no external implementation, selection code, trainer, or assignment solution is fetched. PyTorch/Transformers provide the pretrained architecture, and `math-verify` supplies the scorer required by the manual.

If you already have `ATML-PA3-Data`, set its location before `run`:

```bash
export PA3_TASK3_DATA=/workspace/ATML-PA3-Data
```

That folder must directly contain `README.MD`, `qwen25-math/` and `qwen25vl-mathvista/`. Leave the original files intact. Optional global flags go after the launcher command, for example:

```bash
bash scripts/cloud-task3.sh prepare --source /workspace/ATML-PA3-Data --metadata /workspace/testmini.json
bash scripts/cloud-task3.sh run --workers 4
```

The metadata argument must be the released MathVista `testmini.json`, not a custom answer file. After preparation, the immutable SQLite cache and metadata are sufficient for recovery; the supplied Parquet files are not regenerated.

## Individual stages and progress

```bash
bash scripts/cloud-task3.sh next
bash scripts/cloud-task3.sh prepare
bash scripts/cloud-task3.sh development
bash scripts/cloud-task3.sh fit
bash scripts/cloud-task3.sh evaluation
bash scripts/cloud-task3.sh live
bash scripts/cloud-task3.sh summarize
bash scripts/cloud-task3.sh audit
bash scripts/cloud-task3.sh backup
```

`fit` reads development results exclusively and writes `policies.json` once. Evaluation requires that frozen file. `run` invokes these stages and verified backups automatically; individual stage commands save local progress, so explicitly run `backup` after using them. `audit` distinguishes missing outputs from completed computation. It does not produce report interpretation.

Offline work uses four spawned CPU workers by default (or fewer on a smaller host), compressed question-sized arrays, vectorized reductions, and a disk-backed SQLite join. Each worker operates on one question at a time; no all-layer attention tensors or full model distributions are constructed. Use `--workers 1` to reduce host RAM use. Native BLAS/OpenMP threads default to one per worker to avoid oversubscribing CPUs. Supplied probability and attention measurements are used directly.

Live inference keeps one frozen FP16 model resident and runs **batch 1**, as required. SDPA attention, incremental KV caching and projection of only the last hidden state avoid unnecessary attention matrices and prompt-wide vocabulary logits. Sampling filters operate on the top-20 candidate set after the full last-token vocabulary projection. The model and KV cache fit comfortably within the intended 24 GB GPU class; real peak allocation is measured per call. Filling all 24 GB is not a performance target, and the prescribed batch size is not increased to consume spare memory.

You can complete CPU work before renting a GPU. In an appropriate Python environment, install `requirements-task3.txt` (normal platform torch wheel) and run from the repository:

```bash
export PA3_ROOT=/your/persistent/task3-directory
export PA3_TASK3_BACKUP_REPO=hyderchaudhry/atml-pa3-task3
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1
python -m atml.task3 auth
python -m atml.task3 analyze
```

`analyze` runs through held-out offline analysis, summaries and backup; its audit intentionally leaves live evidence pending. On the GPU, use the cloud setup and the same backup repository with `run` to restore and complete live inference. Keep the same Git revision. The entire live comparison must remain on the **same physical GPU** once it starts.

## Checkpoints and recovery

Run the same command after an interruption:

```bash
bash scripts/cloud-task3.sh run
```

Local checkpoints:

- Parquet ingestion commits every 128 rows with a digest-bound cursor; interrupted ingestion resumes committed rows.
- Offline analysis atomically commits each completed question. Remote backup runs after every 50 questions and at stage boundaries.
- Each live response commits raw/retained tokens, prompt, answer extraction, seed, lengths, cap, time, GPU/session identity and correctness. Each live question's five calls trigger a remote backup.
- A live response also atomically saves a checksum-protected partial checkpoint every 64 generated tokens and at termination/cap. It includes token IDs, CUDA sampling RNG, elapsed time and prior peak allocation. KV tensors are rebuilt by replaying the original prompt and saved one-token forwards; replay performs no new sampling. Replay time is included in resumed call time.

There are no optimizer checkpoints because Task 3 does not train. No completed response is discarded when another call fails. A local crash can lose fewer than 64 tokens of the in-progress call; work since the most recent partial checkpoint is replayed. A destroyed disk can recover only what the last successfully published remote inventory includes. In-progress local partial checkpoints are included when a manual backup is run between attempts. Use the same model revision, source code, package environment and GPU when resuming; exact numerical reproducibility across different hardware/software is not promised.

```bash
# Explicit restore into a fresh root:
bash scripts/cloud-task3.sh restore
```

HF uploads are incremental, in batches of at most 32 files with one concurrent file upload. Remote sizes and SHA-256/LFS or Git blob hashes are checked before publishing an inventory referencing immutable commits. Transient network failures, HTTP 408/429 and common server errors receive up to four attempts with bounded backoff. The underlying failure is printed with signed URL queries and token strings redacted; permanent permission/quota errors are not blindly retried. Each verified batch is recorded locally in `artifacts/task3/backup_progress.json`, so a later failed batch can resume without re-uploading earlier verified batches. The published recovery inventory advances only after all files are verified. Failure leaves local progress intact and stops the workflow; retry the command or `backup`. Restore rechecks content and refuses to overwrite conflicting local files. Source/scorer/order/implementation fingerprints guard result reuse; GPU UUID checks prevent mixing physical devices in the live comparison. Execution sessions are retained separately. No model weights are duplicated into the evidence backup because the original model is revision-pinned in HF's cache.

## Evidence and download

The final command prints:

```text
/workspace/pa3-task3/task3-evidence.zip
```

The archive contains `task3/`. Run **on your local computer**, using the provider's SSH details:

```bash
scp -P YOUR_SSH_PORT root@YOUR_SSH_HOST:/workspace/pa3-task3/task3-evidence.zip .
unzip task3-evidence.zip -d evidence
```

If your SSH key is not the default, use `scp -i /path/to/private_key -P ...`. This produces `evidence/task3/` beside Tasks 1 and 2. The ZIP includes saved question-level results, raw live responses, tables, figures, provenance and the audit. It excludes the prepared database, raw supplied Parquet files, cached models and temporary partial files. For continued computation, use the private backup's `restore`, not only the evidence ZIP.

| Path under `evidence/task3/` | Contents |
| --- | --- |
| `protocol.json`, `splits.json`, `orders.json` | Fixed manual settings, question IDs, ten nested rollout orders |
| `preparation.json`, `asset_lock.json`, `supplied_README.MD` | Source schemas/metadata/hashes and original model revision |
| `scorer.json`, `environment.json` | Fixed scorer version, implementation hashes and environment |
| `offline/{dataset}/{development,evaluation}/*.json.gz` | Per-question/per-order candidate IDs, selected answers, costs, failures and signals |
| `policies.json` | Frozen development-selected controllers and development file hashes |
| `live/*/*.json` | All 250 raw live responses with prompts, independent seeds, costs and grading |
| `live_sessions/`, `live_gpu.json` | Every execution environment and physical GPU identity |
| `tables/` | Sampling/selection, paired differences, signal associations, adaptive grids/stops, live costs/transitions |
| `plots/` | Accuracy/coverage/cost curves, adaptive comparison and fixed token trajectories |
| `illustrative_selection.json`, `live_examples.json`, `qualitative.html` | Stated example-selection rules and aligned raw evidence |
| `completeness.json` | Exact computational checks and pending outputs |

All reported offline intervals use 1,000 paired question bootstrap resamples after averaging the ten orders within each question. The live paired differences bootstrap the 50 questions. Undefined correlations remain null and their defined counts are retained. Costs distinguish the offline retained-token proxy from live physical generated tokens, including termination and critique tokens. Live inference timings include tokenization, prefill, generation, decoding and partial-checkpoint overhead; model loading, grading and network backups are excluded. A resumed response also includes cache replay and time recorded before interruption. Timings do not treat the three serial batch-one samples as concurrent wall time.

Smoke testing uses synthetic Parquet files and randomly initialized tiny models, plus a mocked 250-call cloud orchestration and in-memory Hub. It downloads no pretrained model and runs no actual assignment experiment. Temporary fixtures and results are automatically removed. CUDA/FP16 throughput and full-data completion must be validated on the rented machine. Report writing and interpretation remain your own work under the manual's AI policy.
