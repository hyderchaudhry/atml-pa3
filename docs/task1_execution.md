# Task 1: Kaggle execution checklist

Implementation and synthetic verification are complete only when the checks in
`evidence/task1/system/smoke.json` pass. **No assignment experiment, benchmark score,
or trained checkpoint is supplied by those checks.** Execute the experiments below
yourself. Write the report and qualitative interpretation yourself.

Use the clone/setup and authentication cells in [kaggle.md](kaggle.md) first.
All commands below are Kaggle code-cell commands, run from `/kaggle/working/PA3`.
The shell launcher selects the isolated Python and output locations; there is no
repository notebook to run. Keep one writing session per experiment/backup repo.

## Execution controls

The defaults are **32 prompts per generation batch**, **32 distinct prefixes per
likelihood batch**, and four FP32 activation probes per batch. The likelihood
backend shares identical candidate prefixes and projects only continuation hidden
states, in vocabulary chunks of 256 tokens. It preserves stock tokenization,
conditional likelihood sums, softmax dtype, task normalization, and scoring.
Generation uses the stock harness's stop matchers independently for each row.

The defaults need no extra configuration. To make them explicit in a Kaggle cell:

```python
import os
os.environ["PA3_GENERATION_BATCH"] = "32"
os.environ["PA3_LIKELIHOOD_BATCH"] = "32"
```

SDPA attention, FP16 inference and KV caching are enabled. PyTorch selects the
available SDPA kernel; FlashAttention-2 is not required. Seed 6304, deterministic
algorithms and disabled TF32 remain in effect. Existing training runs retain their
original attention backend on resume; older checkpoints use eager attention.
New runs record SDPA in their configuration. Kernel/batch changes can cause normal
floating point differences; keep a consistent environment for controlled runs.

These execution controls do **not** change the prescribed training microbatch
2, accumulation 16, effective batch 32, sequence limit, epoch, learning rates,
datasets, splits, prompts, greedy decoding or token caps. A process uses GPU 0
of its visible CUDA devices; two visible T4s do not pool their memory. This is
single-GPU execution, without model sharding or distributed training.

T4 throughput and memory at batch 32 have not been measured locally. If Kaggle
reports an inference OOM, explicitly choose a smaller inference batch using the
same environment variables and rerun the identical command. Completed requests
are reused. Training never reduces its fixed microbatch/length/precision to pass
a memory check. `PA3_ATTENTION=eager` is available for diagnosing SDPA issues;
execution sessions record these choices.

## 1. Synthetic verification and preparation

- [ ] `!bash scripts/kaggle.sh smoke --cuda`
- [ ] `!bash scripts/kaggle.sh doctor`
- [ ] `!bash scripts/kaggle.sh fetch-prompts`
- [ ] `!bash scripts/kaggle.sh prepare`
- [ ] `!bash scripts/kaggle.sh bases`
- [ ] `!bash scripts/kaggle.sh cache-probes`

Smoke is offline: a randomly initialized, tiny Qwen model and artificial strings
exercise the code. Its temporary datasets, weights, caches and checkpoints are
removed automatically. The only persistent smoke output is `system/smoke.json`.
It is separate from the experiments and does not download the assignment model.

These preparation commands produce the exact question YAML and source hash,
`asset_lock.json`, `protocol.json`, `preparation.json`, `training_ids.json`,
`probe_ids_and_prompts.json`, `evaluation_splits.json`, `task_definitions.json`,
`bases.json`, `activation_cache.json` and environment snapshots under evidence.
Large outputs are `prepared/`, `tokenizer/`, `bases.pt`, and
`baseline_activations.pt` under artifacts. Partial SVD modules are saved in
`basis_parts/`, allowing interrupted preprocessing to reuse completed modules.
Probe IDs are fixed here, before inspecting any model results.

## 2. Shared baseline

- [ ] `!bash scripts/kaggle.sh evaluate --condition baseline`
- [ ] `!bash scripts/kaggle.sh qualitative --condition baseline`

Do not run a baseline update diagnostic: its weight update is zero. Baseline
activation caching in step 1 is the reference for all trained-model diagnostics.

## 3. Full FT through its midpoint

- [ ] `!bash scripts/kaggle.sh memory-check --method full_ft`
- [ ] `!bash scripts/kaggle.sh train --method full_ft --resume-latest --stop-after 625 --checkpoint-repo "$PA3_BACKUP_REPO"`
- [ ] `!bash scripts/kaggle.sh evaluate --condition full_ft_midpoint`
- [ ] `!bash scripts/kaggle.sh diagnose --condition full_ft_midpoint`
- [ ] `!bash scripts/kaggle.sh qualitative --condition full_ft_midpoint`

This saves recovery checkpoints at updates 312 and 625. Full FT midpoint
displacement is written to `runs/full_ft/displacement.jsonl`.

## 4. Full FT through its final checkpoint

- [ ] `!bash scripts/kaggle.sh train --method full_ft --resume-latest --stop-after 1250 --checkpoint-repo "$PA3_BACKUP_REPO"`
- [ ] `!bash scripts/kaggle.sh evaluate --condition full_ft_final`
- [ ] `!bash scripts/kaggle.sh diagnose --condition full_ft_final`
- [ ] `!bash scripts/kaggle.sh qualitative --condition full_ft_final`

This resumes Full FT from 625 and saves updates 937 and 1250. Final Full FT
displacement is appended to the same displacement file.

## 5. Independent L2-SP run

- [ ] `!bash scripts/kaggle.sh memory-check --method l2sp`
- [ ] `!bash scripts/kaggle.sh train --method l2sp --resume-latest --checkpoint-repo "$PA3_BACKUP_REPO"`
- [ ] `!bash scripts/kaggle.sh evaluate --condition l2sp_final`
- [ ] `!bash scripts/kaggle.sh diagnose --condition l2sp_final`
- [ ] `!bash scripts/kaggle.sh qualitative --condition l2sp_final`

This starts from the original model. Its curve includes Omega, lambda*Omega,
total objective and reference transfer costs. Final displacement is in
`runs/l2sp/displacement.jsonl`.

## 6. Independent LoRA-4 run

- [ ] `!bash scripts/kaggle.sh memory-check --method lora4`
- [ ] `!bash scripts/kaggle.sh train --method lora4 --resume-latest --checkpoint-repo "$PA3_BACKUP_REPO"`
- [ ] `!bash scripts/kaggle.sh evaluate --condition lora4_final`
- [ ] `!bash scripts/kaggle.sh diagnose --condition lora4_final`
- [ ] `!bash scripts/kaggle.sh qualitative --condition lora4_final`

## 7. Independent LoRA-16 run

- [ ] `!bash scripts/kaggle.sh memory-check --method lora16`
- [ ] `!bash scripts/kaggle.sh train --method lora16 --resume-latest --checkpoint-repo "$PA3_BACKUP_REPO"`
- [ ] `!bash scripts/kaggle.sh evaluate --condition lora16_final`
- [ ] `!bash scripts/kaggle.sh diagnose --condition lora16_final`
- [ ] `!bash scripts/kaggle.sh qualitative --condition lora16_final`

## 8. Independent OPLoRA-16 run

- [ ] `!bash scripts/kaggle.sh memory-check --method oplora16`
- [ ] `!bash scripts/kaggle.sh train --method oplora16 --resume-latest --checkpoint-repo "$PA3_BACKUP_REPO"`
- [ ] `!bash scripts/kaggle.sh evaluate --condition oplora16_final`
- [ ] `!bash scripts/kaggle.sh diagnose --condition oplora16_final`
- [ ] `!bash scripts/kaggle.sh qualitative --condition oplora16_final`

The original unprojected initialization matches LoRA-16. The saved inference
export contains projected factors, with export cost logged in checkpoint metadata.

## 9. Tables, figures, human annotations and download

- [ ] `!bash scripts/kaggle.sh summarize`
- [ ] Fill `qualitative/annotations.csv` yourself: exactly 42 rows, selected from
  the first/last underlying questions in supplied plain order, all formats and
  conditions. Preserve the automatic truncation flags and supply the judgments,
  short excerpts and selection notes yourself.
- [ ] `!bash scripts/kaggle.sh audit`
- [ ] `!bash scripts/kaggle.sh backup`
- [ ] `!bash scripts/kaggle.sh export`

`audit` intentionally exits 1 until experiments and annotations are complete.
It writes `completeness.json`; it does not manufacture missing evidence.
The summary outputs are:

- `tables/benchmarks.csv`: seven rows, four primary metrics, baseline deltas in
  percentage points and available raw accuracies.
- `tables/resources.csv`: five runs and method-specific costs.
- `tables/regularization.csv`: Full FT/L2-SP displacement and objective fields.
- `tables/diagnostics.csv`: six trained conditions × layers 0, 13, 27.
- `figures/learning_curves.png`, `figures/benchmarks.png`, `figures/regularization.png`.
- `qualitative/grouped_responses.json`: all 168 responses aligned by question/format.
- `qualitative/annotations.csv` and `annotation_instructions.json`.

Download `/kaggle/working/pa3/task1-evidence.zip` from Kaggle's output files.
Extract its `task1/` directory into your local repo's `evidence/` directory.
Keep large checkpoints in the private backup. Commit the small final evidence
to GitHub yourself after reviewing it. No report text is generated.

## Output paths for every experiment

Let `E=/kaggle/working/pa3/evidence/task1` and
`A=/kaggle/working/pa3/artifacts/task1`. For each condition `C` in the commands
above, and each `B` in `gsm8k_cot`, `hellaswag`, `boolq`, `arc_easy`:

| Command | Expected outputs relative to E |
| --- | --- |
| `evaluate --condition C` | `evaluations/C/B_results.json`, `B_samples.jsonl`, `B_requests.jsonl`, `B_requests.scope.json`; GSM8K also `B_tokens.jsonl`; then `evaluations/C/summary.json` |
| `evaluate --condition C --benchmark B` | Only the selected split files above; summary appears once all four splits are complete |
| Adapter evaluation | Also `evaluations/C/merge_check.json` with passing logit verification |
| `qualitative --condition C` | `qualitative/C.jsonl`, exactly 24 responses including system messages, token IDs and flags |
| `diagnose --condition C` | `diagnostics/C.json`, three layer metrics and all 64 per-probe R(z) values per layer |
| `train --method M ...` | `runs/M/training.jsonl` (1,250 rows at completion), `configuration*.json`, `environment*.json`, `resources.json`, `status.json`, `checkpoints.jsonl` |
| `memory-check --method M` | `system/memory_M.json`; a discarded synthetic maximum-length step, not a training result |

Every evaluation saves `environment.json`, `execution_sessions.jsonl`, and
`timings.jsonl` inside its condition directory. Token records contain unpadded
input IDs, per-row generated IDs, output count, cap, stop reason, actual truncation
flag and execution batch size. GSM8K samples also include stock filter outputs,
flexibly extracted answers and unextractable flags.

AMP-skipped attempts are recorded separately in `runs/M/amp_skips.jsonl` when
they occur. They retry the same effective batch with the reduced scale and restored
dropout RNG, without advancing the examples or LR schedule. `training.jsonl`
counts 1,250 applied updates; resource totals include retry time. Legacy checkpoints
that advanced the data position on skipped updates cannot meet this exact budget
by resuming; restart that affected variant and preserve its old evidence separately.

Every method has four complete recovery checkpoints at
`A/checkpoints/M/update_0312`, `update_0625`, `update_0937`, `update_1250`
(9,984 / 20,000 / 29,984 / 40,000 examples). Each has `recovery.pt` and
`metadata.json`; Full FT/L2-SP also have `model.pt`; adapters have `adapter.pt`
and `projected_factors.pt`. Recovery includes optimizer, analytical scheduler,
scaler, all RNG states and source-order position. With `--checkpoint-repo`, the
launcher uploads and verifies each checkpoint before removing its local copy.
Persistent receipts live at `E/storage/checkpoints/M/update_NNNN.json`; the
private dataset stores files at `checkpoints/M/update_NNNN/` by immutable commit.

For example, independent split commands are:

```python
!bash scripts/kaggle.sh evaluate --condition baseline --benchmark gsm8k_cot
!bash scripts/kaggle.sh evaluate --condition baseline --benchmark hellaswag
!bash scripts/kaggle.sh evaluate --condition baseline --benchmark boolq
!bash scripts/kaggle.sh evaluate --condition baseline --benchmark arc_easy
```

Replace `baseline` with any other listed condition. Scores are published only
after all required examples in a split have been scored: 1,319 / 10,042 / 3,270 /
2,376 respectively. There is no experiment `--limit` switch.

## Interruptions and independent stages

Repeat the same command to resume. For automatic selection of unfinished work use:

```python
!bash scripts/kaggle.sh status
!bash scripts/kaggle.sh next
# Or execute every remaining stage:
!bash scripts/kaggle.sh run
```

After losing a Kaggle session, repeat setup/authentication with the **same private
backup repository**, then `!bash scripts/kaggle.sh restore` before independent
commands. The automatic `run`/`next` commands perform restoration themselves.

Inference requests and token records are committed locally as batches finish.
On restart, only missing requests require model execution; the harness recomputes
the final full-split metrics from cached and new outputs. Completed splits and
conditions are skipped. Partial final JSONL writes are isolated; committed-line
corruption raises an error. Vocabulary tensors are not stored in the journal.

The verified remote evidence bundle is refreshed during inference approximately
every **300 seconds**, at the next completed request, and after each completed
stage. Change the interval with `os.environ["PA3_SYNC_SECONDS"] = "120"` if desired.
Locally completed batches survive a process restart. A destroyed session can lose
work since the last successful remote backup, plus an in-flight batch. Training
resumes exactly from its last required verified recovery boundary; later unsaved
updates repeat. Before update 312, it restarts from the same original seed and
preserves the interrupted attempt's curve. No code can recover state that had not
yet been persisted. Wait for backup verification before intentionally ending a
session. Upload failures stop the command and retain local files for retry.

To stop training at the next required boundary instead of finishing the run:

```python
!bash scripts/kaggle.sh train-stage full_ft
```

Use any of the five method names. This preserves the exact schedule; it only
divides execution across sessions. Quarter and third-quarter checkpoints are
recovery-only, without extra benchmark/qualitative conditions.
