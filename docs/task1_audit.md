# Task 1 implementation audit

Reviewed against **ATML-PA3.pdf, pages 3–6**, including the common recipe,
five methods, evaluation/checkpoint schedule, diagnostics and required evidence.
The mapping to source files and generated artifacts is in
[task1_requirements.md](task1_requirements.md). All implementation changes are
original repository code. The required libraries and benchmark task definitions
are used through their APIs; no public assignment solution was copied.

This is an implementation audit, not a report or experimental analysis.
No assignment training or benchmark evaluation was executed during this review.

Subsequent Kaggle experiments are now imported under `evidence/task1`. The saved
`completeness.json` passes all computational checks; only the 42 human annotations
remain pending. The implementation review below predates those experiments.

## Issues found and addressed

| Finding | Resolution |
| --- | --- |
| GSM8K and qualitative generation were serial, with a hard batch-one assertion | Batch 32 by default; unpad prompts using attention masks, record each row independently, exclude post-EOS padding, distinguish stock stops from cap truncation |
| Stock stop matchers track rows internally but their aggregate result can keep already-finished rows decoding | Reuse those exact matchers and return the union of their per-row completion flags to Transformers; stock text filtering and answer extraction remain in the harness |
| Likelihood batch 32 could materialize vocabulary logits/softmax for every prompt token | Original continuation-only head backend: encode through stock HFLM, batch/share identical causal prefixes, project only scored hidden states in chunks, retain stock softmax dtype and continuation sum/greedy rule |
| Model construction forced eager attention | SDPA for new runs/inference, with eager selectable; resumed training retains its original backend |
| Every training microbatch synchronized a scalar loss with the CPU | Accumulate detached FP32 loss sums on device and read once per effective batch; unchanged response-token denominator |
| AMP-skipped optimizer steps advanced the epoch and could leave fewer than 1,250 applied updates | Retry the same effective batch with the reduced dynamic scale and restored dropout RNG; advance data/LR/update only after success, retain separate skip logs and include retry time in resource totals |
| Interrupted evaluation discarded completed model work and reran the full split | Append-only request journal scoped to task definition, exact checkpoint hash, asset revisions and protocol; replay completed requests, score only missing ones, publish full-split results only when complete |
| Inference evidence reached remote storage only after an entire benchmark | Commit each completed batch locally, periodically upload a verified coherent evidence bundle and back up at stage completion |
| A torn final JSONL write prevented recovery | Isolate an incomplete final write; reject corruption in committed lines; replace pruned curves and final sample files atomically |
| Probe caching did 64 individual causal-LM forwards, including unused vocabulary logits | Batched FP32 backbone-only forwards, explicit left-padding position IDs, hooks at each row's last non-padding prompt token |
| SVD preprocessing restarted all modules after interruption | Atomically save/hash each exact FP32 module SVD and reuse completed modules; persist model/rank scope and per-module timing; periodically back up partial shared artifacts |
| Interrupted preparation could block re-entry or resolve new upstream revisions | Reuse the first asset lock, stage dataset writes atomically, verify/reuse completed tokenization and rebuild the unfinished manifest |
| An incomplete checkpoint directory blocked retry | Retry only the incomplete serialization, protecting complete checkpoints; reconcile metadata/resources if interrupted after the atomic rename |
| Resume pruned training curves by clearing their file first | Archive crash-tail evidence and atomically replace the active curve; also prune displacement measurements after the restored update |
| Interruption before the first checkpoint made automatic restart fail | Preserve that attempt's curve and restart from the fixed original seed with `--resume-latest`; no unavailable optimizer state is claimed to be recovered |
| Presence of a failed merge-check file could bypass verification on retry | Require a passing verification; rerun failed/missing checks before adapter evaluation |
| Restored checkpoint files could be reused locally without rechecking their receipt | Validate local file hashes against verified archive receipts before reuse |

## Manual coverage

| Required item | Implementation status; execution still required |
| --- | --- |
| Original post-trained Qwen3-0.6B; non-thinking; seed 6304 | Locked revision; independent starts; template arguments and seed/environment logs |
| First 40,000 eligible MetaMathQA, source order, ≤2,048 tokens, no truncation/replay | Empty/overlength and normalized query/original-question overlap filtering; retained source IDs and dataset hashes |
| Assistant response + EOT loss; prompt/pad masking | Offset-based labels; summed CE divided by all valid response tokens across the effective batch |
| Shared optimizer/precision/schedule | Microbatch 2, accumulation 16, one epoch/1,250 applied updates, AdamW groups/betas/epsilon, 38 warmup + cosine, FP32 masters/adapters/moments, FP16 AMP/scaler, unscale then global clipping, checkpointing; AMP-skipped attempts logged separately |
| Five independent variants | Full FT 2e-5; L2-SP 2e-5/lambda 1e-3; LoRA-4 alpha 8; LoRA-16/OPLoRA-16 alpha 32 at 2e-4; dropout .05, all seven targets per layer |
| Exact L2-SP | Fixed original CPU FP32 reference; half-sum over unique parameters, including biases/norms and tied parameters once; exact scaled gradient; Omega/objective/storage/transfer logs |
| OPLoRA | Shared rank-128 original FP32 SVD bases, matched unprojected LoRA-16 initialization, effective factor projections, projected exports and cost logs |
| Checkpoints | Each run at updates 312/625/937/1250, complete model/adapter + optimizer/scheduler/scaler/RNG/data position; verified private backup and lazy restore |
| Seven conditions | Shared baseline, Full FT midpoint, all five finals; quarter/third-quarter checkpoints used only for recovery |
| Four complete zero-shot benchmarks per condition | Stock locked tasks and exact split counts; GSM8K non-thinking greedy cap 1,024 with stock filters/stops, flexible EM and unextractable flags; stock completion likelihood prompts for retention, raw and normalized metrics |
| Merged adapter inference | Fresh original backbone plus exported factors, including projected OPLoRA; passing small-batch logit check required |
| Per-example evidence | Request/token journals, original documents/targets, outputs, filtered scores, extracted answers, actual token-cap flags; no results published for partial splits |
| Resource evidence | Per-update loss/LR/timing/scaler/skip logs, peak GPU memory, trainable counts, checkpoint/export/reference/basis costs; setup/segment provenance saved |
| Full-model displacement | Full FT midpoint/final and L2-SP final |
| FP32 weight diagnostics | Layers 0/13/27, six trained conditions; rho/eta, explicit undefined rho for a zero update, OPLoRA constraint residuals |
| FP32 activation diagnostics | Seeded 32 GSM8K + 32 HellaSwag IDs fixed before scores; cached original prompt inputs shared across all local updates; every R(z), math/non-math summaries |
| Qualitative generation | All 24 supplied prompts × seven conditions; JSON system instructions preserved, greedy non-thinking cap 512; grouped 168 responses |
| Fixed human subset | 42-row worksheet, first/last plain-order questions, all formats/conditions; truncation flags retained; judgments and excerpts left to the student |
| Required visual/table evidence | Seven-condition benchmark/delta table, five-run combined SFT curve/resources, regularization/displacement, diagnostics, grouped qualitative outputs |
| Reproducibility | Pinned primary dependencies, exact assets/source IDs/hashes, saved configurations, package/Git/source snapshots, scheduler/RNG recovery and cache scopes |

`successful_updates` and skipped AMP attempts are recorded separately. An overflow
does not apply AdamW or consume the next examples: the same batch is recomputed
with the reduced scale and restored dropout RNG. Each example participates in one
applied update, and checkpoint/data/LR boundaries count successful updates. A legacy
checkpoint that already advanced beyond skipped batches cannot be repaired by
resuming it; the affected variant must be restarted. The audit rejects such a
run's applied-update count rather than claiming the required budget was met.

## Verification and remaining execution

`atml-task1 smoke` exercises synthetic data selection/masks, dense-vs-chunked CE
and every parameter gradient, adapter initialization/count/projection/merge,
unique-parameter L2-SP and optimizer groups, analytic diagnostics, checkpoint
optimizer/scaler/RNG recovery, torn journal recovery, mixed EOS/text-stop/cap
batches, direct comparison with the unmodified pinned HFLM scorer/generator,
FP16 likelihoods, cache-only restart, batched FP32 probes, interrupted SVD reuse,
and qualitative generation/resume/system instructions/grouping/worksheet creation.
An explicit CPU GradScaler overflow check verifies that the retry preserves RNG
and applies exactly one successful update while retaining the reduced scale.
Temporary artifacts are removed; the saved verification record is
`evidence/task1/system/smoke.json`. These checks do not establish benchmark
performance or actual full-model GPU memory use.

The local machine has no CUDA. The subsequent Kaggle evidence includes all five
full-length model memory checks, verified receipts for all 20 required recovery
checkpoints, all 28 complete benchmark splits, six trained-condition diagnostics,
168 real responses, and measured training resources. Local import checks confirmed
the benchmark document IDs, qualitative counts, and all 1,250 ordered applied
updates per method. Comparative GPU speedups were not measured. The 42 human
annotations and the user's report remain pending; no qualitative judgments or
report text are supplied here. The reproducible execution instructions remain in
[task1_execution.md](task1_execution.md).

Resumption is bounded by successful persistence: an in-flight inference batch,
inference work since the last remote sync after session loss, and training work
after the last saved recovery checkpoint can repeat. A retained local working
directory preserves committed batches. Verified backup errors stop execution and
retain local data. Complete output files are reused; corrupted or mismatched
scopes fail explicitly. Tasks 2 and 3 remain unimplemented; common I/O and Task 1
modules remain separated for later additions.
