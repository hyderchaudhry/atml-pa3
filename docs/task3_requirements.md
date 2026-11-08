# Task 3 manual-to-code map

Ground truth: [ATML-PA3.pdf](../ATML-PA3.pdf), pp. 11–14, and the manual-linked `ATML-PA3-Data/README.MD`. The implementation performs the prescribed experiments without changing data size, budgets, splits, sampling parameters or live batch size. There is no training in Task 3.

| Manual requirement | Implementation / evidence |
| --- | --- |
| Supplied 1,000 × 32 MATH and MathVista pools, exact three-field join, retained-token alignment | `data.py`: resumable Arrow-to-SQLite import, component uniqueness, model IDs, scalar/list length checks, probability/attention validation; `preparation.json` records source hashes and complete Arrow metadata |
| Lexicographic IDs, NumPy seed 6304, development 200 / held-out 800 | `data.prepare`; saved `splits.json`; first 50 held-out MATH IDs are saved for live generation |
| Ten random rollout orders, nested budgets 1, 2, 4, 8, 16, 32 shared across methods | `orders.json`; one independently reset seed-6304 RNG per dataset, iterating lexicographic question IDs and drawing ten permutations each |
| Supplied extractions; gold-free grouping; same equivalence for voting/grading | `scoring.py` and `selection.visible_groups`; `Pool` contains no gold/correctness fields; released metadata passed to normalization excludes its answer field |
| MATH `math-verify`; MathVista question types, choices, answer types and precision | Fixed scorer versions and source hash in `scorer.json`; MathVista metadata bytes/source commit preserved under `artifacts/task3/metadata` |
| One NO_ANSWER group, always incorrect, retained in costs/votes | `Scorer`, `visible_groups`, `vote`; no failed/capped outputs removed |
| Earliest supporting rollout breaks ties; average tied ranks; N=1 returns sole candidate | `select`, `vote`, `rank_weights` |
| Mean raw selected-token logp; mean full-distribution KL(U || p) | `quality`; supplied arrays used directly without temperature/filter transformations |
| Visual engagement normalized over three eligible attention totals | `validated_arrays`; no division by group size, no assumption that eligible masses sum to one |
| Current-prefix pooled KL linear Q05/Q95 calibration; clipping; VisConf V×C | `quality`; only current indices enter concatenation/percentiles; denominator floor 1e-8 |
| FirstAns, SC, highest logp, highest certainty, certainty+RWC | `methods` / `evaluate_question` for both datasets and all budgets |
| MathVista vision-only and VisConf, highest and RWC | Same pipeline; only the quality/aggregation component changes |
| Length-only shortest control; within-question length/score/correctness associations | `signal_diagnostics` at N=32; `tables/signal_associations.csv` and shortest rows in selection table |
| Top/bottom quartile correctness averaged across questions | Eight rollout scores per quartile, average within each question; paired question bootstrap on differences |
| Confidence–vision correlation; tokenwise fusion, both aggregation rules | Within-rollout calibrated confidence/vision correlation and population covariance; `token_fusion` and `token_fusion:rwc`; `token_confidence_vision.csv` |
| Fixed token trajectories and representative failures | First three held-out IDs / first two rollout IDs in saved order 0; additional first-two retrospective failure cases; stated rules in `illustrative_selection.json` and `qualitative.html` |
| Oracle opportunity; sampling versus selection failures | `oracle`, `sampling_failure`, `selection_failure` saved per method/order/budget; summed failures equal error |
| Retained token cost and FirstAns first-candidate-only cost | `tokens` and `consumed_ids`; `candidate_ids` describes the comparison pool, `consumed_ids` describes charged candidates. Oracle is comparison-pool coverage |
| Average ten orders within question, then average questions | `summary.offline_tables`; all main differences use 1,000 paired question bootstrap resamples |
| Original pinned Qwen2.5-3B-Instruct; FP16, no quantization, batch 1 | `run_live`; model revision parsed from supplied README and cross-checked against MATH Arrow metadata; `protocol.json` |
| Native chat, exact system/instruction, independent seeded calls | `generate_call`, `run_live`; original problem + two newlines + prescribed instruction. Seeds are SHA-256-derived from 6304, question ID and call kind; saved per response |
| Temperature .7 / top-p .8 / top-k 20 / repetition penalty 1 | Original `sample_token`: raw logits → temperature → top-k → nucleus → multinomial. Repetition factor 1 leaves logits unchanged |
| Long 3072, three parallel 1024, initial 1024 + revision 2048 | Five distinct calls per question: long, parallel_0/1/2, revision; 250 calls total. Parallel_0 reused as the refinement initial; revision prompt embeds that exact response and prescribed review instruction |
| Last literal boxed extraction with balanced braces, no numeric fallback | `last_boxed`; absent/empty/unclosed last box produces missing, nested braces preserved |
| Return revision unconditionally, no gold feedback | Generation happens before downstream grading. No scorer is used in constructing prompts or choosing initial versus revised output |
| Actual input/output cost, shared-initial cost charged twice across strategies | `strategies`, `live_per_question.csv`; all critique tokens and the longer revision prompt counted. Raw generated-token cost includes termination; retained count stored separately |
| Same GPU, runtime excluding load, peak memory, extraction/truncation, revision transitions | GPU UUID lock plus retained session environments; per-call synchronization/timing; live strategy tables and paired differences; wrong→correct / correct→wrong relative to shared initial |
| Stop at 4/8/16 else 32, SC answer for both controllers | `stopping_trace` / `stop`; current-prefix-only measurements and unique nonmissing SC leader |
| Quality rule: agreement + same SC/RWC winner + normalized vote margin | K for MATH, VisConf for MathVista; runner-up zero for a single answer group; RWC weights/ranks recalculated per prefix |
| Exact tau/gamma grids, always32 fallback, development accuracy within .01 | `policy_grid` / `choose_policy`; minimize mean token cost, then higher accuracy, smaller tau/gamma |
| Freeze settings before evaluation | Separate development/fit/evaluation commands; `fit_policies` reads development paths only, binds file hashes and refuses to replace a frozen policy under a changed scope |
| Held-out accuracy/cost/stopping distributions; incorrect early stops fixed32 would recover | `adaptive_evaluation.csv`, `adaptive_per_question.csv`, raw per-order records; correct full-pool fraction is attached after stopping for retrospective use only |
| Machine-readable evidence, environment, IDs and reproduction instructions | Per-question JSON.gz, raw live JSON, CSV, PNG, HTML, hashes, audit and evidence ZIP; [cloud guide](cloud_task3.md) |

## Fixed implementation conventions

The manual leaves the particular permutation seed derivation and some numerical reporting conventions unspecified. These are fixed in advance:

- Candidate permutations use NumPy `default_rng(6304)` per dataset in lexicographic question order, drawing ten permutations of `0..31`. Live calls use deterministic independent hash-derived seeds, distinct from the supplied export's generation seed 42.
- MATH extracted strings are parsed as whole boxed mathematical expressions with `math-verify==0.8.0`, no fallback. Strict mathematical verification is applied symmetrically for a consistent candidate equivalence relation. Candidate grouping compares only revealed answers to the earliest representatives; it never uses gold to merge groups. Parsing/verification timeout is five seconds; unparsed answers do not acquire a numeric fallback.
- MathVista option letters (including parenthesized letters) map to canonical uppercase letters. Other extracted choice text maps to the closest released choice by Levenshtein distance; option order breaks distance ties. Free-form integers use `int(float(value))`, floats use their released decimal precision and text stays text. Missing or invalid numeric extractions become NO_ANSWER. The metadata's gold answer is excluded from normalization inputs.
- Score-quartile ties are resolved in rollout-ID order: the eight lowest/highest of 32 scores form the bottom/top quartiles. Pearson correlations are computed within a question or within a rollout as specified; zero-variance correlations are null, never silently treated as zero. Mean token correlations first average defined rollout correlations within a question, then defined question means. Population covariance uses divisor Ti.
- Bootstrap intervals are percentile 95% intervals from 1,000 paired question resamples with seed 6304. Main paired differences include method versus FirstAns/SC at each N, adaptive versus fixed32, and all three pairwise live strategy comparisons.
- Offline oracle coverage refers to the currently available N-candidate comparison pool. FirstAns consumes/charges only its first candidate, despite comparison with that pool's oracle opportunity. Its consumed IDs are recorded explicitly.
- The supplied MATH metadata includes an original HF revision. MathVista's export labels its source revision `local-qwen2.5-vl-3b-instruct`; its absent HF/dataset commit metadata are preserved as absent. Task 3 does not replace that supplied pool or regenerate it with another VLM.
- Live strategy times sum their serial batch-one calls. Tokenization, prefill, sampling/decoding and partial-save overhead are timed; model loading, grading and HF network backups are excluded. Resumed call time includes already-recorded work plus cache replay. A destroyed machine may have unrecorded work since its last checkpoint; recorded runtime cannot reconstruct that lost interval.

## Originality, tests and limits

Selection, grouping, RWC ranks, confidence calibration/fusion, stopping, threshold fitting, bootstrapping, sampling, cache recovery, analysis/export and cloud orchestration were implemented for this repository. Existing repository I/O/hash helpers and its synthetic Hub test helper are reused from our own Tasks 1/2. The code does not download a public solution or external algorithm implementation. The externally provided components are the assignment data, official question metadata, pretrained Qwen architecture/weights and pinned libraries, including the explicitly required scorer.

`python -m atml.task3 smoke` runs 11 synthetic test groups using real Arrow/SQLite paths, tiny random Qwen2 weights and a mocked 250-call orchestration. It checks interruption/resume, current-prefix and gold isolation, tied ranks, scoring, thresholds, source alignment, exact prompts/caps, shared costs, physical-GPU consistency, checksums, failed uploads, restored state and ZIP integrity. Generated test files are removed by temporary-directory cleanup, including on failure. No synthetic result is presented as assignment evidence. A real GPU/full-data run remains necessary; the report and its interpretation remain the student's work.
