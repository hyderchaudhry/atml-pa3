# Task 1 implementation map

This maps the manual's requirements to code and durable artifacts. It is an implementation checklist, not a claim that experiments have been completed.

| Manual location | Implementation | Evidence generated when run |
| --- | --- | --- |
| p. 3 data selection, leakage, seeds | `data.select_training`, `common.io.seed_all` | preparation counts, 40,000 source IDs, asset revisions, hashes |
| p. 3 assistant-token objective | `data.encode_response`, `loss.response_loss`, effective-batch denominator | valid response-token count and SFT loss per update |
| p. 3 shared optimizer/precision recipe | `training.train`, `optimizer_for`, `model.training_model` | configuration, scaler/skipped updates, loss/LR, environment |
| pp. 3–4 recovery schedule | `training.save_checkpoint`, `_restore` | four model/adapter/optimizer/scheduler/scaler/RNG/data-position checkpoints |
| p. 4 Full FT displacement | `training.StartingPoint`, checkpoint measurements | midpoint/final Full FT, final L2-SP displacement |
| p. 4 L2-SP | fixed CPU reference, exact penalty gradient before unscaling | omega, lambda*omega, total loss, reference bytes and transfer/penalty time |
| p. 4 ranks/targets/initialization | `adapters.AdapterLinear`, `install_adapters` | adapter parameters, counts, configuration |
| p. 4 OPLoRA | exact shared FP32 SVD, projected effective factors | basis cache/provenance, costs, projected exports, constraint residuals |
| pp. 3–4 benchmarks | `harness.fixed_tasks`, `evaluation.evaluate` | seven complete four-benchmark evaluations, answers/likelihoods, extraction/cap flags |
| p. 4 merge verification | `model.verify_merge` | maximum/mean errors, tolerances, prompt IDs, pass/fail |
| p. 5 weight diagnostics | `diagnostics.diagnose` | rho and eta at layers 0/13/27 for six trained conditions |
| p. 5 activation diagnostics | `cache_probes`, `activation_ratios` | original FP32 last-token inputs, fixed IDs, all R(z), math/non-math summaries |
| p. 5 qualitative generation | `qualitative.qualitative` | 168 exact prompt/response/token records, grouped by question/format |
| p. 5 qualitative annotation | `group_and_prepare_annotations` | fixed 42-response human worksheet; flags/excerpts/notes filled by student |
| p. 6 required tables and figures | `summary.summarize` | benchmark/delta, resources, regularization, diagnostics CSVs; learning/score/penalty figures |
| p. 15 reproducibility/submission | environment snapshots, original source, README, `audit` | source/package/asset hashes, small Git-ready evidence; student writes report and submits |

Preparation freezes the probe set before results. Retention data is never used for training, replay, or tuning. Truncation/missing extraction cases stay in evaluation. Production commands expose no split limits or toy-data settings.

Kaggle execution uses `scripts/kaggle.sh` and `docs/kaggle.md`. Staged training keeps the same data order and stops only at the required recovery checkpoints. `storage.py` preserves complete checkpoints in a private Hub dataset, verifies sizes/hashes and a coherent evidence bundle before local offloading, and restores them by immutable commit. `loss.py` reduces vocabulary-logit memory through recomputation without changing supervised tokens or effective-batch normalization. The explicitly separate `memory-check` command uses a discarded synthetic model step; it is never included in training/evaluation results.
