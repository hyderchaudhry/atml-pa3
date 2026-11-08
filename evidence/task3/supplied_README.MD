# PA3 Task 3: Supplied Rollout Data

Pre-generated responses and token-level measurements for offline test-time compute experiments. Reading these files requires neither model weights nor images; images are not included.

| Folder | Model | Dataset | Questions | Rollouts/question |
| --- | --- | --- | ---: | ---: |
| `qwen25-math/` | Qwen2.5-3B-Instruct | MATH test subset | 1,000 | 32 |
| `qwen25vl-mathvista/` | Qwen2.5-VL-3B-Instruct | MathVista testmini | 1,000 | 32 |

## Files and alignment

Each folder contains `generations.parquet` and `probability_metrics.parquet`. The VLM folder additionally contains `attention_metrics.parquet`. Each file has 32,000 rows, one per rollout.

Join files on **`(dataset, sample_id, rollout_id)`**. Question IDs are strings; rollout IDs run from 0 to 31 within each question. MATH IDs are `test:<original zero-based row index>`; MathVista uses original question IDs.

All per-token lists align with `generated_token_ids`. Entry `k` describes the prediction of token `k`, conditioned on the prompt and the preceding `k` generated tokens. Selected stop tokens are excluded from every list.

## Generations

`generations.parquet` contains the join keys and:

| Field | Meaning |
| --- | --- |
| `model`, `question_text` | Model identifier and original question; MathVista includes choices where applicable. |
| `generated_text`, `generated_token_ids` | Stored response and retained token IDs. |
| `n_generated_text_tokens` | Scalar retained response length. |
| `hit_max_new_tokens` | Whether generation reached the token cap. |
| `gold_answer`, `extracted_answer` | Reference answer and nullable extracted prediction. |

Extracted answers are **not correctness labels**. Apply the assignment's answer normalization and equivalence rules. Retain failed extractions and capped responses.

For MATH, extraction finds the last literal `\boxed{` and scans balanced braces, preserving nested LaTeX and stripping surrounding whitespace. An absent, empty, or unclosed last box gives null; there is no numeric fallback. MathVista predictions use the export's stored `dataset_default` extractor (version 1).

## Probability measurements

`probability_metrics.parquet` contains the join keys and two float32 lists:

- `logp`: natural log-probability of the selected token.
- `kl_u_p`: `D_KL(U || p) = -log(V) - mean_j(log(p[j]))`, where `U` is uniform over the full vocabulary of size `V`.

Both use the raw full-vocabulary model distribution **before** temperature or sampling filters. Higher `logp` means greater selected-token confidence; higher `kl_u_p` means greater divergence from uniform.

## Visual attention (VLM only)

`attention_metrics.parquet` contains the join keys, scalar eligible-group counts `n_image_tokens` and `n_prompt_text_tokens`, and these per-token lists:

- `img_attn_total`: attention mass on eligible image positions.
- `prompt_text_attn_total`: attention mass on eligible prompt text.
- `generated_text_attn_total`: attention mass on preceding generated tokens.
- `n_generated_text_tokens`: preceding-token counts `[0, ..., N-1]`, **not** the scalar length in the generations file.

Attention is averaged over all heads and all 36 decoder layers at the position predicting each token. Excluded prompt special/boundary tokens can receive attention, so the three masses need not sum to one. Totals are not divided by group sizes. The first prediction has zero generated-text attention.

## Generation and provenance

Both exports use temperature 0.7, top-p 0.8, top-k 20, repetition penalty 1.0, base seed 42, and a 2,048-new-token cap. This generation seed is separate from the assignment's analysis seed.

MATH uses the native chat template, system message `You are a helpful assistant.`, and the problem followed by two newlines and:

```text
Think step by step inside <think> </think> tags, then give your final answer inside \boxed{}.
```

Pinned MATH identities:

- Model: `Qwen/Qwen2.5-3B-Instruct`, revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`.
- Dataset: `DigitalLearningGmbH/MATH-lighteval`, revision `0530c78699ea5e8eb5530600900e1f328b48acad`; 1,000 of 5,000 test rows sampled without replacement using NumPy `default_rng(42)`, then retained in source order.

| Export | Retained tokens | Capped rollouts | Null extracted answers |
| --- | ---: | ---: | ---: |
| MATH | 20,742,094 | 523 | 521 |
| MathVista | 8,961,703 | 487 | 3,032 |
