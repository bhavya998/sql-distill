# ADR 001: Design decisions

## Status
Accepted

## 1. Student model: Qwen3-1.7B, not a Qwen3.5 small model

Qwen3.5-0.8B/2B are unified vision-language models (`Qwen3_5ForConditionalGeneration`) with
linear-attention layers whose fast kernels are not readily available on Windows. Qwen3-1.7B
is a plain `Qwen3ForCausalLM`: first-class support in transformers, PEFT and TRL, and it fits
bf16 LoRA training on an 8 GB laptop GPU. Qwen3.5-4B is still in the comparison, served
through llama.cpp, where its kernels exist.

## 2. Metric: execution accuracy with column-relaxed matching

String match on SQL rejects correct queries; execution match is what an analyst cares about.
Rows must match exactly (floats to 2 decimals). Column order and extra columns are ignored,
because "which customers…" answered with `id, name` is correct. Row order only counts when the
gold query sorts at the top level. Gold queries with ties at a `LIMIT` boundary are removed at
build time, since any tie-breaking would be arbitrary.

## 3. Three test sets, none from the training generator alone

- `test_seen` — training templates, parameter combinations whose SQL never appears in training.
- `test_heldout` — templates that never appear in training.
- `test_handwritten` — 59 questions written by hand, independent of the templates.

Only the last measures what a user would experience; the first two separate memorisation from
generalisation.

## 4. Teacher data is kept only when execution verifies it

Paraphrases must return the template's gold rows. Invented pairs must be reproduced by a
second, independent teacher query (self-consistency by execution). Every candidate is
decontaminated against all test questions (identical SQL, or question token-Jaccard ≥ 0.6).
About 40% of invented pairs are dropped; that is the point.

## 5. SFT, then GRPO with an execution reward

SFT on verified data teaches format and schema. GRPO then optimises the metric itself: reward
1.0 for the gold rows, 0.1 for SQL that runs but is wrong, 0.0 for SQL that fails. `beta = 0`
(no reference model) keeps the run inside 8 GB. Generation uses Hugging Face `generate`;
vLLM is not available on Windows.

## 6. Why a local model when the API costs $0.06 per 1k queries

The teacher is cheap; cost is not the argument. The arguments are data residency (the
questions and the schema never leave the machine), offline operation, predictable latency and
no dependency on a vendor's model changes. The experiment measures what that costs in accuracy.
