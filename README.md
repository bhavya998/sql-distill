<div align="center">

# sql-distill

**Can a 1.7B model on a laptop GPU replace an API LLM as the text-to-SQL engine behind an analytics tool? Execution-verified distillation, LoRA SFT, then GRPO with an execution reward — measured on questions the training data never saw.**

[![CI](https://github.com/bhavya998/sql-distill/actions/workflows/ci.yml/badge.svg)](https://github.com/bhavya998/sql-distill/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![TRL](https://img.shields.io/badge/TRL-SFT%20%2B%20GRPO-ff9d00)](https://github.com/huggingface/trl)
[![License](https://img.shields.io/badge/License-MIT-22c55e)](LICENSE)

</div>

---

## Results

Execution accuracy: a prediction counts when it returns the gold rows. One RTX 3070 Ti laptop
GPU (8 GB). Latency is single-query, batch size 1.

> **Status:** baselines measured; the SFT and GRPO rows are added as each training run
> finishes. Nothing in this table is estimated.

<!-- RESULTS:START -->

| Model | Seen templates | Held-out templates | Hand-written | p50 latency | $ / 1k queries |
|---|---:|---:|---:|---:|---:|
| Qwen3-1.7B, no training | 51.3% | 29.2% | 47.5% | 2.01 s | local |
| Qwen3.5-4B Q4_K_M (llama.cpp), no training | 89.7% | 87.5% | 71.2% | 2.10 s | local |
| DeepSeek V4.1 Flash (teacher, API) | 100.0% | 91.7% | 94.9% | 0.90 s | $0.056 |

<!-- RESULTS:END -->

## The question

[analytics-mcp](https://github.com/bhavya998/analytics-mcp) exposes a sales database to LLM
clients. The natural next tool is "ask in English, get rows" — which today means sending the
schema and every question to an API model. This project asks what it takes to keep that on
the machine, and what it costs in accuracy.

## How it works

```
 templates (37 train, 8 held out)  ─┐
 hand-written questions (59)       ─┼─► splits: train · test_seen · test_heldout · test_handwritten
                                    │
 teacher LLM ─► paraphrases ─► keep if teacher SQL returns the template's gold rows
             └► new Q/SQL pairs ─► keep if a 2nd independent teacher query returns the same rows
                                    └► decontaminate against every test question
                                    │
 Qwen3-1.7B ─► LoRA SFT on verified data ─► GRPO, reward = execution match (1.0 / 0.1 / 0.0)
                                    │
 evaluate: execution accuracy on 3 test sets · single-query latency · $ per 1k queries
```

## Reproduce

```bash
uv sync --extra train                       # torch (CUDA 13.0 wheels), transformers, TRL, PEFT
uv run sql-distill build-data               # database + template and hand-written splits
uv run sql-distill distill --budget 0.40    # teacher data (needs DEEPSEEK_API_KEY)
uv run sql-distill sft                      # LoRA SFT  -> outputs/sft-1.7b
uv run sql-distill grpo                     # GRPO      -> outputs/grpo-1.7b
uv run sql-distill eval hf:Qwen/Qwen3-1.7B --adapter outputs/grpo-1.7b --name qwen3-1.7b-grpo
uv run sql-distill report                   # results table above
uv run pytest -q                            # offline tests: no GPU, no keys
```

The committed `data/` and `results/` directories hold the exact splits, the distilled training
set and every per-question prediction behind the table.

## Layout

```
src/sql_distill/
  db.py          schema + deterministic seed (same data as analytics-mcp)
  templates.py   question/SQL templates by skill; held-out templates
  handwritten.py 59 hand-written analyst questions with gold SQL
  dataset.py     splits, tie and emptiness checks
  teacher.py     paraphrases, invented pairs, self-consistency, decontamination
  execute.py     read-only execution, result matching (metric and reward)
  train.py       LoRA SFT, GRPO with the execution reward
  evaluate.py    endpoint and local-model backends, accuracy, latency
docs/adr/        design decisions
```

## License

MIT. Qwen3 and Qwen3.5 weights are Apache-2.0.
