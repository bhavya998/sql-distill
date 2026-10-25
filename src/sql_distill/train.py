"""LoRA SFT on the distilled data, then GRPO with an execution reward.

Prompts are rendered to text with the same chat template the evaluator uses
(``enable_thinking=False``), so training and inference see identical tokens.
"""

from __future__ import annotations

from pathlib import Path

from sql_distill.db import DEFAULT_DB
from sql_distill.execute import extract_sql, is_ordered, results_match, run_sql
from sql_distill.prompts import chat

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _prompt(tok, question: str) -> str:
    return tok.apply_chat_template(chat(question), tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)


def sft(rows: list[dict], model_id: str, out_dir: Path, epochs: float = 3.0, lr: float = 2e-4,
        rank: int = 16, batch: int = 4, accum: int = 4, seed: int = 0) -> Path:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tok = AutoTokenizer.from_pretrained(model_id)
    end = "<|im_end|>"
    data = Dataset.from_list([{"prompt": _prompt(tok, r["question"]),
                               "completion": r["sql"].strip() + end} for r in rows])
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
    config = SFTConfig(
        output_dir=str(out_dir), num_train_epochs=epochs, learning_rate=lr,
        per_device_train_batch_size=batch, gradient_accumulation_steps=accum,
        lr_scheduler_type="cosine", warmup_steps=0.05, logging_steps=10, save_strategy="no",
        bf16=True, gradient_checkpointing=True, max_length=1280, seed=seed,
        completion_only_loss=True, report_to="none",
    )
    lora = LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.05,
                      target_modules=LORA_TARGETS, task_type="CAUSAL_LM")
    trainer = SFTTrainer(model=model, args=config, train_dataset=data, processing_class=tok,
                         peft_config=lora)
    trainer.train()
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    return out_dir


def execution_reward(completions: list[str], sql: list[str], **_: object) -> list[float]:
    """1.0 for the gold rows, 0.1 for SQL that runs but is wrong, 0.0 for invalid SQL."""
    rewards = []
    for completion, gold_sql in zip(completions, sql, strict=True):
        text = completion if isinstance(completion, str) else completion[0]["content"]
        pred = run_sql(DEFAULT_DB, extract_sql(text))
        if not pred.ok:
            rewards.append(0.0)
            continue
        gold = run_sql(DEFAULT_DB, gold_sql)
        rewards.append(1.0 if results_match(gold, pred, is_ordered(gold_sql)) else 0.1)
    return rewards


def grpo(rows: list[dict], model_id: str, sft_adapter: Path, out_dir: Path, steps: int = 150,
         lr: float = 5e-6, generations: int = 8, accum: int = 4, seed: int = 0) -> Path:
    import torch
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    tok = AutoTokenizer.from_pretrained(str(sft_adapter))
    base = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, str(sft_adapter), is_trainable=True)
    data = Dataset.from_list([{"prompt": _prompt(tok, r["question"]), "sql": r["sql"]}
                              for r in rows]).shuffle(seed=seed)
    config = GRPOConfig(
        output_dir=str(out_dir), max_steps=steps, learning_rate=lr,
        per_device_train_batch_size=generations, num_generations=generations,
        gradient_accumulation_steps=accum, max_completion_length=200, temperature=1.0,
        beta=0.0, logging_steps=5, save_strategy="no", bf16=True,
        gradient_checkpointing=True, seed=seed, report_to="none",
    )
    trainer = GRPOTrainer(model=model, reward_funcs=[execution_reward], args=config,
                          train_dataset=data, processing_class=tok)
    trainer.train()
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    return out_dir
