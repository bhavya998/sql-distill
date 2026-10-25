"""Execution-accuracy evaluation for API endpoints and local Hugging Face models."""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from sql_distill.execute import extract_sql, score
from sql_distill.prompts import chat


@dataclass
class Prediction:
    text: str
    seconds: float = 0.0
    output_tokens: int = 0


class EndpointBackend:
    """Any OpenAI-compatible endpoint (DeepSeek, llama.cpp)."""

    def __init__(self, name: str, concurrency: int = 4) -> None:
        from sql_distill.llm import ENDPOINTS, Usage

        self.name = name
        self.ep = ENDPOINTS[name]
        self.concurrency = concurrency
        self.usage = Usage()

    def _run(self, questions: list[str], concurrency: int):
        from sql_distill.llm import ChatClient

        async def go():
            client = ChatClient(self.ep, concurrency=concurrency)
            try:
                return await asyncio.gather(
                    *(client.complete(chat(q), max_tokens=400) for q in questions))
            finally:
                await client.aclose()
                for f in ("calls", "cached", "uncached", "output", "seconds"):
                    setattr(self.usage, f, getattr(self.usage, f) + getattr(client.usage, f))

        return asyncio.run(go())

    def predict(self, questions: list[str]) -> list[Prediction]:
        return [Prediction(r.text, r.seconds, r.output)
                for r in self._run(questions, self.concurrency)]

    def latency(self, questions: list[str]) -> list[float]:
        """Sequential single-query latency (concurrency 1)."""
        return [r.seconds for r in self._run(questions, 1)]

    def cost_per_1k(self) -> float:
        return 1000 * self.usage.cost(self.ep) / max(1, self.usage.calls)


class HFBackend:
    """A local causal LM, optionally with a LoRA adapter, decoded greedily."""

    def __init__(self, model_id: str, adapter: str | None = None, batch_size: int = 16,
                 max_new_tokens: int = 256) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = adapter or model_id
        self.tok = AutoTokenizer.from_pretrained(adapter or model_id, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16,
                                                     device_map="cuda")
        if adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
        self.model = model.eval()
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.torch = torch

    def _prompts(self, questions: list[str]) -> list[str]:
        return [self.tok.apply_chat_template(chat(q), tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False) for q in questions]

    def _generate(self, prompts: list[str]) -> tuple[list[str], list[int]]:
        enc = self.tok(prompts, return_tensors="pt", padding=True).to("cuda")
        with self.torch.inference_mode():
            out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id)
        new = out[:, enc["input_ids"].shape[1]:]
        texts = self.tok.batch_decode(new, skip_special_tokens=True)
        lengths = [int((row != self.tok.pad_token_id).sum()) for row in new]
        return texts, lengths

    def predict(self, questions: list[str]) -> list[Prediction]:
        preds: list[Prediction] = []
        prompts = self._prompts(questions)
        for i in range(0, len(prompts), self.batch_size):
            texts, lengths = self._generate(prompts[i:i + self.batch_size])
            preds += [Prediction(t, 0.0, n) for t, n in zip(texts, lengths, strict=True)]
        return preds

    def latency(self, questions: list[str]) -> list[float]:
        """Single-query wall-clock latency, batch size 1, after one warm-up call."""
        self._generate(self._prompts(questions[:1]))
        times = []
        for p in self._prompts(questions):
            start = time.perf_counter()
            self._generate([p])
            self.torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
        return times


def evaluate(backend, rows: list[dict], db: Path, out_path: Path) -> dict:
    preds = backend.predict([r["question"] for r in rows])
    by_skill: dict[str, list[bool]] = defaultdict(list)
    records = []
    for row, pred in zip(rows, preds, strict=True):
        sql = extract_sql(pred.text)
        ok, res = score(db, row["sql"], sql)
        by_skill[row["skill"]].append(ok)
        records.append({"id": row["id"], "question": row["question"], "gold": row["sql"],
                        "pred": sql, "correct": ok, "valid": res.ok, "error": res.error,
                        "seconds": round(pred.seconds, 3), "output_tokens": pred.output_tokens})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    n = len(records)
    return {
        "n": n,
        "accuracy": round(sum(r["correct"] for r in records) / n, 4),
        "valid_sql": round(sum(r["valid"] for r in records) / n, 4),
        "by_skill": {k: round(sum(v) / len(v), 3) for k, v in sorted(by_skill.items())},
    }


def latency_summary(seconds: list[float]) -> dict:
    s = sorted(seconds)
    return {"p50": round(statistics.median(s), 3),
            "p95": round(s[min(len(s) - 1, int(0.95 * len(s)))], 3), "n": len(s)}
