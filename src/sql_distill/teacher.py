"""Distillation data from a teacher LLM, kept only when execution verifies it.

Two sources:
- **paraphrases** of template questions: kept when the teacher's SQL for the reworded question
  returns the same rows as the template's gold SQL;
- **new pairs** the teacher invents across skills and personas: kept when a second, independent
  teacher query for the same question returns the same rows (self-consistency by execution).

Everything is decontaminated against all test splits before it reaches training.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
from pathlib import Path

from sql_distill.db import SCHEMA_PROMPT
from sql_distill.execute import extract_sql, is_ordered, results_match, run_sql
from sql_distill.llm import ChatClient
from sql_distill.prompts import chat

FOCUS = [
    "simple filters and counts on a single table",
    "aggregates (SUM, AVG, MIN, MAX) with filters",
    "GROUP BY breakdowns, some sorted",
    "joins between orders and customers",
    "joins through order_items to products",
    "rankings with ORDER BY and LIMIT where the answer is unambiguous",
    "HAVING conditions on groups",
    "date logic: months, quarters, ranges, weekdays, day differences with julianday",
    "subqueries: comparisons with averages, NOT IN, EXISTS",
    "percentages and ratios with CASE WHEN",
    "employees: roles, regions, hire dates and the orders they handled",
    "products: inventory, prices, costs and margins",
    "customers: tiers, signup cohorts, companies and lifetime value",
    "order statuses: cancellations, refunds, pending orders over time",
]
PERSONAS = ["a CFO", "a sales manager", "an operations lead", "a product manager",
            "a customer success manager", "a data analyst", "a regional director"]

GEN_PROMPT = """Database schema (SQLite):
{schema}
Write {n} different questions that {persona} might ask about this data, focusing on: {focus}.
Rules:
- each question is answerable with ONE SQLite query on this schema and has one clear answer
  (no ties for "the top/most"; say "top 3" etc. when listing);
- use concrete values from the schema notes (regions, tiers, categories, statuses, 2024/2025);
- vary the wording: some casual, some formal, some terse;
- the SQL must be correct SQLite.
Return JSON: {{"items": [{{"question": "...", "sql": "..."}}]}}"""

PARA_PROMPT = """Database schema (SQLite):
{schema}
Rewrite this question in {k} different ways a real person might ask it (one casual, one more
formal), keeping exactly the same meaning and values, then write the SQLite query for each.
Question: {question}
Return JSON: {{"items": [{{"question": "...", "sql": "..."}}]}}"""


def _norm_q(q: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", q.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / max(1, len(a | b))


class Decontaminator:
    """Rejects training candidates that are near-copies of any test question or gold SQL."""

    def __init__(self, test_rows: list[dict], threshold: float = 0.6) -> None:
        self.sqls = {re.sub(r"\s+", " ", r["sql"].strip().lower()) for r in test_rows}
        self.questions = [_norm_q(r["question"]) for r in test_rows]
        self.threshold = threshold

    def clean(self, question: str, sql: str) -> bool:
        if re.sub(r"\s+", " ", sql.strip().lower()) in self.sqls:
            return False
        q = _norm_q(question)
        return all(_jaccard(q, t) < self.threshold for t in self.questions)


def _items(text: str) -> list[dict]:
    try:
        data = json.loads(text[text.find("{"): text.rfind("}") + 1])
    except json.JSONDecodeError:
        return []
    return [i for i in data.get("items", []) if isinstance(i, dict)
            and isinstance(i.get("question"), str) and isinstance(i.get("sql"), str)]


def _good(db: Path, sql: str):
    res = run_sql(db, sql)
    if not res.ok or not res.rows or len(res.rows) > 200:
        return None
    if all(v is None for row in res.rows for v in row):
        return None
    return res


async def paraphrase(client: ChatClient, db: Path, rows: list[dict], decon: Decontaminator,
                     k: int = 2, budget_usd: float = 0.15) -> list[dict]:
    out: list[dict] = []

    async def one(row: dict) -> None:
        if client.usage.cost(client.ep) > budget_usd:
            return
        prompt = PARA_PROMPT.format(schema=SCHEMA_PROMPT, k=k, question=row["question"])
        reply = await client.complete([{"role": "user", "content": prompt}], temperature=0.7,
                                      max_tokens=600, response_format={"type": "json_object"})
        gold = run_sql(db, row["sql"])
        for item in _items(reply.text):
            pred = run_sql(db, extract_sql(item["sql"]))
            if (results_match(gold, pred, is_ordered(row["sql"]))
                    and decon.clean(item["question"], row["sql"])):
                out.append({**row, "id": "", "question": item["question"].strip(),
                            "source": "paraphrase"})

    await asyncio.gather(*(one(r) for r in rows))
    return out


async def invent(client: ChatClient, db: Path, decon: Decontaminator, rounds: int,
                 per_round: int = 12, seed: int = 11, budget_usd: float = 0.3) -> list[dict]:
    rng = random.Random(seed)
    jobs = [(rng.choice(PERSONAS), FOCUS[i % len(FOCUS)]) for i in range(rounds)]
    candidates: list[dict] = []
    out: list[dict] = []

    async def generate(persona: str, focus: str) -> None:
        if client.usage.cost(client.ep) > budget_usd:
            return
        prompt = GEN_PROMPT.format(schema=SCHEMA_PROMPT, n=per_round, persona=persona,
                                   focus=focus)
        reply = await client.complete([{"role": "user", "content": prompt}], temperature=0.9,
                                      max_tokens=3000, response_format={"type": "json_object"})
        for item in _items(reply.text):
            candidates.append({"question": item["question"].strip(),
                               "sql": extract_sql(item["sql"]), "focus": focus})

    async def verify(c: dict) -> None:
        if client.usage.cost(client.ep) > budget_usd:
            return
        gold = _good(db, c["sql"])
        if gold is None or not decon.clean(c["question"], c["sql"]):
            return
        reply = await client.complete(chat(c["question"]), temperature=0.0, max_tokens=400)
        second = run_sql(db, extract_sql(reply.text))
        if results_match(gold, second, is_ordered(c["sql"])):
            out.append({"id": "", "question": c["question"], "sql": c["sql"],
                        "template": "teacher", "skill": c["focus"], "split": "train",
                        "source": "teacher"})

    await asyncio.gather(*(generate(p, f) for p, f in jobs))
    seen: set[str] = set()
    unique = []
    for c in candidates:
        key = " ".join(sorted(_norm_q(c["question"])))
        if key not in seen:
            seen.add(key)
            unique.append(c)
    await asyncio.gather(*(verify(c) for c in unique))
    stats = {"generated": len(candidates), "unique": len(unique), "kept": len(out)}
    print(f"teacher invent: {stats}")
    return out
