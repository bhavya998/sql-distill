"""Build the train / test splits from templates and hand-written questions.

Splits:
- ``train``: template questions (later expanded with verified paraphrases).
- ``test_seen``: training templates, but parameter combinations whose SQL never appears in
  training — in-distribution.
- ``test_heldout``: templates that never appear in training — compositional generalisation.
- ``test_handwritten``: analyst questions written by hand, outside any template.
"""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from sql_distill.execute import run_sql
from sql_distill.handwritten import HANDWRITTEN
from sql_distill.templates import HELDOUT_TEMPLATES, TRAIN_TEMPLATES, Template


@dataclass
class Example:
    id: str
    question: str
    sql: str
    template: str
    skill: str
    split: str
    source: str = "template"


_TIEBREAK = re.compile(r",\s*((?:\w+\.)?id)\s+LIMIT", re.I)


def _usable(db: Path, sql: str) -> bool:
    """Returns rows, not all NULL, and no ties a model's ORDER BY could legitimately break
    differently (checked by reversing the gold query's id tie-breaker)."""
    res = run_sql(db, sql)
    if not (res.ok and res.rows and any(v is not None for row in res.rows for v in row)):
        return False
    m = _TIEBREAK.search(sql)
    if m:
        flipped = sql[:m.start()] + f", {m.group(1)} DESC LIMIT" + sql[m.end():]
        return run_sql(db, flipped).rows == res.rows
    return True


def _pool(t: Template, db: Path, rng: random.Random, attempts: int) -> dict[str, set[str]]:
    """Unique questions per unique SQL string for one template."""
    by_sql: dict[str, set[str]] = defaultdict(set)
    for _ in range(attempts):
        question, sql, _ = t.render(rng)
        by_sql[sql].add(question)
    return {sql: qs for sql, qs in by_sql.items() if _usable(db, sql)}


def build_splits(db: Path, seed: int = 7, attempts: int = 400, holdout_frac: float = 0.25,
                 heldout_per_template: int = 6) -> dict[str, list[Example]]:
    rng = random.Random(seed)
    splits: dict[str, list[Example]] = {"train": [], "test_seen": [], "test_heldout": [],
                                        "test_handwritten": []}
    for t in TRAIN_TEMPLATES:
        pool = _pool(t, db, rng, attempts)
        sqls = sorted(pool)
        rng.shuffle(sqls)
        n_test = int(len(sqls) * holdout_frac) if len(sqls) >= 4 else 0
        for i, sql in enumerate(sqls):
            split = "test_seen" if i < n_test else "train"
            questions = sorted(pool[sql])
            if split == "test_seen":
                questions = [rng.choice(questions)]
            for q in questions:
                splits[split].append(Example("", q, sql, t.id, t.skill, split))
    for t in HELDOUT_TEMPLATES:
        pool = _pool(t, db, rng, attempts)
        sqls = sorted(pool)
        rng.shuffle(sqls)
        for sql in sqls[:heldout_per_template]:
            q = rng.choice(sorted(pool[sql]))
            splits["test_heldout"].append(Example("", q, sql, t.id, t.skill, "test_heldout"))
    for i, (question, sql, skill) in enumerate(HANDWRITTEN):
        if not _usable(db, sql):
            raise ValueError(f"hand-written example {i} returns no rows: {sql}")
        splits["test_handwritten"].append(
            Example("", question, sql, f"hw{i:02d}", skill, "test_handwritten", "handwritten"))
    for name, rows in splits.items():
        rng.shuffle(rows)
        for i, ex in enumerate(rows):
            ex.id = f"{name}-{i:04d}"
    return splits


def write_jsonl(rows: list[Example] | list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(asdict(row) if isinstance(row, Example) else row,
                               ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
