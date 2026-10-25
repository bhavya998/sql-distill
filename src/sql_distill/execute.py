"""Read-only SQL execution and result comparison — the metric and the RL reward.

A prediction is correct when it returns the same rows as the gold query. Column order and
extra columns are ignored (``SELECT id, name`` answers "which customers"); row order matters
only when the gold query sorts at the top level, i.e. the question asks for a ranking.
"""

from __future__ import annotations

import re
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass
from itertools import combinations, permutations
from pathlib import Path

MAX_ROWS = 5000
_FENCE = re.compile(r"```(?:sql|sqlite)?\s*(.*?)```", re.S | re.I)
_START = re.compile(r"\b(WITH|SELECT)\b", re.I)


@dataclass(frozen=True)
class ExecResult:
    ok: bool
    rows: tuple[tuple, ...] = ()
    error: str = ""
    seconds: float = 0.0

    @property
    def width(self) -> int:
        return len(self.rows[0]) if self.rows else 0


def extract_sql(text: str) -> str:
    """Pull one SQL statement out of a model reply (fenced block, or from SELECT/WITH on)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    fenced = _FENCE.findall(text)
    candidate = fenced[-1] if fenced else text
    match = _START.search(candidate)
    if not match:
        return candidate.strip()
    sql = candidate[match.start():].strip()
    return sql.split(";")[0].strip()


def run_sql(db: Path | str, sql: str, timeout_s: float = 3.0) -> ExecResult:
    """Execute one SELECT/WITH statement on a read-only connection with a time limit."""
    sql = sql.strip().rstrip(";").strip()
    if not _START.match(sql):
        return ExecResult(False, error="not a SELECT/WITH statement")
    if ";" in sql:
        return ExecResult(False, error="multiple statements")
    conn = sqlite3.connect(f"file:{Path(db).as_posix()}?mode=ro", uri=True)
    deadline = time.perf_counter() + timeout_s
    conn.set_progress_handler(lambda: int(time.perf_counter() > deadline), 10_000)
    start = time.perf_counter()
    try:
        rows = conn.execute(sql).fetchmany(MAX_ROWS + 1)
    except sqlite3.Error as exc:
        return ExecResult(False, error=str(exc), seconds=time.perf_counter() - start)
    finally:
        conn.close()
    if len(rows) > MAX_ROWS:
        return ExecResult(False, error=f"more than {MAX_ROWS} rows")
    return ExecResult(True, tuple(tuple(r) for r in rows), seconds=time.perf_counter() - start)


def is_ordered(sql: str) -> bool:
    """True when the outermost query has ORDER BY (nested ones do not count)."""
    depth, upper = 0, sql.upper()
    for i, ch in enumerate(upper):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif depth == 0 and upper.startswith("ORDER BY", i):
            return True
    return False


def _norm(value: object) -> object:
    if isinstance(value, float):
        return round(value, 2) if abs(value - round(value)) > 1e-9 else int(round(value))
    if isinstance(value, str):
        return value.strip()
    return value


def _project(rows: tuple[tuple, ...], cols: tuple[int, ...]) -> list[tuple]:
    return [tuple(_norm(r[c]) for c in cols) for r in rows]


def results_match(gold: ExecResult, pred: ExecResult, ordered: bool) -> bool:
    if not (gold.ok and pred.ok) or len(gold.rows) != len(pred.rows):
        return False
    if not gold.rows:
        return True
    g = _project(gold.rows, tuple(range(gold.width)))
    if pred.width < gold.width or pred.width > 8:
        return False
    for subset in combinations(range(pred.width), gold.width):
        for cols in permutations(subset):
            p = _project(pred.rows, cols)
            if (p == g) if ordered else (Counter(p) == Counter(g)):
                return True
    return False


def score(db: Path | str, gold_sql: str, pred_sql: str) -> tuple[bool, ExecResult]:
    gold = run_sql(db, gold_sql)
    pred = run_sql(db, pred_sql)
    return results_match(gold, pred, is_ordered(gold_sql)), pred
