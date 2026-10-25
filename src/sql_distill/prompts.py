"""The one prompt format every model sees — at training time, at evaluation, and in serving."""

from __future__ import annotations

from sql_distill.db import SCHEMA_PROMPT

SYSTEM = (
    "You translate questions about a SQLite sales database into a single SQLite query. "
    "Reply with the SQL only, no explanation."
)


def user_message(question: str) -> str:
    return f"Schema:\n{SCHEMA_PROMPT}\nQuestion: {question}"


def chat(question: str, answer: str | None = None) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_message(question)},
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer})
    return messages
