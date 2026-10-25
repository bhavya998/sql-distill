"""Render results/summary.json into the README results table."""

from __future__ import annotations

import json
import re
from pathlib import Path

ORDER = ["qwen3-1.7b-base", "qwen3-1.7b-sft-template", "qwen3-1.7b-sft", "qwen3-1.7b-grpo",
         "qwen3.5-4b-q4-zeroshot", "deepseek-v4.1-flash"]
LABELS = {
    "qwen3-1.7b-base": "Qwen3-1.7B, no training",
    "qwen3-1.7b-sft-template": "Qwen3-1.7B + SFT, template data only",
    "qwen3-1.7b-sft": "Qwen3-1.7B + SFT, distilled data",
    "qwen3-1.7b-grpo": "Qwen3-1.7B + SFT + GRPO",
    "qwen3.5-4b-q4-zeroshot": "Qwen3.5-4B Q4_K_M (llama.cpp), no training",
    "deepseek-v4.1-flash": "DeepSeek V4.1 Flash (teacher, API)",
}


def _pct(entry: dict, split: str) -> str:
    m = entry.get(split)
    return f"{100 * m['accuracy']:.1f}%" if m else "—"


def table(summary: dict) -> str:
    names = [n for n in ORDER if n in summary] + [n for n in summary if n not in ORDER]
    lines = [
        "| Model | Seen templates | Held-out templates | Hand-written | p50 latency | $ / 1k queries |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for n in names:
        e = summary[n]
        usd = e.get("usd_per_1k_queries", 0.0)
        cost = f"${usd:.3f}" if usd else "local"
        p50 = e.get("latency_s", {}).get("p50")
        lat = f"{p50:.2f} s" if p50 is not None else "—"
        lines.append(f"| {LABELS.get(n, n)} | {_pct(e, 'test_seen')} | "
                     f"{_pct(e, 'test_heldout')} | {_pct(e, 'test_handwritten')} | {lat} | {cost} |")
    return "\n".join(lines)


def write_readme(summary_path: Path = Path("results/summary.json"),
                 readme: Path = Path("README.md")) -> None:
    summary = json.loads(summary_path.read_text())
    block = "<!-- RESULTS:START -->\n\n" + table(summary) + "\n\n<!-- RESULTS:END -->"
    text = readme.read_text(encoding="utf-8")
    text = re.sub(r"<!-- RESULTS:START -->.*?<!-- RESULTS:END -->", block, text, flags=re.S)
    readme.write_text(text, encoding="utf-8")
