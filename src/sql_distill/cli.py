"""sql-distill command line: data, distillation, training, evaluation, report."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer

from sql_distill.dataset import build_splits, read_jsonl, write_jsonl
from sql_distill.db import DEFAULT_DB, build_database

app = typer.Typer(add_completion=False, no_args_is_help=True)
DATA = Path("data")
TEST_SPLITS = ["test_seen", "test_heldout", "test_handwritten"]


@app.command("build-data")
def build_data(overwrite: bool = False) -> None:
    """Create the sales database and the template / hand-written splits."""
    db = build_database(DEFAULT_DB, overwrite=overwrite)
    splits = build_splits(db)
    for name, rows in splits.items():
        write_jsonl(rows, DATA / f"{name}.jsonl")
        typer.echo(f"{name:18s} {len(rows):5d}")


@app.command()
def distill(rounds: int = 120, per_round: int = 12, budget: float = 0.45,
            endpoint: str = "deepseek-flash") -> None:
    """Expand the training set with execution-verified teacher data."""
    from sql_distill.llm import ENDPOINTS, ChatClient
    from sql_distill.teacher import Decontaminator, invent, paraphrase

    train = read_jsonl(DATA / "train.jsonl")
    tests = [r for s in TEST_SPLITS for r in read_jsonl(DATA / f"{s}.jsonl")]
    decon = Decontaminator(tests)

    async def run() -> tuple[list[dict], list[dict], dict]:
        client = ChatClient(ENDPOINTS[endpoint], concurrency=8)
        try:
            paras = await paraphrase(client, DEFAULT_DB, train, decon, budget_usd=budget / 3)
            invented = await invent(client, DEFAULT_DB, decon, rounds, per_round,
                                    budget_usd=budget)
        finally:
            await client.aclose()
        u = client.usage
        usage = {"calls": u.calls, "cached_tokens": u.cached, "uncached_tokens": u.uncached,
                 "output_tokens": u.output, "usd": round(u.cost(client.ep), 4)}
        return paras, invented, usage

    paras, invented, usage = asyncio.run(run())
    rows = train + paras + invented
    for i, r in enumerate(rows):
        r["id"] = f"train-{i:05d}"
    write_jsonl(rows, DATA / "train_distilled.jsonl")
    stats = {"template": len(train), "paraphrase": len(paras), "teacher": len(invented),
             "total": len(rows), "teacher_usage": usage}
    (DATA / "distill_stats.json").write_text(json.dumps(stats, indent=2))
    typer.echo(json.dumps(stats, indent=2))


RESULTS = Path("results")


def _update_summary(name: str, entry: dict) -> None:
    path = RESULTS / "summary.json"
    summary = json.loads(path.read_text()) if path.exists() else {}
    summary[name] = {**summary.get(name, {}), **entry}
    RESULTS.mkdir(exist_ok=True)
    path.write_text(json.dumps(summary, indent=2))


@app.command("eval")
def eval_cmd(
    backend: str = typer.Argument(..., help="endpoint:<name> or hf:<model id>"),
    name: str = typer.Option(..., help="Row name in the results table"),
    adapter: str = typer.Option("", help="LoRA adapter directory for hf backends"),
    splits: str = typer.Option(",".join(TEST_SPLITS)),
    latency_n: int = typer.Option(30, help="Sequential single-query latency samples"),
) -> None:
    """Score a model on the test splits by execution accuracy."""
    from sql_distill.evaluate import EndpointBackend, HFBackend, evaluate, latency_summary

    kind, _, target = backend.partition(":")
    model = (EndpointBackend(target) if kind == "endpoint"
             else HFBackend(target, adapter or None))
    entry: dict = {"backend": backend, "adapter": adapter}
    for split in splits.split(","):
        rows = read_jsonl(DATA / f"{split}.jsonl")
        metrics = evaluate(model, rows, DEFAULT_DB, RESULTS / name / f"{split}.jsonl")
        entry[split] = metrics
        typer.echo(f"{name:28s} {split:18s} acc={metrics['accuracy']:.3f} "
                   f"valid={metrics['valid_sql']:.3f} n={metrics['n']}")
    sample = [r["question"] for r in read_jsonl(DATA / "test_handwritten.jsonl")][:latency_n]
    entry["latency_s"] = latency_summary(model.latency(sample))
    entry["usd_per_1k_queries"] = (round(model.cost_per_1k(), 4) if kind == "endpoint"
                                   else 0.0)
    _update_summary(name, entry)
    typer.echo(json.dumps({k: v for k, v in entry.items() if k.startswith(("lat", "usd"))}))


@app.command("sft")
def sft_cmd(model_id: str = "Qwen/Qwen3-1.7B", out: Path = Path("outputs/sft-1.7b"),
            data: Path = DATA / "train_distilled.jsonl", epochs: float = 3.0) -> None:
    """LoRA supervised fine-tuning on the (distilled) training set."""
    from sql_distill.train import sft

    sft(read_jsonl(data), model_id, out, epochs=epochs)


@app.command("grpo")
def grpo_cmd(model_id: str = "Qwen/Qwen3-1.7B", sft_adapter: Path = Path("outputs/sft-1.7b"),
             out: Path = Path("outputs/grpo-1.7b"), data: Path = DATA / "train_distilled.jsonl",
             steps: int = 150) -> None:
    """GRPO from the SFT adapter with the execution reward."""
    from sql_distill.train import grpo

    grpo(read_jsonl(data), model_id, sft_adapter, out, steps=steps)


@app.command()
def report() -> None:
    """Write the results table from results/summary.json into README.md."""
    from sql_distill.report import write_readme

    write_readme()
    typer.echo("README results table updated")


if __name__ == "__main__":
    app()
