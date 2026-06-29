"""Run the frozen Phase-2e final evaluator on the isolated kv2 test cache."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from jepa4d.evaluation.phase2e_final import run_final_evaluation

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def main(
    train_validation_cache: Annotated[
        Path,
        typer.Option("--train-validation-cache", exists=True, dir_okay=False),
    ],
    test_cache: Annotated[Path, typer.Option("--test-cache", exists=True, dir_okay=False)],
    feature_cache_receipt: Annotated[
        Path,
        typer.Option("--feature-cache-receipt", exists=True, dir_okay=False),
    ],
    shard_dirs: Annotated[
        list[Path],
        typer.Option("--shard-dir", exists=True, file_okay=False),
    ],
    output: Annotated[Path, typer.Option("--output")],
    test_receipt: Annotated[Path, typer.Option("--test-receipt", exists=True, dir_okay=False)],
    device: Annotated[str, typer.Option("--device")] = "cuda:0",
    batch_size: Annotated[int, typer.Option("--batch-size", min=1)] = 8,
    latency_warmup: Annotated[int, typer.Option("--latency-warmup", min=0)] = 20,
    latency_iterations: Annotated[int, typer.Option("--latency-iterations", min=1)] = 100,
    latency_repetitions: Annotated[int, typer.Option("--latency-repetitions", min=1)] = 5,
    wandb_project: Annotated[str, typer.Option("--wandb-project")] = "jepa4d-worldmodel",
    wandb_entity: Annotated[str | None, typer.Option("--wandb-entity")] = None,
    run_name: Annotated[str, typer.Option("--run-name")] = "phase2e-final-evaluation",
) -> None:
    if len(shard_dirs) != 4:
        raise typer.BadParameter("repeat --shard-dir exactly four times", param_hint="--shard-dir")
    result = run_final_evaluation(
        train_validation_cache,
        test_cache,
        feature_cache_receipt,
        shard_dirs,
        output,
        test_receipt=test_receipt,
        device_name=device,
        batch_size=batch_size,
        latency_warmup=latency_warmup,
        latency_iterations=latency_iterations,
        latency_repetitions=latency_repetitions,
        wandb_enabled=True,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        run_name=run_name,
    )
    typer.echo(json.dumps(result["gate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    app()
