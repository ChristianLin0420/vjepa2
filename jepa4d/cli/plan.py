"""Run a deterministic verified pick-and-place episode."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import torch
import typer

from jepa4d.models.latent_dynamics import ActionConditionedLatentDynamics
from jepa4d.planning.execution import VerifiedTaskPlanner
from jepa4d.planning.latent_mpc import CEMConfig, CEMPlanner
from jepa4d.planning.task_graph import TaskGraph
from jepa4d.planning.verification import VerificationPolicy
from jepa4d.robotics.mock_robot import MockRobot
from jepa4d.visualization.observability import ExperimentLogger

app = typer.Typer(add_completion=False)


@app.command()
def main(
    object_name: Annotated[str, typer.Option("--object")] = "mug",
    destination: Annotated[str, typer.Option("--destination")] = "table",
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("outputs/phase5_plan/trace.json"),
    confidence_threshold: Annotated[float, typer.Option("--confidence-threshold")] = 0.8,
    inject_pick_failure: Annotated[bool, typer.Option("--inject-pick-failure/--no-inject-pick-failure")] = False,
    feature_artifact: Annotated[Path | None, typer.Option("--feature-artifact")] = None,
    device: Annotated[str, typer.Option("--device")] = "cpu",
    wandb: Annotated[bool, typer.Option("--wandb/--no-wandb")] = False,
    wandb_project: Annotated[str, typer.Option("--wandb-project")] = "jepa4d-worldmodel",
    wandb_mode: Annotated[str, typer.Option("--wandb-mode")] = "online",
    run_name: Annotated[str, typer.Option("--run-name")] = "phase5-verified-planning",
) -> None:
    logger = ExperimentLogger(
        enabled=wandb,
        project=wandb_project,
        name=run_name,
        mode=wandb_mode,
        tags=["phase-5", "planning", "verified-execution", device],
        config={
            "object": object_name,
            "destination": destination,
            "confidence_threshold": confidence_threshold,
            "inject_pick_failure": inject_pick_failure,
            "feature_artifact": None if feature_artifact is None else str(feature_artifact),
            "device": device,
        },
    )
    robot = MockRobot(
        objects={object_name: "counter"},
        fail_once={f"pick:{object_name}"} if inject_pick_failure else set(),
    )
    graph = TaskGraph.pick_and_place(object_name, destination)
    trace = VerifiedTaskPlanner(verification=VerificationPolicy(confidence_threshold)).execute(graph, robot)
    mpc_payload = None
    if feature_artifact is not None:
        feature_payload = torch.load(feature_artifact, map_location=device, weights_only=True)
        global_tokens = feature_payload["global_tokens"]
        tokens = global_tokens.reshape(1, -1, global_tokens.shape[-1]).to(device)
        dynamics = ActionConditionedLatentDynamics().to(device)
        mpc_plan = CEMPlanner(7, CEMConfig(horizon=3, population=64, iterations=3, seed=5)).plan(tokens, dynamics)
        mpc_payload = {
            "score": mpc_plan.score,
            "predicted_uncertainty": mpc_plan.predicted_uncertainty,
            "horizon": mpc_plan.actions.shape[0],
            "action_dim": mpc_plan.actions.shape[1],
            "action_abs_mean": float(mpc_plan.actions.abs().mean()),
            "token_count": tokens.shape[1],
            "token_dim": tokens.shape[2],
            "real_vjepa_features": True,
            "actions": mpc_plan.actions.cpu().tolist(),
        }
    logger.log_planning_trace(trace, mpc=mpc_payload)
    payload = trace.to_serializable()
    payload["mpc"] = mpc_payload
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    logger.log_artifact(output, "planning-trace")
    logger.finish(
        {
            "result": "success" if trace.success else "failure",
            "task_success": trace.success,
            "replans": trace.replans,
            "verification_actions": trace.verification_actions,
            "device": device,
        }
    )
    typer.echo(
        json.dumps(
            {"trace": str(output), "success": trace.success, "replans": trace.replans, "wandb_url": logger.url},
            indent=2,
        )
    )
    if not trace.success:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
