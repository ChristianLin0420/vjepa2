"""Arm registry, gradient-firewall audit, and profiling hooks for Phase 2f."""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypeVar

import torch
from torch import nn

from jepa4d.models.phase2f_scale_geometry import (
    DEFAULT_PHASE2F_ARMS,
    OPTIONAL_PHASE2F_ARMS,
    PHASE2F_COMPONENTS,
    Phase2fArm,
    Phase2fGeometryConfig,
    Phase2fGeometryOutput,
    Phase2fScaleGeometryProbe,
)
from jepa4d.training.phase2f_losses import Phase2fLossConfig, Phase2fLossResult, phase2f_loss

_T = TypeVar("_T")
PHASE2F_CHECKPOINT_SCHEMA = "jepa4d-phase2f-checkpoint-v1"


def phase2f_arm_configs(
    input_dim: int,
    *,
    include_optional_m4: bool = False,
) -> dict[Phase2fArm, Phase2fGeometryConfig]:
    """Return the registered development matrix; M4 is opt-in only."""

    arms = DEFAULT_PHASE2F_ARMS + (OPTIONAL_PHASE2F_ARMS if include_optional_m4 else ())
    return {arm: Phase2fGeometryConfig(input_dim=input_dim, arm=arm) for arm in arms}


@dataclass(frozen=True, slots=True)
class GradientFirewallReport:
    """Gradient norms from each objective into every owned parameter group."""

    norms: dict[str, dict[str, float]]
    maximum_forbidden_norm: float
    tolerance: float
    passed: bool


@dataclass(frozen=True, slots=True)
class Phase2fTrainingStepResult:
    metrics: dict[str, float]
    firewall: GradientFirewallReport | None
    parameter_counts: dict[str, int]


def phase2f_checkpoint_payload(model: Phase2fScaleGeometryProbe) -> dict[str, Any]:
    """Build a CPU, schema-bound checkpoint with an exact parameter receipt."""

    return {
        "schema_version": PHASE2F_CHECKPOINT_SCHEMA,
        "config": asdict(model.config),
        "state_dict": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
        "parameter_counts": model.parameter_counts(),
    }


def save_phase2f_checkpoint(model: Phase2fScaleGeometryProbe, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(phase2f_checkpoint_payload(model), path)
    return path


def load_phase2f_checkpoint(
    path: Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[Phase2fScaleGeometryProbe, dict[str, Any]]:
    """Strictly reload a Phase 2f checkpoint and verify its parameter receipt."""

    payload: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != PHASE2F_CHECKPOINT_SCHEMA:
        raise ValueError(f"unexpected Phase 2f checkpoint schema: {payload.get('schema_version')}")
    config_values = payload.get("config")
    state_dict = payload.get("state_dict")
    if not isinstance(config_values, dict) or not isinstance(state_dict, dict):
        raise ValueError("Phase 2f checkpoint is missing config or state_dict")
    config = Phase2fGeometryConfig(**config_values)
    model = Phase2fScaleGeometryProbe(config).to(device)
    model.load_state_dict(state_dict, strict=True)
    if payload.get("parameter_counts") != model.parameter_counts():
        raise ValueError("Phase 2f checkpoint parameter receipt does not match the reconstructed model")
    return model, payload


def phase2f_outputs_equal(left: Phase2fGeometryOutput, right: Phase2fGeometryOutput) -> bool:
    """Require exact equality for every exposed prediction factor."""

    if left.arm != right.arm:
        return False
    fields = (
        "log_depth",
        "log_variance",
        "centered_shape",
        "global_log_scale",
        "coarse_scale_field",
        "scale_field",
        "shape_log_variance",
        "global_scale_log_variance",
        "canonical_camera_features",
    )
    for field in fields:
        first = getattr(left, field)
        second = getattr(right, field)
        if first is None or second is None:
            if first is not None or second is not None:
                return False
        elif not torch.equal(first, second):
            return False
    return True


def assert_strict_phase2f_reload(
    original: Phase2fScaleGeometryProbe,
    reloaded: Phase2fScaleGeometryProbe,
    features: torch.Tensor,
    *,
    intrinsics: torch.Tensor | None = None,
    intrinsics_image_size: tuple[int, int] | None = None,
) -> None:
    """Fail when a strict reload changes any mean, field, camera, or variance output."""

    original_state = original.state_dict()
    reloaded_state = reloaded.state_dict()
    if tuple(original_state) != tuple(reloaded_state) or any(
        not torch.equal(original_state[name].detach().cpu(), reloaded_state[name].detach().cpu())
        for name in original_state
    ):
        raise RuntimeError(f"strict Phase 2f checkpoint reload changed {original.config.arm} state tensors")
    original_was_training = original.training
    reloaded_was_training = reloaded.training
    original.eval()
    reloaded.eval()
    try:
        with torch.inference_mode():
            expected = original(
                features,
                intrinsics=intrinsics,
                intrinsics_image_size=intrinsics_image_size,
            )
            actual = reloaded(
                features,
                intrinsics=intrinsics,
                intrinsics_image_size=intrinsics_image_size,
            )
    finally:
        original.train(original_was_training)
        reloaded.train(reloaded_was_training)
    if not phase2f_outputs_equal(expected, actual):
        raise RuntimeError(f"strict Phase 2f checkpoint reload changed {original.config.arm} predictions")


def _gradient_norm(
    objective: torch.Tensor,
    parameters: list[nn.Parameter],
    *,
    retain_graph: bool,
) -> float:
    if not parameters or not objective.requires_grad:
        return 0.0
    gradients = torch.autograd.grad(
        objective,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    squared = objective.new_zeros(())
    for gradient in gradients:
        if gradient is not None:
            squared = squared + gradient.detach().float().square().sum()
    return float(squared.sqrt())


def audit_gradient_firewall(
    model: Phase2fScaleGeometryProbe,
    loss: Phase2fLossResult,
    *,
    tolerance: float = 0.0,
) -> GradientFirewallReport:
    """Raise if any objective reaches a parameter group it does not own."""

    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("firewall tolerance must be finite and non-negative")
    groups = {
        "shape": list(model.shape_parameters()),
        "scale": list(model.scale_parameters()),
        "field": list(model.field_parameters()),
    }
    objectives = {
        "shape": loss.shape_objective,
        "scale": loss.scale_objective,
        "field": loss.field_objective,
    }
    norms: dict[str, dict[str, float]] = {}
    for objective_name, objective in objectives.items():
        norms[objective_name] = {
            group_name: _gradient_norm(objective, parameters, retain_graph=True)
            for group_name, parameters in groups.items()
        }
    forbidden = (
        norms["shape"]["scale"],
        norms["shape"]["field"],
        norms["scale"]["shape"],
        norms["scale"]["field"],
        norms["field"]["shape"],
        norms["field"]["scale"],
    )
    maximum = max(forbidden)
    report = GradientFirewallReport(
        norms=norms,
        maximum_forbidden_norm=maximum,
        tolerance=tolerance,
        passed=maximum <= tolerance,
    )
    if not report.passed:
        raise RuntimeError(f"Phase 2f gradient firewall failed: forbidden norm {maximum:.6e} exceeds {tolerance:.6e}")
    return report


def _owned_gradient_norm(parameters: list[nn.Parameter]) -> float:
    squared = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            squared += float(parameter.grad.detach().float().square().sum())
    return math.sqrt(squared)


def train_phase2f_step(
    model: Phase2fScaleGeometryProbe,
    optimizer: torch.optim.Optimizer,
    features: torch.Tensor,
    target_depth: torch.Tensor,
    *,
    intrinsics: torch.Tensor | None = None,
    intrinsics_image_size: tuple[int, int] | None = None,
    valid_mask: torch.Tensor | None = None,
    loss_config: Phase2fLossConfig | None = None,
    group_count: int | None = None,
    views: int = 1,
    maximum_gradient_norm: float = 5.0,
    verify_firewall: bool = True,
    firewall_tolerance: float = 0.0,
) -> Phase2fTrainingStepResult:
    """Run one optimizer step with separated loss and gradient diagnostics."""

    if not math.isfinite(maximum_gradient_norm) or maximum_gradient_norm <= 0:
        raise ValueError("maximum_gradient_norm must be finite and positive")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(
        features,
        intrinsics=intrinsics,
        intrinsics_image_size=intrinsics_image_size,
    )
    loss = phase2f_loss(
        output,
        target_depth,
        valid_mask=valid_mask,
        config=loss_config,
        group_count=group_count,
        views=views,
    )
    firewall = audit_gradient_firewall(model, loss, tolerance=firewall_tolerance) if verify_firewall else None
    loss.total.backward()
    groups = {
        "shape": list(model.shape_parameters()),
        "scale": list(model.scale_parameters()),
        "field": list(model.field_parameters()),
    }
    group_norms = {name: _owned_gradient_norm(parameters) for name, parameters in groups.items()}
    unclipped = torch.nn.utils.clip_grad_norm_(model.parameters(), maximum_gradient_norm)
    if not torch.isfinite(unclipped):
        raise RuntimeError("Phase 2f training produced a non-finite gradient norm")
    optimizer.step()
    metrics = {name: float(value) for name, value in loss.components.items()}
    metrics.update({f"gradient_norm_{name}": value for name, value in group_norms.items()})
    metrics["gradient_norm_total_before_clip"] = float(unclipped)
    if firewall is not None:
        metrics["gradient_firewall_max_forbidden_norm"] = firewall.maximum_forbidden_norm
        for objective, destinations in firewall.norms.items():
            for destination, norm in destinations.items():
                metrics[f"gradient_firewall_{objective}_to_{destination}"] = norm
    return Phase2fTrainingStepResult(
        metrics=metrics,
        firewall=firewall,
        parameter_counts=model.parameter_counts(),
    )


class ComponentLatencyRecorder:
    """Synchronized callable hook used by the model around named components."""

    def __init__(self, synchronize: Callable[[], None] | None = None) -> None:
        self.synchronize = synchronize or (lambda: None)
        self.samples_ns: dict[str, list[int]] = {name: [] for name in PHASE2F_COMPONENTS}

    def __call__(self, name: str, operation: Callable[[], _T]) -> _T:
        if name not in self.samples_ns:
            raise ValueError(f"unknown Phase 2f latency component: {name}")
        self.synchronize()
        started = time.perf_counter_ns()
        result = operation()
        self.synchronize()
        self.samples_ns[name].append(time.perf_counter_ns() - started)
        return result

    def summary_ms(self) -> dict[str, dict[str, float | int | None]]:
        summary: dict[str, dict[str, float | int | None]] = {}
        for name, values_ns in self.samples_ns.items():
            values_ms = [value / 1_000_000.0 for value in values_ns]
            summary[name] = {
                "count": len(values_ms),
                "mean": None if not values_ms else statistics.fmean(values_ms),
                "median": None if not values_ms else statistics.median(values_ms),
                "minimum": None if not values_ms else min(values_ms),
                "maximum": None if not values_ms else max(values_ms),
            }
        return summary


@dataclass(frozen=True, slots=True)
class Phase2fLatencyReport:
    arm: Phase2fArm
    warmups: int
    repeats: int
    component_ms: dict[str, dict[str, float | int | None]]
    end_to_end_ms: dict[str, float]
    parameter_counts: dict[str, int]


def _device_synchronizer(device: torch.device) -> Callable[[], None]:
    if device.type != "cuda":
        return lambda: None
    return lambda: torch.cuda.synchronize(device)


def profile_phase2f_latency(
    model: Phase2fScaleGeometryProbe,
    features: torch.Tensor,
    *,
    intrinsics: torch.Tensor | None = None,
    intrinsics_image_size: tuple[int, int] | None = None,
    warmups: int = 5,
    repeats: int = 20,
) -> Phase2fLatencyReport:
    """Measure synchronized component and uninstrumented end-to-end latency."""

    if isinstance(warmups, bool) or not isinstance(warmups, int) or warmups < 0:
        raise ValueError("warmups must be a non-negative integer")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats <= 0:
        raise ValueError("repeats must be a positive integer")
    device = features.device
    synchronize = _device_synchronizer(device)
    recorder = ComponentLatencyRecorder(synchronize)
    was_training = model.training
    model.eval()
    end_to_end_ns: list[int] = []
    try:
        with torch.inference_mode():
            for _ in range(warmups):
                model(
                    features,
                    intrinsics=intrinsics,
                    intrinsics_image_size=intrinsics_image_size,
                )
            for _ in range(repeats):
                model(
                    features,
                    intrinsics=intrinsics,
                    intrinsics_image_size=intrinsics_image_size,
                    timing_hook=recorder,
                )
            for _ in range(repeats):
                synchronize()
                started = time.perf_counter_ns()
                model(
                    features,
                    intrinsics=intrinsics,
                    intrinsics_image_size=intrinsics_image_size,
                )
                synchronize()
                end_to_end_ns.append(time.perf_counter_ns() - started)
    finally:
        model.train(was_training)
    end_to_end = [value / 1_000_000.0 for value in end_to_end_ns]
    return Phase2fLatencyReport(
        arm=model.config.arm,
        warmups=warmups,
        repeats=repeats,
        component_ms=recorder.summary_ms(),
        end_to_end_ms={
            "mean": statistics.fmean(end_to_end),
            "median": statistics.median(end_to_end),
            "minimum": min(end_to_end),
            "maximum": max(end_to_end),
        },
        parameter_counts=model.parameter_counts(),
    )
