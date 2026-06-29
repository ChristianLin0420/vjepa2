"""Same-checkpoint causal attribution for Phase-2c residual layer fusion.

This module deliberately does not train a model.  It reloads one learned-fusion
checkpoint, keeps its dense probe fixed, and intervenes only on the three scalar
fusion gates.  The resulting comparisons diagnose whether a selected checkpoint
benefits from its learned gates or merely from the jointly trained probe.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F

from jepa4d.models.geometry_student import ResidualFusionGeometryProbe

LAYER_ORDER = (2, 5, 8)
PHASE2C_SCHEMA = "jepa4d-phase2c-cross-sequence-comparison-v1"
PHASE2D_SCHEMA = "jepa4d-phase2d-same-checkpoint-fusion-attribution-v1"
DEPTH_PREDICTION_SCHEMA = "jepa4d-phase2d-depth-predictions-v1"
QUALITATIVE_SCHEMA = "jepa4d-phase2d-qualitative-v1"

InterventionFamily = Literal["original", "zero", "fixed_average", "layer_permutation", "sign_flip"]


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest for an artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class GateIntervention:
    """A deterministic replacement for the learned raw gate vector."""

    intervention_id: str
    family: InterventionFamily
    description: str
    raw_gates: tuple[float, float, float]
    layer_order: tuple[int, int, int] = LAYER_ORDER
    source_layer_order: tuple[int, int, int] | None = None
    flipped_layers: tuple[int, ...] = ()

    @property
    def effective_coefficients(self) -> tuple[float, float, float]:
        return tuple(math.tanh(value) / len(self.layer_order) for value in self.raw_gates)  # type: ignore[return-value]

    @property
    def final_coefficient(self) -> float:
        return 1.0 - sum(self.effective_coefficients)

    def to_serializable(self) -> dict[str, Any]:
        return {
            "intervention_id": self.intervention_id,
            "family": self.family,
            "description": self.description,
            "layer_order": list(self.layer_order),
            "source_layer_order": None if self.source_layer_order is None else list(self.source_layer_order),
            "flipped_layers": list(self.flipped_layers),
            "raw_gates": {str(layer): value for layer, value in zip(self.layer_order, self.raw_gates, strict=True)},
            "effective_coefficients": {
                str(layer): value for layer, value in zip(self.layer_order, self.effective_coefficients, strict=True)
            },
            "final_coefficient": self.final_coefficient,
        }


@dataclass(slots=True)
class Phase2CArtifacts:
    """Content-validated subset of a completed Phase-2c output directory."""

    root: Path
    comparison: dict[str, Any]
    comparison_sha256: str
    normalization: dict[str, dict[str, torch.Tensor]]
    normalization_sha256: str
    learned_rows: dict[int, dict[str, Any]]
    checkpoints: dict[int, dict[str, Any]]
    checkpoint_paths: dict[int, Path]


def build_gate_interventions(raw_gates: torch.Tensor | Sequence[float]) -> list[GateIntervention]:
    """Build the registered original/zero/fixed/permutation/sign controls.

    Five non-identity layer permutations test whether an observed gain depends on
    assigning a coefficient to a particular intermediate layer.  Seven non-empty
    sign-flip subsets test every alternative sign pattern while preserving each
    learned magnitude.  ``fixed_average`` uses ``atanh(0.75)`` because the model's
    effective coefficient is ``tanh(g) / 3``; this makes all four layer weights
    exactly 0.25 up to floating-point precision.
    """
    values = torch.as_tensor(raw_gates, dtype=torch.float64).flatten()
    if tuple(values.shape) != (3,) or not torch.isfinite(values).all():
        raise ValueError("raw_gates must contain exactly three finite values")
    learned = (float(values[0]), float(values[1]), float(values[2]))
    fixed_raw = math.atanh(0.75)
    interventions = [
        GateIntervention(
            "original",
            "original",
            "Checkpoint gates without intervention.",
            learned,
        ),
        GateIntervention(
            "zero",
            "zero",
            "All residual gates are zero; this is the same trained probe on the final feature only.",
            (0.0, 0.0, 0.0),
        ),
        GateIntervention(
            "fixed_average",
            "fixed_average",
            "All gates are atanh(0.75), giving 0.25 weight to final/layer-2/layer-5/layer-8.",
            (fixed_raw, fixed_raw, fixed_raw),
        ),
    ]
    for permutation in itertools.permutations(range(3)):
        if permutation == (0, 1, 2):
            continue
        source_layers = tuple(LAYER_ORDER[index] for index in permutation)
        permuted = tuple(learned[index] for index in permutation)
        label = "_".join(str(value) for value in source_layers)
        interventions.append(
            GateIntervention(
                f"permute_sources_{label}",
                "layer_permutation",
                f"Assign learned source-layer gates {source_layers} to target layers {LAYER_ORDER}.",
                permuted,  # type: ignore[arg-type]
                source_layer_order=source_layers,  # type: ignore[arg-type]
            )
        )
    for width in range(1, len(LAYER_ORDER) + 1):
        for positions in itertools.combinations(range(len(LAYER_ORDER)), width):
            flipped_layers = tuple(LAYER_ORDER[index] for index in positions)
            signed = tuple(-value if index in positions else value for index, value in enumerate(learned))
            label = "_".join(str(value) for value in flipped_layers)
            interventions.append(
                GateIntervention(
                    f"sign_flip_{label}",
                    "sign_flip",
                    f"Flip learned gate signs for layers {flipped_layers}; preserve all magnitudes.",
                    signed,  # type: ignore[arg-type]
                    flipped_layers=flipped_layers,
                )
            )
    if len(interventions) != 15 or len({value.intervention_id for value in interventions}) != 15:
        raise RuntimeError("the Phase-2d intervention registry must contain exactly 15 unique controls")
    return interventions


def load_phase2c_artifacts(root: Path) -> Phase2CArtifacts:
    """Load and content-validate comparison, normalization, and learned checkpoints."""
    root = root.resolve()
    comparison_path = root / "comparison.json"
    normalization_path = root / "vjepa_learned_fusion-normalization.pt"
    if not comparison_path.is_file() or not normalization_path.is_file():
        raise FileNotFoundError("Phase-2c output must contain comparison.json and learned-fusion normalization")
    comparison = json.loads(comparison_path.read_text())
    if comparison.get("schema_version") != PHASE2C_SCHEMA:
        raise ValueError(f"expected {PHASE2C_SCHEMA}, found {comparison.get('schema_version')!r}")
    if comparison.get("failures"):
        raise ValueError("Phase-2c comparison contains recorded failures")
    artifacts = comparison.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Phase-2c comparison has no artifact hash map")
    expected_normalization_hash = artifacts.get(normalization_path.name)
    normalization_hash = sha256(normalization_path)
    if expected_normalization_hash != normalization_hash:
        raise ValueError("learned-fusion normalization SHA-256 does not match comparison.json")

    rows = [value for value in comparison.get("variants", []) if value.get("variant_id") == "vjepa_learned_fusion"]
    learned_rows = {int(value["seed"]): value for value in rows if value.get("seed") is not None}
    if set(learned_rows) != {0, 1, 2} or len(rows) != 3:
        raise ValueError("Phase-2c attribution requires exactly learned-fusion seeds 0, 1, and 2")

    checkpoint_paths: dict[int, Path] = {}
    checkpoints: dict[int, dict[str, Any]] = {}
    input_dims: set[int] = set()
    for seed, row in learned_rows.items():
        relative = Path("checkpoints") / f"vjepa_learned_fusion-seed{seed}.pt"
        checkpoint_path = root / relative
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"missing learned-fusion checkpoint: {checkpoint_path}")
        actual_hash = sha256(checkpoint_path)
        if row.get("checkpoint_sha256") != actual_hash or artifacts.get(str(relative)) != actual_hash:
            raise ValueError(f"checkpoint SHA-256 mismatch for learned-fusion seed {seed}")
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if (
            payload.get("variant") != "vjepa_learned_fusion"
            or int(payload.get("seed", -1)) != seed
            or payload.get("model_type") != "ResidualFusionGeometryProbe"
        ):
            raise ValueError(f"checkpoint identity mismatch for learned-fusion seed {seed}")
        state = payload.get("state_dict")
        if not isinstance(state, dict) or "fusion.raw_gates" not in state:
            raise ValueError(f"checkpoint seed {seed} has no fusion gate state")
        raw_gates = state["fusion.raw_gates"]
        if tuple(raw_gates.shape) != (3,) or not torch.isfinite(raw_gates).all():
            raise ValueError(f"checkpoint seed {seed} has invalid fusion gates")
        input_dims.add(int(payload["input_dim"]))
        checkpoint_paths[seed] = checkpoint_path
        checkpoints[seed] = payload
    if len(input_dims) != 1:
        raise ValueError(f"learned checkpoints disagree on input dimension: {sorted(input_dims)}")

    normalization = torch.load(normalization_path, map_location="cpu", weights_only=True)
    expected_keys = {"vjepa_final", "vjepa_layer_2", "vjepa_layer_5", "vjepa_layer_8"}
    if not isinstance(normalization, dict) or set(normalization) != expected_keys:
        raise ValueError(f"unexpected learned-fusion normalization keys: {sorted(normalization)}")
    input_dim = next(iter(input_dims))
    for name, statistics in normalization.items():
        if not isinstance(statistics, dict) or set(statistics) != {"mean", "std"}:
            raise ValueError(f"normalization {name} must contain mean and std")
        mean, std = statistics["mean"], statistics["std"]
        if tuple(mean.shape) != (1, input_dim, 1, 1) or tuple(std.shape) != tuple(mean.shape):
            raise ValueError(f"normalization shape mismatch for {name}")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not (std > 0).all():
            raise ValueError(f"normalization values are invalid for {name}")
    return Phase2CArtifacts(
        root=root,
        comparison=comparison,
        comparison_sha256=sha256(comparison_path),
        normalization=normalization,
        normalization_sha256=normalization_hash,
        learned_rows=learned_rows,
        checkpoints=checkpoints,
        checkpoint_paths=checkpoint_paths,
    )


def normalize_phase2c_feature_grids(
    feature_grids: Mapping[str, torch.Tensor], normalization: Mapping[str, Mapping[str, torch.Tensor]]
) -> torch.Tensor:
    """Apply the frozen train-only Phase-2c statistics and stack final/layer-2/5/8."""
    ordered = ("vjepa_final", "vjepa_layer_2", "vjepa_layer_5", "vjepa_layer_8")
    if set(feature_grids) != set(ordered) or set(normalization) != set(ordered):
        raise ValueError("features and normalization must contain final and layers 2/5/8 exactly")
    reference_shape = tuple(feature_grids[ordered[0]].shape)
    if len(reference_shape) != 4:
        raise ValueError("feature grids must have shape [N,C,H,W]")
    normalized = []
    for key in ordered:
        value = feature_grids[key]
        if tuple(value.shape) != reference_shape or not torch.isfinite(value).all():
            raise ValueError(f"feature grid {key} has an invalid shape or non-finite values")
        mean = normalization[key]["mean"]
        std = normalization[key]["std"]
        normalized.append(((value.float() - mean) / std).half())
    result = torch.stack(normalized, dim=1)
    if not torch.isfinite(result).all():
        raise ValueError("normalized Phase-2c feature stack is non-finite")
    return result


def _valid(target: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(target) & (target > 0.1) & (target < 10.0)


def _mean_dict(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot average an empty metric collection")
    keys = tuple(rows[0])
    if any(tuple(row) != keys for row in rows):
        raise ValueError("metric rows have inconsistent keys")
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def _frame_depth_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    mask = _valid(target)
    if not mask.any():
        raise ValueError("depth metric frame has no valid target pixels")
    predicted, truth = prediction[mask].float(), target[mask].float()
    if not torch.isfinite(predicted).all() or not (predicted > 0).all():
        raise ValueError("depth prediction is non-finite or non-positive on valid target pixels")
    error = predicted - truth
    ratio = torch.maximum(predicted / truth, truth / predicted.clamp_min(1e-8))
    scale = truth.median() / predicted.median().clamp_min(1e-8)
    aligned = predicted * scale
    aligned_error = aligned - truth
    aligned_ratio = torch.maximum(aligned / truth, truth / aligned.clamp_min(1e-8))
    return {
        "metric_abs_rel": float((error.abs() / truth).mean()),
        "metric_rmse_m": float(error.square().mean().sqrt()),
        "metric_delta_1": float((ratio < 1.25).float().mean()),
        "aligned_abs_rel": float((aligned_error.abs() / truth).mean()),
        "aligned_rmse_m": float(aligned_error.square().mean().sqrt()),
        "aligned_log_rmse": float((aligned.clamp_min(1e-8).log() - truth.log()).square().mean().sqrt()),
        "aligned_delta_1": float((aligned_ratio < 1.25).float().mean()),
        "aligned_delta_2": float((aligned_ratio < 1.25**2).float().mean()),
        "aligned_delta_3": float((aligned_ratio < 1.25**3).float().mean()),
        "metric_abs_log_scale_error": abs(math.log(max(float(scale), 1e-12))),
    }


def _sequence_depth_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    return _mean_dict(
        [_frame_depth_metrics(prediction, target) for prediction, target in zip(predictions, targets, strict=True)]
    )


def _fit_variance_multiplier(
    predicted_log_depth: torch.Tensor, predicted_log_variance: torch.Tensor, target: torch.Tensor
) -> float:
    mask = _valid(target)
    truth = target.clamp_min(1e-4).log()
    ratio = (predicted_log_depth - truth).square() / predicted_log_variance.exp().clamp_min(1e-8)
    return float(ratio[mask].mean().clamp(1e-4, 1e4))


def _sequence_nll(
    predicted_log_depth: torch.Tensor,
    predicted_log_variance: torch.Tensor,
    target: torch.Tensor,
    multiplier: float,
) -> dict[str, float]:
    mask = _valid(target)
    truth = target.clamp_min(1e-4).log()
    residual = predicted_log_depth - truth
    raw_variance = predicted_log_variance.exp().clamp_min(1e-8)
    calibrated_variance = raw_variance * multiplier
    raw = 0.5 * (raw_variance.log() + residual.square() / raw_variance)
    calibrated = 0.5 * (calibrated_variance.log() + residual.square() / calibrated_variance)
    return {
        "raw_log_depth_nll": float(raw[mask].mean()),
        "calibrated_log_depth_nll": float(calibrated[mask].mean()),
    }


def _prediction_delta(prediction: torch.Tensor, reference: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    mask = _valid(target)
    delta = prediction[mask].float() - reference[mask].float()
    reference_values = reference[mask].float().clamp_min(1e-8)
    return {
        "prediction_delta_abs_m": float(delta.abs().mean()),
        "prediction_delta_rmse_m": float(delta.square().mean().sqrt()),
        "prediction_delta_relative": float((delta.abs() / reference_values).mean()),
        "prediction_delta_log_rmse": float(
            (prediction[mask].float().clamp_min(1e-8).log() - reference_values.log()).square().mean().sqrt()
        ),
        "prediction_delta_max_m": float(delta.abs().max()),
    }


def _residual_contribution_rows(
    features: torch.Tensor, intervention: GateIntervention, *, batch_size: int
) -> dict[str, torch.Tensor]:
    """Return per-frame residual L2 norms and ratios without retaining feature-sized intermediates."""
    coefficients = torch.tensor(intervention.effective_coefficients, dtype=torch.float32).view(1, 3, 1, 1, 1)
    rows: dict[str, list[torch.Tensor]] = {
        "final_feature_norm": [],
        "residual_total_norm": [],
        "residual_total_norm_ratio": [],
        **{f"residual_layer_{layer}_norm": [] for layer in LAYER_ORDER},
        **{f"residual_layer_{layer}_norm_ratio": [] for layer in LAYER_ORDER},
    }
    for offset in range(0, len(features), batch_size):
        value = features[offset : offset + batch_size].float()
        final = value[:, 0]
        residuals = coefficients * (value[:, 1:] - final.unsqueeze(1))
        final_norm = final.flatten(1).norm(dim=1).clamp_min(1e-12)
        contribution_norms = residuals.flatten(2).norm(dim=2)
        total_norm = residuals.sum(dim=1).flatten(1).norm(dim=1)
        rows["final_feature_norm"].append(final_norm)
        rows["residual_total_norm"].append(total_norm)
        rows["residual_total_norm_ratio"].append(total_norm / final_norm)
        for layer_index, layer in enumerate(LAYER_ORDER):
            rows[f"residual_layer_{layer}_norm"].append(contribution_norms[:, layer_index])
            rows[f"residual_layer_{layer}_norm_ratio"].append(contribution_norms[:, layer_index] / final_norm)
    return {key: torch.cat(values) for key, values in rows.items()}


def _predict(
    model: ResidualFusionGeometryProbe, features: torch.Tensor, *, device: str, batch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    log_depths, log_variances = [], []
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(features), batch_size):
            log_depth, log_variance = model(features[offset : offset + batch_size].to(device))
            log_depths.append(log_depth.cpu())
            log_variances.append(log_variance.cpu())
    return torch.cat(log_depths), torch.cat(log_variances)


def evaluate_checkpoint_attribution(
    checkpoint_payload: Mapping[str, Any],
    comparison_row: Mapping[str, Any],
    validation_features: torch.Tensor,
    test_features: torch.Tensor,
    validation_targets_24: torch.Tensor,
    test_targets_24: torch.Tensor,
    test_targets_full: torch.Tensor,
    test_sequence_ids: Sequence[str],
    *,
    device: str = "cpu",
    batch_size: int = 8,
    interventions: Sequence[GateIntervention] | None = None,
    prediction_callback: (
        Callable[
            [int, GateIntervention, torch.Tensor, torch.Tensor, torch.Tensor, float],
            None,
        ]
        | None
    ) = None,
) -> dict[str, Any]:
    """Evaluate gate interventions while holding the checkpoint probe exactly fixed."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if len(test_features) != len(test_targets_24) or len(test_features) != len(test_targets_full):
        raise ValueError("test feature and target counts differ")
    if len(test_features) != len(test_sequence_ids):
        raise ValueError("test sequence ID count differs from feature count")
    if len(validation_features) != len(validation_targets_24):
        raise ValueError("validation feature and target counts differ")
    if test_features.ndim != 5 or test_features.shape[1] != 4:
        raise ValueError("attribution features must have shape [N,4,C,H,W]")

    state = checkpoint_payload["state_dict"]
    input_dim = int(checkpoint_payload["input_dim"])
    hidden_dim = int(state["probe.network.0.weight"].shape[0])
    layer_order = tuple(
        int(value) for value in checkpoint_payload.get("fusion_state", {}).get("layer_order", LAYER_ORDER)
    )
    if layer_order != LAYER_ORDER:
        raise ValueError(f"Phase-2d expects layer order {LAYER_ORDER}, found {layer_order}")
    model = ResidualFusionGeometryProbe(input_dim, hidden_dim=hidden_dim, layer_order=LAYER_ORDER).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    original_raw = state["fusion.raw_gates"].detach().cpu().float()
    controls = list(interventions or build_gate_interventions(original_raw))
    if not controls or controls[0].intervention_id != "original":
        raise ValueError("the first intervention must be the original checkpoint gates")

    sequence_indices: dict[str, list[int]] = {}
    for index, sequence_id in enumerate(test_sequence_ids):
        sequence_indices.setdefault(str(sequence_id), []).append(index)
    if len(sequence_indices) < 1:
        raise ValueError("attribution evaluation has no test sequences")

    original_prediction_full: torch.Tensor | None = None
    intervention_rows: list[dict[str, Any]] = []
    for control in controls:
        with torch.no_grad():
            model.fusion.raw_gates.copy_(
                torch.tensor(control.raw_gates, device=device, dtype=model.fusion.raw_gates.dtype)
            )
        validation_log_depth, validation_log_variance = _predict(
            model, validation_features, device=device, batch_size=batch_size
        )
        test_log_depth, test_log_variance = _predict(model, test_features, device=device, batch_size=batch_size)
        variance_multiplier = _fit_variance_multiplier(
            validation_log_depth, validation_log_variance, validation_targets_24
        )
        prediction_full = F.interpolate(
            test_log_depth.exp().unsqueeze(1),
            size=tuple(test_targets_full.shape[-2:]),
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        if original_prediction_full is None:
            original_prediction_full = prediction_full.clone()
        if prediction_callback is not None:
            prediction_callback(
                int(checkpoint_payload["seed"]),
                control,
                prediction_full,
                test_log_depth.exp(),
                test_log_variance,
                variance_multiplier,
            )
        contribution_rows = _residual_contribution_rows(test_features, control, batch_size=batch_size)
        per_sequence: list[dict[str, Any]] = []
        sequence_metric_rows: list[Mapping[str, float]] = []
        for sequence_id, indices in sorted(sequence_indices.items()):
            metrics = _sequence_depth_metrics(prediction_full[indices], test_targets_full[indices])
            metrics.update(
                _sequence_nll(
                    test_log_depth[indices],
                    test_log_variance[indices],
                    test_targets_24[indices],
                    variance_multiplier,
                )
            )
            assert original_prediction_full is not None
            metrics.update(
                _prediction_delta(
                    prediction_full[indices], original_prediction_full[indices], test_targets_full[indices]
                )
            )
            metrics.update({key: float(values[indices].mean()) for key, values in contribution_rows.items()})
            per_sequence.append({"sequence_id": sequence_id, "frames": len(indices), "metrics": metrics})
            sequence_metric_rows.append(metrics)
        macro = _mean_dict(sequence_metric_rows)
        macro["variance_multiplier"] = variance_multiplier
        intervention_rows.append(
            {
                "intervention": control.to_serializable(),
                "macro": macro,
                "per_sequence": per_sequence,
            }
        )

    original_macro = intervention_rows[0]["macro"]
    for row in intervention_rows:
        macro = row["macro"]
        row["causal_delta_from_original"] = {
            "metric_abs_rel_absolute": macro["metric_abs_rel"] - original_macro["metric_abs_rel"],
            "metric_abs_rel_relative": (macro["metric_abs_rel"] / max(original_macro["metric_abs_rel"], 1e-12) - 1.0),
            "aligned_abs_rel_absolute": macro["aligned_abs_rel"] - original_macro["aligned_abs_rel"],
            "calibrated_log_depth_nll_absolute": (
                macro["calibrated_log_depth_nll"] - original_macro["calibrated_log_depth_nll"]
            ),
        }
    with torch.no_grad():
        model.fusion.raw_gates.copy_(original_raw.to(device))
    return {
        "seed": int(checkpoint_payload["seed"]),
        "checkpoint_validation_abs_rel": float(checkpoint_payload["validation_abs_rel"]),
        "phase2c_reported_test_metrics": dict(comparison_row.get("metrics", {})),
        "original_raw_gates": original_raw.tolist(),
        "interventions": intervention_rows,
    }


def write_full_predictions_npz(
    path: Path,
    *,
    predictions: Sequence[torch.Tensor],
    target: torch.Tensor,
    sample_ids: Sequence[str],
    sequence_ids: Sequence[str],
    variant_ids: Sequence[str],
    seeds: Sequence[int],
) -> Path:
    """Persist full-resolution Phase-2d predictions in the shared audit schema."""
    if not predictions:
        raise ValueError("full-prediction handoff requires at least one variant")
    target = target.detach().cpu().float().contiguous()
    if target.ndim != 3 or not torch.isfinite(target).all():
        raise ValueError("target must be a finite [N,H,W] tensor")
    variant_count = len(predictions)
    sample_count = len(target)
    if len(sample_ids) != sample_count or len(sequence_ids) != sample_count:
        raise ValueError("sample/sequence IDs must match the target frame count")
    if len(variant_ids) != variant_count or len(seeds) != variant_count:
        raise ValueError("variant IDs and seeds must match the prediction variant count")
    if len(set(variant_ids)) != variant_count:
        raise ValueError("full-prediction variant IDs must be unique")
    prediction_values = []
    for variant_id, prediction in zip(variant_ids, predictions, strict=True):
        value = prediction.detach().cpu().float().contiguous()
        if tuple(value.shape) != tuple(target.shape):
            raise ValueError(f"prediction {variant_id} shape {tuple(value.shape)} != target {tuple(target.shape)}")
        if not torch.isfinite(value).all() or not (value > 0).all():
            raise ValueError(f"prediction {variant_id} is non-finite or non-positive")
        prediction_values.append(value.numpy())
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(
            stream,
            schema_version=np.asarray(DEPTH_PREDICTION_SCHEMA),
            prediction_m=np.stack(prediction_values),
            target_m=target.numpy(),
            sample_ids=np.asarray(sample_ids),
            sequence_ids=np.asarray(sequence_ids),
            variant_ids=np.asarray(variant_ids),
            seeds=np.asarray(seeds, dtype=np.int64),
            audit_scope=np.asarray("full_phase2c_test"),
        )
    temporary.replace(path)
    return path


def write_qualitative_examples_npz(
    path: Path,
    *,
    predictions: Sequence[torch.Tensor],
    log_variances: Sequence[torch.Tensor],
    calibrated_log_depth_sigmas: Sequence[torch.Tensor],
    target: torch.Tensor,
    sample_ids: Sequence[str],
    sequence_ids: Sequence[str],
    variant_ids: Sequence[str],
    seeds: Sequence[int],
) -> Path:
    """Persist a bounded, fixed-sample qualitative bundle for local/W&B panels."""
    variant_count = len(predictions)
    if variant_count < 1 or len(target) < 1 or len(target) > 8:
        raise ValueError("qualitative handoff requires one or more variants and between one and eight fixed samples")
    if not (len(log_variances) == len(calibrated_log_depth_sigmas) == len(variant_ids) == len(seeds) == variant_count):
        raise ValueError("qualitative variant tensors and identities have inconsistent lengths")
    if len(sample_ids) != len(target) or len(sequence_ids) != len(target):
        raise ValueError("qualitative sample/sequence identities differ from the fixed target set")
    if len(set(sample_ids)) != len(sample_ids) or len(set(variant_ids)) != len(variant_ids):
        raise ValueError("qualitative sample and variant identities must be unique")
    target_value = target.detach().cpu().float().contiguous()
    if target_value.ndim != 3 or not torch.isfinite(target_value).all():
        raise ValueError("qualitative target must be finite [Q,H,W]")
    prediction_values, log_variance_values, sigma_values = [], [], []
    for variant_id, prediction, log_variance, sigma in zip(
        variant_ids,
        predictions,
        log_variances,
        calibrated_log_depth_sigmas,
        strict=True,
    ):
        prediction_value = prediction.detach().cpu().float().contiguous()
        log_variance_value = log_variance.detach().cpu().float().contiguous()
        sigma_value = sigma.detach().cpu().float().contiguous()
        if not (prediction_value.shape == log_variance_value.shape == sigma_value.shape == target_value.shape):
            raise ValueError(f"qualitative tensor shape mismatch for {variant_id}")
        if (
            not torch.isfinite(prediction_value).all()
            or not torch.isfinite(log_variance_value).all()
            or not torch.isfinite(sigma_value).all()
            or not bool((prediction_value > 0).all())
            or not bool((sigma_value > 0).all())
        ):
            raise ValueError(f"qualitative tensor is invalid for {variant_id}")
        prediction_values.append(prediction_value.numpy())
        log_variance_values.append(log_variance_value.numpy())
        sigma_values.append(sigma_value.numpy())
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            schema_version=np.asarray(QUALITATIVE_SCHEMA),
            prediction_m=np.stack(prediction_values),
            target_m=target_value.numpy(),
            log_variance=np.stack(log_variance_values),
            calibrated_log_depth_sigma=np.stack(sigma_values),
            sample_ids=np.asarray(sample_ids),
            sequence_ids=np.asarray(sequence_ids),
            variant_ids=np.asarray(variant_ids),
            seeds=np.asarray(seeds, dtype=np.int64),
            selection_policy=np.asarray("deterministic sequence-balanced fixed samples; maximum four"),
        )
    temporary.replace(path)
    return path


def aggregate_seed_attributions(seed_results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate each intervention across seeds without treating seeds as scenes."""
    if not seed_results:
        raise ValueError("cannot aggregate an empty seed collection")
    expected_ids = [row["intervention"]["intervention_id"] for row in seed_results[0]["interventions"]]
    grouped: dict[str, list[Mapping[str, Any]]] = {key: [] for key in expected_ids}
    for seed_result in seed_results:
        observed_ids = [row["intervention"]["intervention_id"] for row in seed_result["interventions"]]
        if observed_ids != expected_ids:
            raise ValueError("seed attribution results have inconsistent intervention registries")
        for row in seed_result["interventions"]:
            grouped[row["intervention"]["intervention_id"]].append(row)
    aggregates = []
    for intervention_id in expected_ids:
        rows = grouped[intervention_id]
        metric_keys = tuple(rows[0]["macro"])
        metrics = {}
        for key in metric_keys:
            values = np.asarray([float(row["macro"][key]) for row in rows], dtype=np.float64)
            metrics[key] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "values": values.tolist(),
            }
        aggregates.append(
            {
                "intervention": rows[0]["intervention"],
                "metrics": metrics,
                "seeds": [int(value["seed"]) for value in seed_results],
            }
        )
    return aggregates


def build_attribution_record(
    *,
    artifacts: Phase2CArtifacts,
    seed_results: Sequence[Mapping[str, Any]],
    dataset_manifest: Path,
    dataset_split_hash: str,
    sample_ids: Sequence[str],
    output_directory: Path,
) -> dict[str, Any]:
    """Build the durable JSON payload for a completed diagnostic run."""
    if dataset_split_hash != artifacts.comparison.get("split_hash"):
        raise ValueError("reloaded dataset split does not match the Phase-2c comparison")
    record = {
        "schema_version": PHASE2D_SCHEMA,
        "evidence_level": "post-hoc-mechanism-diagnostic",
        "status": "complete",
        "source": {
            "phase2c_output": str(artifacts.root),
            "phase2c_experiment_id": artifacts.comparison.get("experiment_id"),
            "phase2c_schema_version": artifacts.comparison.get("schema_version"),
            "phase2c_wandb_url": artifacts.comparison.get("wandb_url"),
            "comparison_sha256": artifacts.comparison_sha256,
            "normalization_sha256": artifacts.normalization_sha256,
            "checkpoint_sha256": {
                str(seed): artifacts.learned_rows[seed]["checkpoint_sha256"] for seed in sorted(artifacts.learned_rows)
            },
            "dataset_manifest": str(dataset_manifest.resolve()),
            "dataset_split_hash": dataset_split_hash,
        },
        "protocol": {
            "causal_unit": "same learned-fusion checkpoint and probe; only fusion.raw_gates are replaced",
            "controls": "original, zero, fixed-average-equivalent, five non-identity layer permutations, seven sign flips",
            "fixed_equivalent_raw_gate": math.atanh(0.75),
            "fixed_equivalent_effective_coefficient": 0.25,
            "normalization": "Phase-2c training-only statistics, unchanged for every intervention",
            "uncertainty": "one multiplier fitted on Freiburg-2 validation pixels separately for each intervention",
            "metric_aggregation": "frame mean within sequence, then equal-weight macro across two Freiburg-3 sequences",
            "prediction_delta_reference": "same-seed original-gate prediction",
            "claim_boundary": (
                "Freiburg-3 was already consumed by Phase 2c. This is a post-hoc causal mechanism diagnostic, "
                "not fresh generalization evidence and not a basis for changing the frozen promotion decision."
            ),
        },
        "test_samples": {"count": len(sample_ids), "sample_ids": list(sample_ids)},
        "seeds": list(seed_results),
        "aggregate": {"controls": aggregate_seed_attributions(seed_results)},
        "output_directory": str(output_directory.resolve()),
    }
    json.dumps(record, allow_nan=False)
    return record
