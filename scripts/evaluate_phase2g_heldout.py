#!/usr/bin/env python3
"""Evaluate one frozen Phase 2g checkpoint on its isolated held-out family."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from jepa4d.evaluation.phase2g_data import file_identity, validate_heldout_shards
from jepa4d.evaluation.phase2g_metrics import (
    evaluate_phase2g_predictions,
    opaque_frame_id,
    scale_mechanism_diagnostics,
)
from jepa4d.evaluation.phase2g_visualization import write_local_qualitative_panels
from jepa4d.training.phase2f_training import load_phase2f_checkpoint
from jepa4d.training.phase2g_protocol import (
    ARMS,
    EVALUATION_RECEIPT_SCHEMA,
    FORMAL_SEEDS,
    FORMAL_TRAINING_RECEIPT_SCHEMA,
    IMAGE_SIZE,
    ROTATIONS,
)
from jepa4d.training.phase2g_runtime import (
    assert_same_execution,
    atomic_json,
    complete_output,
    finish_wandb_run,
    load_execution_provenance,
    load_json,
    prepare_output,
    start_wandb_run,
)
from jepa4d.training.phase2g_training import _normalize, _predict, load_normalization


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--rotation", choices=tuple(ROTATIONS), required=True)
    parser.add_argument("--seed", type=int, choices=FORMAL_SEEDS, required=True)
    parser.add_argument("--input-shard", type=Path, required=True)
    parser.add_argument("--feature-shard", type=Path, required=True)
    parser.add_argument("--target-shard", type=Path, required=True)
    parser.add_argument("--training-receipt", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wandb-entity", default="crlc112358")
    parser.add_argument("--wandb-project", default="jepa4d-worldmodel")
    return parser.parse_args()


def _metrics(
    predictions: Mapping[str, torch.Tensor | None],
    targets: torch.Tensor,
    valid: torch.Tensor,
    *,
    sample_ids: list[str],
    family: str,
    variance_multiplier: float,
) -> dict[str, Any]:
    return evaluate_phase2g_predictions(
        predictions["log_depth"],  # type: ignore[arg-type]
        predictions["log_variance"],  # type: ignore[arg-type]
        targets,
        valid_mask=valid,
        variance_multiplier=variance_multiplier,
        frame_ids=sample_ids,
        family_ids=[family] * len(sample_ids),
    )


def _optimal_scale(
    predictions: Mapping[str, torch.Tensor | None], target: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    centered = predictions["centered_shape"]
    if centered is None:
        raise ValueError("scale diagnostics require factorized centered shape")
    field = predictions["scale_field"]
    correction: torch.Tensor | float = 0.0 if field is None else field
    residual = target.clamp_min(torch.finfo(target.dtype).tiny).log() - centered - correction
    return torch.stack([residual[index][valid[index]].median() for index in range(len(residual))])


def _camera_controls(
    model: Any,
    input_shard: Mapping[str, Any],
    feature_shard: Mapping[str, Any],
    target_shard: Mapping[str, Any],
    normalization: Mapping[str, Any],
    *,
    device: torch.device,
    family: str,
    variance_multiplier: float,
) -> dict[str, Any]:
    if not model.config.consumes_intrinsics:
        feature = _normalize(feature_shard["ordinary_features"][:1, 0], normalization, device)
        camera = input_shard["ordinary_inputs"]["intrinsics_384"][:1, 0].to(device).float()
        rejected = False
        try:
            model(feature, intrinsics=camera, intrinsics_image_size=IMAGE_SIZE)
        except ValueError:
            rejected = True
        if not rejected:
            raise RuntimeError("non-camera arm accepted intrinsics during structural negative control")
        return {
            "status": "not_applicable_nonconsumer",
            "consumes_intrinsics": False,
            "camera_parameters": 0,
            "evaluator_intrinsics_call_rejected": True,
        }
    features = feature_shard["paired_features"][:, 1:].flatten(0, 1)
    target = target_shard["paired_targets"]["depth_24"][:, 1:].flatten(0, 1).float()
    valid = target_shard["paired_targets"]["valid_24"][:, 1:].flatten(0, 1)
    sample_ids = [
        opaque_frame_id(sample_id, profile=f"P{profile}")
        for sample_id in input_shard["samples"]["sample_ids"]
        for profile in range(1, 8)
    ]
    outputs: dict[str, torch.Tensor] = {}
    metrics: dict[str, Any] = {}
    profile_raw_abs_rel: dict[str, dict[str, float]] = {}
    for condition in ("updated", "stale", "wrong", "permuted"):
        intrinsics = input_shard["paired_inputs"][f"{condition}_k"][:, 1:].flatten(0, 1)
        prediction = _predict(model, features, intrinsics, normalization, device=device)
        outputs[condition] = prediction["log_depth"].exp()  # type: ignore[union-attr]
        metrics[condition] = _metrics(
            prediction,
            target,
            valid,
            sample_ids=sample_ids,
            family=family,
            variance_multiplier=variance_multiplier,
        )
        profile_raw_abs_rel[condition] = {}
        for profile_offset, profile_index in enumerate(range(1, 8)):
            indices = torch.arange(profile_offset, len(features), 7)
            profile_prediction = {
                name: None if value is None else value.index_select(0, indices) for name, value in prediction.items()
            }
            profile_metrics = _metrics(
                profile_prediction,
                target.index_select(0, indices),
                valid.index_select(0, indices),
                sample_ids=[
                    opaque_frame_id(sample_id, profile=f"P{profile_index}")
                    for sample_id in input_shard["samples"]["sample_ids"]
                ],
                family=family,
                variance_multiplier=variance_multiplier,
            )
            profile_raw_abs_rel[condition][f"P{profile_index}"] = profile_metrics["equal_family_macro"]["raw_abs_rel"]
    deltas = {
        condition: float((outputs["updated"] - outputs[condition]).abs().double().mean())
        for condition in ("stale", "wrong", "permuted")
    }
    audit = input_shard["audit"]
    return {
        "status": "evaluated",
        "consumes_intrinsics": True,
        "profiles": [f"P{index}" for index in range(1, 8)],
        "identity_profile_excluded": True,
        "metrics": metrics,
        "raw_abs_rel": {condition: value["equal_family_macro"]["raw_abs_rel"] for condition, value in metrics.items()},
        "profile_raw_abs_rel": profile_raw_abs_rel,
        "mean_absolute_prediction_delta_m": deltas,
        "distinct_analytic_intrinsics_per_source_min": min(audit["distinct_updated_intrinsics_per_source"]),
        "distinct_analytic_intrinsics_per_source_max": max(audit["distinct_updated_intrinsics_per_source"]),
        "permutation_assignment_change_fraction": audit["permutation_assignment_change_fraction"],
        "permutation_matrix_change_fraction": audit["permutation_matrix_change_fraction"],
    }


def _aggregate_metric_report(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Remove frame identities/scalars while retaining frozen aggregate metrics."""

    keys = (
        "schema_version",
        "variance_multiplier",
        "frames",
        "valid_frames",
        "failure_count",
        "valid_pixels",
        "per_family",
        "frame_macro",
        "equal_family_macro",
        "coverage",
        "risk_coverage",
        "aggregation",
    )
    return {key: metrics[key] for key in keys}


def _aggregate_mechanism_report(
    *,
    scale_diagnostics: Mapping[str, Any],
    fixed_scale: Mapping[str, Any] | None,
    controls: Mapping[str, Any],
    zero_field: Mapping[str, Any] | None,
) -> dict[str, Any]:
    scale = {key: value for key, value in scale_diagnostics.items() if key != "per_frame"}
    fixed = None
    if fixed_scale is not None:
        fixed = {
            "train_median_log_scale": fixed_scale["train_median_log_scale"],
            "metrics": _aggregate_metric_report(fixed_scale["metrics"]),
        }
    camera = {key: value for key, value in controls.items() if key != "metrics"}
    zero = None
    if zero_field is not None:
        zero = {
            **{
                key: value
                for key, value in zero_field.items()
                if key != "metrics" and not key.startswith("per_frame_")
            },
            "metrics": _aggregate_metric_report(zero_field["metrics"]),
        }
    return {
        "scale_mechanism": scale,
        "fixed_train_median_scale": fixed,
        "camera_controls": camera,
        "zero_field_intervention": zero,
        "per_frame_values_uploaded": False,
        "sample_identifiers_uploaded": False,
    }


def main() -> None:
    args = _parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Phase 2g held-out evaluation may run only inside Slurm")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Phase 2g held-out evaluation requires an allocated CUDA device")
    output = prepare_output(args.output)
    provenance = load_execution_provenance(args.provenance)
    training = load_json(args.training_receipt)
    if (
        training.get("schema_version") != FORMAL_TRAINING_RECEIPT_SCHEMA
        or training.get("status") != "success"
        or (training.get("arm"), training.get("rotation"), training.get("seed"))
        != (args.arm, args.rotation, args.seed)
    ):
        raise ValueError("evaluation training receipt identity/status mismatch")
    assert_same_execution((training,), provenance)
    if training.get("rotation_view", {}).get("heldout_target_exposed") is not False:
        raise ValueError("formal training receipt does not prove held-out isolation")
    input_shard, feature_shard, target_shard, shard_identities = validate_heldout_shards(
        rotation=args.rotation,
        input_path=args.input_shard,
        feature_path=args.feature_shard,
        target_path=args.target_shard,
    )
    heldout_family = ROTATIONS[args.rotation]["heldout"]
    checkpoint_path = Path(training["checkpoint"]["path"])
    normalization_path = Path(training["feature_normalization"]["path"])
    if file_identity(checkpoint_path)["sha256"] != training["checkpoint"]["sha256"]:
        raise ValueError("selected checkpoint changed after formal training")
    if file_identity(normalization_path)["sha256"] != training["feature_normalization"]["sha256"]:
        raise ValueError("feature normalization changed after formal training")
    normalization = load_normalization(normalization_path)
    model, _ = load_phase2f_checkpoint(checkpoint_path, device=device)
    if model.config.arm != args.arm:
        raise ValueError("checkpoint arm differs from evaluation cell")
    features = feature_shard["ordinary_features"][:, 0]
    intrinsics = input_shard["ordinary_inputs"]["intrinsics_384"][:, 0]
    targets = target_shard["ordinary_targets"]["depth_24"][:, 0].float()
    valid = target_shard["ordinary_targets"]["valid_24"][:, 0]
    sample_ids = list(input_shard["samples"]["sample_ids"])
    metric_ids = [opaque_frame_id(value) for value in sample_ids]
    predictions = _predict(model, features, intrinsics, normalization, device=device)
    variance_multiplier = float(training["validation_variance_calibration"]["multiplier"])
    metrics = _metrics(
        predictions,
        targets,
        valid,
        sample_ids=metric_ids,
        family=heldout_family,
        variance_multiplier=variance_multiplier,
    )
    scale_diagnostics: dict[str, Any]
    fixed_scale: dict[str, Any] | None = None
    if args.arm == "M0":
        scale_diagnostics = {"status": "not_applicable_monolithic"}
    else:
        optimal = _optimal_scale(predictions, targets, valid)
        predicted_scale = predictions["global_log_scale"]
        assert predicted_scale is not None
        scale_diagnostics = scale_mechanism_diagnostics(predicted_scale.reshape(-1), optimal, frame_ids=metric_ids)
        if args.arm == "M1":
            fixed_value = float(training["train_median_log_scale"])
            fixed_predictions = dict(predictions)
            fixed_predictions["log_depth"] = (
                predictions["log_depth"] - predicted_scale + fixed_value  # type: ignore[operator]
            )
            fixed_metrics = _metrics(
                fixed_predictions,
                targets,
                valid,
                sample_ids=metric_ids,
                family=heldout_family,
                variance_multiplier=variance_multiplier,
            )
            fixed_scale = {"train_median_log_scale": fixed_value, "metrics": fixed_metrics}
    zero_field: dict[str, Any] | None = None
    if args.arm == "M3":
        scale_field = predictions["scale_field"]
        assert scale_field is not None
        zero_prediction = dict(predictions)
        zero_prediction["log_depth"] = predictions["log_depth"] - scale_field  # type: ignore[operator]
        zero_metrics = _metrics(
            zero_prediction,
            targets,
            valid,
            sample_ids=metric_ids,
            family=heldout_family,
            variance_multiplier=variance_multiplier,
        )
        zero_field = {
            "same_checkpoint": True,
            "metrics": zero_metrics,
            "full_field_mean": float(scale_field.double().mean()),
            "full_field_max_abs": float(scale_field.double().abs().max()),
            "per_frame_field_mean": scale_field.double().mean(dim=(1, 2)).tolist(),
            "per_frame_field_sd": scale_field.double().flatten(1).std(dim=1, unbiased=False).tolist(),
        }
    controls = _camera_controls(
        model,
        input_shard,
        feature_shard,
        target_shard,
        normalization,
        device=device,
        family=heldout_family,
        variance_multiplier=variance_multiplier,
    )
    qualitative_panel, qualitative_manifest, qualitative_ids = write_local_qualitative_panels(
        output / "protected-local-qualitative",
        family=heldout_family,
        sample_ids=sample_ids,
        rgb_uint8=input_shard["ordinary_inputs"]["images_384_uint8"][:, 0],
        target_depth=targets,
        valid_mask=valid,
        log_depth=predictions["log_depth"],  # type: ignore[arg-type]
        log_variance=predictions["log_variance"],  # type: ignore[arg-type]
        scale_field=predictions["scale_field"],
    )
    if len(qualitative_ids) != 16:
        raise RuntimeError("formal held-out qualitative selection did not produce 16 fixed IDs")
    semantic = f"eval-{args.arm.lower()}-{args.rotation.lower()}-s{args.seed}"
    run = start_wandb_run(
        provenance=provenance,
        job_type="heldout-evaluation",
        semantic_name=semantic,
        config={
            "arm": args.arm,
            "rotation": args.rotation,
            "seed": args.seed,
            "heldout_family": heldout_family,
            "checkpoint_sha256": training["checkpoint"]["sha256"],
        },
        entity=args.wandb_entity,
        project=args.wandb_project,
    )
    receipt = {
        "schema_version": EVALUATION_RECEIPT_SCHEMA,
        "status": "success",
        "arm": args.arm,
        "rotation": args.rotation,
        "seed": args.seed,
        "heldout_family": heldout_family,
        "checkpoint": training["checkpoint"],
        "feature_normalization": training["feature_normalization"],
        "validation_variance_calibration": training["validation_variance_calibration"],
        "shards": shard_identities,
        "metrics": metrics,
        "scale_mechanism": scale_diagnostics,
        "fixed_train_median_scale": fixed_scale,
        "camera_controls": controls,
        "zero_field_intervention": zero_field,
        "qualitative": {
            "selection": "lowest_sha256_sample_id_before_training",
            "count": len(qualitative_ids),
            "panel": file_identity(qualitative_panel),
            "manifest": file_identity(qualitative_manifest),
            "contains_protected_rgb_and_target_previews": True,
            "local_only": True,
            "wandb_uploaded": False,
        },
        "training_diagnostics": {
            "first_objective_decile_mean": training["health"]["first_objective_decile_mean"],
            "final_objective_decile_mean": training["health"]["final_objective_decile_mean"],
            "maximum_forbidden_gradient_norm": training["health"]["maximum_forbidden_gradient_norm"],
            "elapsed_seconds": training["elapsed_seconds"],
            "peak_allocated_bytes": training["hardware"]["peak_allocated_bytes"],
            "parameter_counts": training["parameter_counts"],
            "epoch_diagnostics": training["epoch_diagnostics"],
            "resource_metrics_are_descriptive_only": True,
        },
        "all_values_finite": True,
        "checkpoint_frozen_before_heldout_access": True,
        "external_final_authorized": False,
        "external_final_accessed": False,
        "training_receipt": file_identity(args.training_receipt, schema=FORMAL_TRAINING_RECEIPT_SCHEMA),
        "execution_provenance": provenance,
    }
    atomic_json(output / "metrics.json", metrics)
    atomic_json(
        output / "mechanisms.json",
        {
            "scale_mechanism": scale_diagnostics,
            "fixed_train_median_scale": fixed_scale,
            "camera_controls": controls,
            "zero_field_intervention": zero_field,
        },
    )
    aggregate_metrics_path = atomic_json(
        output / "wandb_aggregate_metrics.json",
        {
            "schema_version": "jepa4d-phase2g-heldout-aggregate-report-v1",
            "status": "success",
            "arm": args.arm,
            "rotation": args.rotation,
            "seed": args.seed,
            "heldout_family": heldout_family,
            "metrics": _aggregate_metric_report(metrics),
            "per_frame_values_uploaded": False,
            "sample_identifiers_uploaded": False,
        },
    )
    aggregate_mechanisms_path = atomic_json(
        output / "wandb_aggregate_mechanisms.json",
        {
            "schema_version": "jepa4d-phase2g-heldout-mechanism-report-v1",
            "status": "success",
            **_aggregate_mechanism_report(
                scale_diagnostics=scale_diagnostics,
                fixed_scale=fixed_scale,
                controls=controls,
                zero_field=zero_field,
            ),
        },
    )
    atomic_json(output / "evaluation_receipt.json", receipt)
    macro = metrics["equal_family_macro"]
    run.log({f"heldout/{name}": value for name, value in macro.items()})
    wandb_receipt = finish_wandb_run(
        run,
        artifact_name=f"phase2g-{semantic}-{provenance['execution_id']}",
        job_type="heldout-evaluation",
        files=(aggregate_metrics_path, aggregate_mechanisms_path),
        summary={
            "heldout_raw_abs_rel": macro["raw_abs_rel"],
            "heldout_absolute_log_scale_error": macro["absolute_log_scale_error"],
            "heldout_family": heldout_family,
        },
    )
    complete_output(output, receipt_name="evaluation_receipt.json", receipt=receipt, wandb_receipt=wandb_receipt)
    print(
        json.dumps(
            {"status": "success", "arm": args.arm, "rotation": args.rotation, "seed": args.seed},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
