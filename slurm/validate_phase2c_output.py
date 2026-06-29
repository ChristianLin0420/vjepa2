"""Fail closed when a formal Phase 2c output contract is incomplete."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, TypeGuard

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from slurm.phase2c_gate import EPOCHS, SPLIT_COUNTS, VARIANT_SEEDS, protocol_contract  # noqa: E402

EXPECTED_TEST_SEQUENCES = {
    "freiburg3_long_office_household",
    "freiburg3_structure_texture_far",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, errors: list[str], label: str, default: Any) -> Any:
    if not path.is_file():
        errors.append(f"missing {label}: {path}")
        return default
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as error:
        errors.append(f"cannot read {label}: {error}")
        return default


def _check_finite(value: Any, location: str, errors: list[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _check_finite(item, f"{location}.{key}", errors)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _check_finite(item, f"{location}[{index}]", errors)
    elif isinstance(value, float) and not math.isfinite(value):
        errors.append(f"non-finite value at {location}: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--require-wandb", action="store_true")
    return parser.parse_args()


def _expected_keys() -> set[tuple[str, int | None]]:
    return {(variant, seed) for variant, seeds in VARIANT_SEEDS.items() for seed in seeds}


def _finite_numeric_mapping(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and bool(value)
        and all(
            isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(float(item))
            for item in value.values()
        )
    )


def _positive_finite_number(value: Any) -> TypeGuard[int | float]:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0
    )


def _expected_profile_sample_ids(dataset: dict[str, Any], errors: list[str]) -> list[str]:
    sequences = dataset.get("sequences")
    if not isinstance(sequences, list):
        errors.append("dataset fingerprint sequences are not a list")
        return []
    test_sequences = [row for row in sequences if isinstance(row, dict) and row.get("split") == "test"]
    if len(test_sequences) != 2 or {str(row.get("sequence_id")) for row in test_sequences} != EXPECTED_TEST_SEQUENCES:
        errors.append("dataset fingerprint does not contain the exact two formal test sequences")
        return []
    sample_ids: list[str] = []
    for sequence in test_sequences:
        samples = sequence.get("samples")
        if not isinstance(samples, list) or len(samples) != 64:
            errors.append(f"dataset fingerprint test samples are invalid for {sequence.get('sequence_id')}")
            return []
        for sample in samples:
            sample_id = sample.get("sample_id") if isinstance(sample, dict) else None
            if not isinstance(sample_id, str) or not sample_id:
                errors.append(f"dataset fingerprint has an invalid test sample ID for {sequence.get('sequence_id')}")
                return []
            sample_ids.append(sample_id)
    if len(sample_ids) != 128 or len(set(sample_ids)) != 128:
        errors.append("dataset fingerprint formal test sample IDs are not 128 unique values")
        return []
    selected = sample_ids[::16][:8]
    if len(selected) != 8 or len(set(selected)) != 8:
        errors.append("deterministic formal profile selection is not eight unique sample IDs")
        return []
    return selected


def _results_integrity_valid(
    row_by_key: dict[tuple[str, int | None], dict[str, Any]],
    failures: list[Any],
    result_row_count: int,
) -> bool:
    if failures or result_row_count != len(_expected_keys()) or set(row_by_key) != _expected_keys():
        return False
    try:
        for key, row in row_by_key.items():
            if not _finite_numeric_mapping(row.get("metrics")) or not _finite_numeric_mapping(row.get("runtime")):
                return False
            sequence_metrics = row.get("sequence_metrics")
            if not isinstance(sequence_metrics, dict) or set(sequence_metrics) != EXPECTED_TEST_SEQUENCES:
                return False
            if not all(_finite_numeric_mapping(value) for value in sequence_metrics.values()):
                return False
            expected_macro = sum(float(sequence_metrics[name]["metric_abs_rel"]) for name in sequence_metrics) / 2
            if not math.isclose(float(row["metrics"]["metric_abs_rel"]), expected_macro, rel_tol=0.0, abs_tol=1e-10):
                return False
            if key[1] is None:
                continue
            checkpoint_value = row.get("checkpoint")
            checkpoint = Path(str(checkpoint_value)) if checkpoint_value else None
            metadata = row.get("model_metadata")
            if (
                checkpoint is None
                or not checkpoint.is_file()
                or not row.get("checkpoint_sha256")
                or _sha256(checkpoint) != row.get("checkpoint_sha256")
                or not isinstance(metadata, dict)
                or metadata.get("checkpoint_reload") != "strict-prediction-equality-pass"
            ):
                return False
        for seed in (0, 1, 2):
            final = row_by_key[("vjepa_final", seed)]
            candidate = row_by_key[("vjepa_learned_fusion", seed)]
            final_parameters = final.get("trainable_parameters")
            candidate_parameters = candidate.get("trainable_parameters")
            final_metadata = final.get("model_metadata")
            candidate_metadata = candidate.get("model_metadata")
            if (
                not isinstance(final_parameters, (int, float))
                or isinstance(final_parameters, bool)
                or not math.isfinite(float(final_parameters))
                or not isinstance(candidate_parameters, (int, float))
                or isinstance(candidate_parameters, bool)
                or not math.isfinite(float(candidate_parameters))
                or candidate_parameters != final_parameters + 3
                or not isinstance(final_metadata, dict)
                or not isinstance(candidate_metadata, dict)
            ):
                return False
            probe_initial_sha256 = final_metadata.get("probe_initial_sha256")
            if (
                not isinstance(probe_initial_sha256, str)
                or not probe_initial_sha256
                or candidate_metadata.get("probe_initial_sha256") != probe_initial_sha256
            ):
                return False
            fusion = candidate_metadata.get("fusion_state")
            if not isinstance(fusion, dict):
                return False
            coefficients = [fusion.get(f"coefficient_layer_{layer}") for layer in (2, 5, 8)]
            if not all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and abs(float(value)) <= 1 / 3 + 1e-7
                for value in coefficients
            ):
                return False
    except (KeyError, OSError, TypeError, ValueError):
        return False
    return True


def _expected_promotion_gate(
    row_by_key: dict[tuple[str, int | None], dict[str, Any]],
    failures: list[Any],
    result_row_count: int,
) -> dict[str, Any] | None:
    final = [row_by_key.get(("vjepa_final", seed)) for seed in (0, 1, 2)]
    candidate = [row_by_key.get(("vjepa_learned_fusion", seed)) for seed in (0, 1, 2)]
    if any(row is None for row in [*final, *candidate]):
        return None
    final_rows = [row for row in final if row is not None]
    candidate_rows = [row for row in candidate if row is not None]
    try:
        final_primary = sum(float(row["metrics"]["metric_abs_rel"]) for row in final_rows) / 3
        candidate_primary = sum(float(row["metrics"]["metric_abs_rel"]) for row in candidate_rows) / 3
        per_sequence: dict[str, dict[str, float | bool]] = {}
        sequence_condition = True
        for sequence_id in sorted(EXPECTED_TEST_SEQUENCES):
            final_value = sum(float(row["sequence_metrics"][sequence_id]["metric_abs_rel"]) for row in final_rows) / 3
            candidate_value = (
                sum(float(row["sequence_metrics"][sequence_id]["metric_abs_rel"]) for row in candidate_rows) / 3
            )
            relative_regression = (candidate_value - final_value) / max(final_value, 1e-12)
            passes = relative_regression <= 0.05
            sequence_condition &= passes
            per_sequence[sequence_id] = {
                "final_absrel": final_value,
                "candidate_absrel": candidate_value,
                "relative_regression": relative_regression,
                "passes_maximum_5pct_regression": passes,
            }
        final_latency = sum(float(row["runtime"]["total_ms_per_frame"]) for row in final_rows) / 3
        candidate_latency = sum(float(row["runtime"]["total_ms_per_frame"]) for row in candidate_rows) / 3

        def inference_memory(row: dict[str, Any]) -> float:
            runtime = row["runtime"]
            return float(runtime["peak_end_to_end_memory_gb"])

        final_memory = sum(inference_memory(row) for row in final_rows) / 3
        candidate_memory = sum(inference_memory(row) for row in candidate_rows) / 3
    except (KeyError, TypeError, ValueError):
        return None
    conditions = {
        "primary_macro_absrel_strictly_better": candidate_primary < final_primary,
        "no_sequence_regression_above_5pct": sequence_condition,
        "latency_at_most_1p10x_final": candidate_latency <= 1.10 * final_latency,
        "peak_inference_memory_at_most_1p10x_final": candidate_memory <= 1.10 * final_memory,
        "all_results_finite_valid_and_checkpointed": _results_integrity_valid(row_by_key, failures, result_row_count),
        "zero_failures": not failures,
    }
    promoted = all(conditions.values())
    return {
        "schema_version": "jepa4d-phase2c-promotion-v1",
        "decision": "promote_learned_fusion" if promoted else "retain_final_layer",
        "promoted": promoted,
        "conditions": conditions,
        "primary": {
            "final_macro_absrel": final_primary,
            "candidate_macro_absrel": candidate_primary,
            "relative_change": (candidate_primary - final_primary) / max(final_primary, 1e-12),
        },
        "per_sequence": per_sequence,
        "latency": {
            "final_ms_per_frame": final_latency,
            "candidate_ms_per_frame": candidate_latency,
            "ratio": candidate_latency / max(final_latency, 1e-12),
        },
        "peak_inference_memory": {
            "final_gib": final_memory,
            "candidate_gib": candidate_memory,
            "ratio": candidate_memory / max(final_memory, 1e-12),
        },
    }


def _compare_gate(actual: Any, expected: Any, location: str, errors: list[str]) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            errors.append(f"promotion gate {location} is not an object")
            return
        if set(actual) != set(expected):
            errors.append(
                f"promotion gate {location} keys differ: found={sorted(actual)}, expected={sorted(expected)}"
            )
            return
        for key in expected:
            _compare_gate(actual[key], expected[key], f"{location}.{key}", errors)
    elif isinstance(expected, bool):
        if actual is not expected:
            errors.append(f"promotion gate {location} is {actual}, expected {expected}")
    elif isinstance(expected, float):
        if (
            not isinstance(actual, (int, float))
            or not math.isfinite(float(actual))
            or not math.isclose(float(actual), expected, rel_tol=1e-9, abs_tol=1e-12)
        ):
            errors.append(f"promotion gate {location} is {actual}, expected {expected}")
    elif actual != expected:
        errors.append(f"promotion gate {location} is {actual}, expected {expected}")


def _validate_checkpoint(path: Path, variant: str, seed: int, errors: list[str]) -> None:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        errors.append(f"cannot load checkpoint {path.name}: {type(error).__name__}: {error}")
        return
    if payload.get("variant") != variant or payload.get("seed") != seed:
        errors.append(f"checkpoint identity mismatch: {path.name}")
    state = payload.get("state_dict")
    if not isinstance(state, dict) or not state:
        errors.append(f"checkpoint state is missing: {path.name}")
    else:
        for name, value in state.items():
            if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()):
                errors.append(f"checkpoint tensor is invalid: {path.name}:{name}")
    if variant == "vjepa_learned_fusion":
        if payload.get("model_type") != "ResidualFusionGeometryProbe":
            errors.append(f"learned-fusion model type is invalid: {path.name}")
        fusion = payload.get("fusion_state")
        if not isinstance(fusion, dict) or fusion.get("layer_order") != [2, 5, 8]:
            errors.append(f"learned-fusion state is incomplete: {path.name}")
        elif any(
            not math.isfinite(float(fusion.get(f"coefficient_layer_{layer}", float("nan"))))
            or abs(float(fusion.get(f"coefficient_layer_{layer}", float("nan")))) > 1 / 3 + 1e-7
            for layer in (2, 5, 8)
        ):
            errors.append(f"learned-fusion coefficients are invalid: {path.name}")


def main() -> None:
    args = parse_args()
    root = args.output.resolve()
    destination = args.report or root / "postflight-validation.json"
    errors: list[str] = []

    comparison_path = root / "comparison.json"
    comparison: dict[str, Any] = _read_json(comparison_path, errors, "comparison report", {})
    failures: list[Any] = _read_json(root / "failures.json", errors, "failures report", [])
    resolved: dict[str, Any] = _read_json(root / "resolved_config.json", errors, "resolved config", {})
    dataset: dict[str, Any] = _read_json(root / "dataset_fingerprint.json", errors, "dataset fingerprint", {})
    authorization: dict[str, Any] = _read_json(root / "formal_authorization.json", errors, "formal authorization", {})

    if comparison.get("schema_version") != "jepa4d-phase2c-cross-sequence-comparison-v1":
        errors.append("unexpected Phase 2c comparison schema")
    if resolved.get("protocol") != "phase2c-cross-sequence-v1":
        errors.append("resolved config does not select the Phase 2c protocol")
    if resolved.get("split_counts") != SPLIT_COUNTS:
        errors.append(f"resolved split counts are not {SPLIT_COUNTS}: {resolved.get('split_counts')}")
    if resolved.get("epochs") != EPOCHS or resolved.get("seeds") != [0, 1, 2]:
        errors.append("resolved epochs/seeds differ from the formal contract")
    if resolved.get("wandb", {}).get("mode") != "online" or not resolved.get("wandb", {}).get("enabled"):
        errors.append("resolved W&B configuration is not online and enabled")
    if authorization.get("schema_version") != "jepa4d-phase2c-authorization-v1":
        errors.append("unexpected formal authorization schema")
    if authorization.get("status") != "pass":
        errors.append("formal authorization does not pass")
    if authorization.get("protocol_sha256") != protocol_contract()["sha256"]:
        errors.append("formal authorization protocol differs from the postflight contract")
    if authorization.get("split_hash") != comparison.get("split_hash") or authorization.get(
        "split_hash"
    ) != resolved.get("split_hash"):
        errors.append("formal authorization split hash differs from the result split")
    resolved_authorization = resolved.get("authorization", {})
    authorization_path = root / "formal_authorization.json"
    if not isinstance(resolved_authorization, dict) or resolved_authorization.get("sha256") != (
        _sha256(authorization_path) if authorization_path.is_file() else None
    ):
        errors.append("resolved config is not bound to the persisted formal authorization")

    variants = comparison.get("variants", [])
    if len(variants) != 13:
        errors.append(f"expected 13 result rows (one teacher plus twelve probes), found {len(variants)}")
    actual_keys: list[tuple[str, int | None]] = []
    counts: Counter[str] = Counter()
    required_metrics = {
        "metric_abs_rel",
        "metric_rmse_m",
        "metric_delta_1",
        "aligned_abs_rel",
        "metric_abs_log_scale_error",
    }
    required_runtime = {
        "encoder_ms_per_frame",
        "head_ms_per_frame",
        "total_ms_per_frame",
        "peak_encoder_memory_gb",
        "peak_head_memory_gb",
    }
    row_by_key: dict[tuple[str, int | None], dict[str, Any]] = {}
    for index, row in enumerate(variants):
        variant = str(row.get("variant_id"))
        seed = row.get("seed")
        key = (variant, seed)
        actual_keys.append(key)
        counts[variant] += 1
        row_by_key[key] = row
        _check_finite(row.get("metrics", {}), f"variants[{index}].metrics", errors)
        _check_finite(row.get("runtime", {}), f"variants[{index}].runtime", errors)
        missing_metrics = required_metrics - set(row.get("metrics", {}))
        if missing_metrics:
            errors.append(f"variants[{index}] is missing metrics: {sorted(missing_metrics)}")
        missing_runtime = required_runtime - set(row.get("runtime", {}))
        if missing_runtime:
            errors.append(f"variants[{index}] is missing runtime: {sorted(missing_runtime)}")
        if variant in {"vjepa_final", "vjepa_multilayer", "vjepa_learned_fusion"}:
            runtime = row.get("runtime", {})
            end_to_end_required = {"end_to_end_ms_per_frame", "peak_end_to_end_memory_gb"}
            missing_end_to_end = end_to_end_required - set(runtime)
            if missing_end_to_end:
                errors.append(
                    f"variants[{index}] lacks co-resident end-to-end runtime fields: {sorted(missing_end_to_end)}"
                )
            elif not math.isclose(
                float(runtime["total_ms_per_frame"]),
                float(runtime["end_to_end_ms_per_frame"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                errors.append(f"variants[{index}] total latency is not its end-to-end profile")
            profile = row.get("model_metadata", {}).get("end_to_end_profile", {})
            if profile.get("profile") != "co-resident-batch1-encoder-normalization-fusion-probe-v1":
                errors.append(f"variants[{index}] has no formal co-resident runtime profile")
        sequence_metrics = row.get("sequence_metrics", {})
        if not isinstance(sequence_metrics, dict) or set(sequence_metrics) != EXPECTED_TEST_SEQUENCES:
            found = set(sequence_metrics) if isinstance(sequence_metrics, dict) else type(sequence_metrics).__name__
            errors.append(
                f"variants[{index}] held-out sequence metrics are {found}, expected {EXPECTED_TEST_SEQUENCES}"
            )
        else:
            sequence_abs_rel = [sequence_metrics[name].get("metric_abs_rel") for name in sorted(sequence_metrics)]
            if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in sequence_abs_rel):
                errors.append(f"variants[{index}] has invalid per-sequence metric_abs_rel values")
            else:
                macro_abs_rel = sum(float(value) for value in sequence_abs_rel) / 2
                recorded_abs_rel = row.get("metrics", {}).get("metric_abs_rel")
                if not isinstance(recorded_abs_rel, (int, float)) or not math.isclose(
                    float(recorded_abs_rel), macro_abs_rel, rel_tol=1e-7, abs_tol=1e-9
                ):
                    errors.append(
                        f"variants[{index}] metric_abs_rel {recorded_abs_rel} is not the arithmetic mean "
                        f"of held-out sequences {macro_abs_rel}"
                    )
            if variant != "vggt_teacher":
                for metric in ("raw_log_depth_nll", "calibrated_log_depth_nll"):
                    sequence_values = [sequence_metrics[name].get(metric) for name in sorted(sequence_metrics)]
                    recorded = row.get("metrics", {}).get(metric)
                    if not all(
                        isinstance(value, (int, float)) and math.isfinite(float(value)) for value in sequence_values
                    ):
                        errors.append(f"variants[{index}] has invalid per-sequence {metric} values")
                    elif not isinstance(recorded, (int, float)) or not math.isclose(
                        float(recorded),
                        sum(float(value) for value in sequence_values) / 2,
                        rel_tol=1e-7,
                        abs_tol=1e-9,
                    ):
                        errors.append(f"variants[{index}] {metric} is not the held-out sequence macro")
    if len(actual_keys) != len(set(actual_keys)):
        errors.append("duplicate variant/seed result rows")
    if set(actual_keys) != _expected_keys():
        errors.append(f"unexpected variant/seed set: {sorted(actual_keys, key=str)}")
    expected_counts = {name: len(seeds) for name, seeds in VARIANT_SEEDS.items()}
    if dict(counts) != expected_counts:
        errors.append(f"unexpected variant counts: {dict(counts)}")
    if comparison.get("failures"):
        errors.append(f"comparison contains {len(comparison['failures'])} failure(s)")
    if failures:
        errors.append(f"failures.json contains {len(failures)} failure(s)")
    _check_finite(comparison.get("aggregates", {}), "aggregates", errors)
    if not comparison.get("wandb_url"):
        errors.append("formal comparison does not contain an online W&B URL")

    promotion_gate: dict[str, Any] = _read_json(root / "promotion_gate.json", errors, "promotion gate", {})
    _check_finite(promotion_gate, "promotion_gate", errors)
    expected_promotion = _expected_promotion_gate(row_by_key, failures, len(variants))
    if expected_promotion is None:
        errors.append("cannot recompute the promotion decision from result rows")
    else:
        _compare_gate(promotion_gate, expected_promotion, "root", errors)

    end_to_end_profiles: list[dict[str, Any]] = _read_json(
        root / "end_to_end_profiles.json", errors, "end-to-end runtime profiles", []
    )
    _check_finite(end_to_end_profiles, "end_to_end_profiles", errors)
    profile_keys = {(row.get("variant"), row.get("seed")) for row in end_to_end_profiles}
    expected_profile_keys = {
        (variant, seed)
        for variant in ("vjepa_final", "vjepa_multilayer", "vjepa_learned_fusion")
        for seed in (0, 1, 2)
    }
    if len(end_to_end_profiles) != 9 or profile_keys != expected_profile_keys:
        errors.append(f"end-to-end runtime profile coverage is invalid: {sorted(profile_keys, key=str)}")
    profile_by_key = {(row.get("variant"), row.get("seed")): row for row in end_to_end_profiles}
    expected_profile_boundary = "preloaded RGBInputBatch before device transfer and model preprocessing"
    expected_profile_sample_ids = _expected_profile_sample_ids(dataset, errors)
    for key in expected_profile_keys:
        persisted_profile = profile_by_key.get(key)
        comparison_row = row_by_key.get(key, {})
        embedded_profile = comparison_row.get("model_metadata", {}).get("end_to_end_profile")
        if (
            not isinstance(persisted_profile, dict)
            or not isinstance(embedded_profile, dict)
            or persisted_profile != embedded_profile
        ):
            errors.append(f"persisted end-to-end profile differs from comparison metadata for {key}")
            continue
        expected_capture_layers = [] if key[0] == "vjepa_final" else [2, 5, 8]
        if persisted_profile.get("profile") != "co-resident-batch1-encoder-normalization-fusion-probe-v1":
            errors.append(f"end-to-end profile protocol is invalid for {key}")
        if persisted_profile.get("capture_layers") != expected_capture_layers:
            errors.append(f"end-to-end profile capture layers are invalid for {key}")
        if persisted_profile.get("input_boundary") != expected_profile_boundary:
            errors.append(f"end-to-end profile input boundary is invalid for {key}")
        if (
            persisted_profile.get("warmup_iterations") != 30
            or persisted_profile.get("measured_iterations_per_repetition") != 30
            or persisted_profile.get("repetitions") != 3
        ):
            errors.append(f"end-to-end profile iteration contract is invalid for {key}")
        repetitions = persisted_profile.get("repetition_ms_per_frame")
        if (
            not isinstance(repetitions, list)
            or len(repetitions) != 3
            or not all(_positive_finite_number(value) for value in repetitions)
        ):
            errors.append(f"end-to-end profile repetitions are invalid for {key}")
        median = persisted_profile.get("median_ms_per_frame")
        if not _positive_finite_number(median):
            errors.append(f"end-to-end profile median is invalid for {key}")
        elif (
            isinstance(repetitions, list)
            and len(repetitions) == 3
            and all(_positive_finite_number(value) for value in repetitions)
            and not math.isclose(
                float(median), sorted(float(value) for value in repetitions)[1], rel_tol=0.0, abs_tol=1e-12
            )
        ):
            errors.append(f"end-to-end profile median is not recomputed from repetitions for {key}")
        peak_memory = persisted_profile.get("peak_end_to_end_memory_gb")
        if not _positive_finite_number(peak_memory):
            errors.append(f"end-to-end profile peak memory is invalid for {key}")
        sample_ids = persisted_profile.get("sample_ids")
        if sample_ids != expected_profile_sample_ids:
            errors.append(f"end-to-end profile sample IDs differ from deterministic formal selection for {key}")
        runtime = comparison_row.get("runtime", {})
        runtime_latency = runtime.get("end_to_end_ms_per_frame")
        if (
            not _positive_finite_number(runtime_latency)
            or not _positive_finite_number(median)
            or not math.isclose(float(median), float(runtime_latency), rel_tol=0.0, abs_tol=1e-12)
        ):
            errors.append(f"persisted end-to-end latency differs from comparison runtime for {key}")
        runtime_memory = runtime.get("peak_end_to_end_memory_gb")
        if (
            not _positive_finite_number(runtime_memory)
            or not _positive_finite_number(peak_memory)
            or not math.isclose(float(peak_memory), float(runtime_memory), rel_tol=0.0, abs_tol=1e-12)
        ):
            errors.append(f"persisted end-to-end memory differs from comparison runtime for {key}")

    final_parameters = {
        seed: row_by_key.get(("vjepa_final", seed), {}).get("trainable_parameters") for seed in (0, 1, 2)
    }
    for seed in (0, 1, 2):
        candidate = row_by_key.get(("vjepa_learned_fusion", seed), {})
        reference_parameters = final_parameters[seed]
        expected_candidate_parameters = (
            int(reference_parameters) + 3 if isinstance(reference_parameters, (int, float)) else None
        )
        if candidate.get("trainable_parameters") != expected_candidate_parameters:
            errors.append(f"learned fusion seed {seed} does not add exactly three trainable parameters")
        metadata = candidate.get("model_metadata", {})
        if metadata.get("additional_trainable_parameters") != 3:
            errors.append(f"learned fusion seed {seed} metadata is incomplete")

    test_sequences = {
        str(row.get("sequence_id")) for row in dataset.get("sequences", []) if row.get("split") == "test"
    }
    if test_sequences != EXPECTED_TEST_SEQUENCES or dataset.get("split_counts") != SPLIT_COUNTS:
        errors.append(
            "dataset fingerprint does not prove the exact two held-out sequences and 128 test frames: "
            f"{test_sequences}"
        )

    sequence_rows: list[dict[str, Any]] = _read_json(
        root / "per_sequence_metrics.json", errors, "per-sequence metrics", []
    )
    sequence_groups: dict[tuple[str, int | None], set[str]] = defaultdict(set)
    persisted_sequence_rows: dict[tuple[str, int | None, str], dict[str, Any]] = {}
    for index, row in enumerate(sequence_rows):
        _check_finite(row, f"per_sequence_metrics[{index}]", errors)
        result_key = (str(row.get("variant")), row.get("seed"))
        sequence_id = str(row.get("sequence_id"))
        sequence_groups[result_key].add(sequence_id)
        persisted_key = (*result_key, sequence_id)
        if persisted_key in persisted_sequence_rows:
            errors.append(f"duplicate persisted per-sequence metric row: {persisted_key}")
        persisted_sequence_rows[persisted_key] = row
    if len(sequence_rows) != 26:
        errors.append(f"expected 26 per-sequence rows, found {len(sequence_rows)}")
    for key in _expected_keys():
        if sequence_groups[key] != test_sequences:
            errors.append(f"per-sequence coverage for {key} is {sequence_groups[key]}, expected {test_sequences}")
        comparison_row = row_by_key.get(key, {})
        embedded_sequence_metrics = comparison_row.get("sequence_metrics", {})
        for sequence_id in test_sequences:
            persisted = persisted_sequence_rows.get((*key, sequence_id))
            embedded = embedded_sequence_metrics.get(sequence_id)
            expected_persisted = (
                {
                    "variant": key[0],
                    "seed": key[1],
                    "sequence_id": sequence_id,
                    **embedded,
                }
                if isinstance(embedded, dict)
                else None
            )
            if persisted != expected_persisted:
                errors.append(
                    f"persisted per-sequence metrics differ from comparison sequence_metrics for {(*key, sequence_id)}"
                )

    frame_rows: list[dict[str, Any]] = _read_json(root / "per_frame_metrics.json", errors, "per-frame metrics", [])
    frame_groups: dict[tuple[str, int | None], Counter[str]] = defaultdict(Counter)
    for index, row in enumerate(frame_rows):
        _check_finite(row, f"per_frame_metrics[{index}]", errors)
        frame_groups[(str(row.get("variant")), row.get("seed"))][str(row.get("sequence_id"))] += 1
    if len(frame_rows) != 13 * SPLIT_COUNTS["test"]:
        errors.append(f"expected {13 * SPLIT_COUNTS['test']} per-frame rows, found {len(frame_rows)}")
    expected_frame_counts = Counter({sequence: 64 for sequence in test_sequences})
    for key in _expected_keys():
        if frame_groups[key] != expected_frame_counts:
            errors.append(f"per-frame coverage for {key} is {dict(frame_groups[key])}")

    checkpoints = sorted((root / "checkpoints").glob("*.pt"))
    if len(checkpoints) != 12:
        errors.append(f"expected 12 probe checkpoints, found {len(checkpoints)}")
    checkpoint_hashes = {path.name: _sha256(path) for path in checkpoints}
    for variant, seeds in VARIANT_SEEDS.items():
        if variant == "vggt_teacher":
            continue
        for seed_value in seeds:
            assert seed_value is not None
            path = root / "checkpoints" / f"{variant}-seed{seed_value}.pt"
            row = row_by_key.get((variant, seed_value), {})
            if not path.is_file():
                errors.append(f"missing checkpoint: {path.name}")
                continue
            if row.get("checkpoint_sha256") != _sha256(path):
                errors.append(f"comparison checkpoint hash mismatch: {path.name}")
            if not row.get("checkpoint") or Path(str(row["checkpoint"])).resolve() != path.resolve():
                errors.append(f"comparison checkpoint path mismatch: {path.name}")
            _validate_checkpoint(path, variant, seed_value, errors)

            history_path = root / "histories" / f"{variant}-seed{seed_value}.jsonl"
            if not history_path.is_file():
                errors.append(f"missing training history: {history_path.name}")
            else:
                try:
                    history = [json.loads(line) for line in history_path.read_text().splitlines() if line]
                except json.JSONDecodeError as error:
                    errors.append(f"invalid training history {history_path.name}: {error}")
                    history = []
                if len(history) != EPOCHS or [row.get("epoch") for row in history] != list(range(EPOCHS)):
                    errors.append(f"training history is not exactly {EPOCHS} epochs: {history_path.name}")
                _check_finite(history, f"history.{history_path.name}", errors)
                if variant == "vjepa_learned_fusion" and any(
                    "gate_gradient_norm" not in history_row or "coefficient_layer_2" not in history_row
                    for history_row in history
                ):
                    errors.append(f"fusion history lacks gate diagnostics: {history_path.name}")

    recorded_artifacts = comparison.get("artifacts", {})
    for name in protocol_contract()["normalization_artifacts"]:
        normalization = root / name
        if not normalization.is_file():
            errors.append(f"missing normalization artifact: {name}")
        elif recorded_artifacts.get(name) != _sha256(normalization):
            errors.append(f"normalization hash is missing or mismatched: {name}")

    completion: dict[str, Any] = _read_json(root / "completion_gate.json", errors, "completion gate", {})
    if completion.get("status") != "success":
        errors.append(f"runner completion gate did not pass: {completion}")
    if (
        completion.get("result_rows") != 13
        or completion.get("probe_checkpoints") != 12
        or completion.get("seed_failures") != 0
    ):
        errors.append(f"runner completion counts are invalid: {completion}")
    if expected_promotion is not None and completion.get("promotion_decision") != expected_promotion["decision"]:
        errors.append(
            f"completion promotion decision {completion.get('promotion_decision')} differs from recomputed "
            f"decision {expected_promotion['decision']}"
        )
    if (root / "run_failure.json").exists():
        errors.append("run_failure.json exists")

    html_report = root / "geometry_student_report.html"
    if not html_report.is_file() or html_report.stat().st_size == 0:
        errors.append("missing or empty geometry_student_report.html")
    else:
        html = html_report.read_text(errors="replace")
        if "Plotly.newPlot" not in html:
            errors.append("geometry report does not contain interactive Plotly visualizations")
        if re.search(r"<script\b[^>]*\bsrc\s*=", html, flags=re.IGNORECASE):
            errors.append("geometry report loads an external script instead of being self-contained")

    artifact_manifest_path = root / "artifact_manifest.json"
    artifact_manifest: dict[str, Any] = _read_json(artifact_manifest_path, errors, "artifact manifest", {})
    if artifact_manifest:
        excluded = {"artifact_manifest.json", "wandb_artifact_receipt.json"}
        actual_files = {
            str(path.relative_to(root))
            for path in root.rglob("*")
            if path.is_file() and str(path.relative_to(root)) not in excluded
        }
        if set(artifact_manifest) != actual_files:
            errors.append(
                "artifact manifest file set differs from disk: "
                f"missing={sorted(actual_files - set(artifact_manifest))}, "
                f"extra={sorted(set(artifact_manifest) - actual_files)}"
            )
        for relative, entry in artifact_manifest.items():
            relative_path = Path(relative)
            if relative_path.is_absolute() or ".." in relative_path.parts or not isinstance(entry, dict):
                errors.append(f"unsafe or invalid artifact manifest entry: {relative}")
                continue
            artifact = root / relative_path
            if artifact.is_file() and (
                entry.get("bytes") != artifact.stat().st_size or entry.get("sha256") != _sha256(artifact)
            ):
                errors.append(f"artifact manifest identity mismatch: {relative}")

    if args.require_wandb:
        receipt: dict[str, Any] = _read_json(root / "wandb_artifact_receipt.json", errors, "W&B artifact receipt", {})
        required = (
            "run_id",
            "run_url",
            "artifact_name",
            "artifact_version",
            "artifact_digest",
            "artifact_manifest_sha256",
        )
        if receipt.get("schema_version") != "jepa4d-phase2c-wandb-artifact-v1":
            errors.append("unexpected Phase 2c W&B artifact receipt schema")
        if receipt.get("status") != "success" or receipt.get("mode") != "online":
            errors.append("W&B artifact receipt is not an online success")
        if any(not receipt.get(key) for key in required):
            errors.append("W&B artifact receipt is incomplete")
        if receipt.get("run_url") != comparison.get("wandb_url"):
            errors.append("W&B receipt run URL differs from comparison")
        if artifact_manifest_path.is_file() and receipt.get("artifact_manifest_sha256") != _sha256(
            artifact_manifest_path
        ):
            errors.append("W&B receipt refers to a different artifact manifest")

    report = {
        "schema_version": "jepa4d-phase2c-postflight-v1",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "pass" if not errors else "fail",
        "output": str(root),
        "protocol_sha256": protocol_contract()["sha256"],
        "comparison_sha256": _sha256(comparison_path) if comparison_path.is_file() else None,
        "checkpoint_sha256": checkpoint_hashes,
        "result_rows": len(variants),
        "variant_counts": dict(counts),
        "per_sequence_rows": len(sequence_rows),
        "per_frame_rows": len(frame_rows),
        "failures_count": len(failures),
        "errors": errors,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(destination)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()
