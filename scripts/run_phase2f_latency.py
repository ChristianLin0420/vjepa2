#!/usr/bin/env python3
"""Run one independent preregistered Phase 2f latency allocation."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from collections import defaultdict
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import torch
from PIL import Image, ImageDraw

from jepa4d.data.rgb_input import collate_rgb_inputs, from_view_sequences
from jepa4d.evaluation.phase2f_data_cache import (
    rotation_indices,
    validate_sun_development_feature_cache,
    validate_sun_development_input_cache,
)
from jepa4d.evaluation.phase2f_metrics import (
    atomic_json,
    cuda_hardware_identity,
    file_identity,
    require_finite_tree,
    self_contained_html,
)
from jepa4d.models.phase2f_scale_geometry import PHASE2F_COMPONENTS, Phase2fScaleGeometryProbe
from jepa4d.models.vjepa21_adapter import VJEPA21FeatureExtractor
from jepa4d.training.phase2f_training import phase2f_arm_configs
from scripts.run_phase2f_training import _finish_wandb, _normalize, _validate_provenance, fit_rotation_normalization

ARMS = ("M0", "M1", "M2", "M3")
LATENCY_REPLICA_SCHEMA = "jepa4d-phase2f-latency-replica-v1"
INITIALIZATION_SEED = 260629
WARMUPS = 30
BLOCKS = 30
ITERATIONS = 100
EXPECTED_GPU = "NVIDIA A100-SXM4-80GB"
_T = TypeVar("_T")


def _load_cache(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    try:
        value = torch.load(resolved, map_location="cpu", weights_only=True, mmap=True)
    except (RuntimeError, TypeError):
        value = torch.load(resolved, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise TypeError(f"cache must contain a mapping: {path}")
    return value


def _time_cuda_block(operation: Callable[[int], Any], iterations: int) -> tuple[float, float]:
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    started = time.perf_counter_ns()
    start_event.record()
    for index in range(iterations):
        operation(index)
    end_event.record()
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter_ns() - started) / 1_000_000 / iterations
    cuda_ms = float(start_event.elapsed_time(end_event)) / iterations
    if not np.isfinite((wall_ms, cuda_ms)).all() or wall_ms <= 0 or cuda_ms <= 0:
        raise RuntimeError("CUDA latency measurement was non-finite or non-positive")
    return wall_ms, cuda_ms


class _ComponentHook:
    def __init__(self) -> None:
        self.wall_ms: defaultdict[str, list[float]] = defaultdict(list)
        self.cuda_ms: defaultdict[str, list[float]] = defaultdict(list)

    def __call__(self, name: str, operation: Callable[[], _T]) -> _T:
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        started = time.perf_counter_ns()
        start.record()
        result = operation()
        end.record()
        torch.cuda.synchronize()
        self.wall_ms[name].append((time.perf_counter_ns() - started) / 1_000_000)
        self.cuda_ms[name].append(float(start.elapsed_time(end)))
        return result


def _single_image_batch(image: torch.Tensor) -> Any:
    return collate_rgb_inputs([from_view_sequences([[image]])])


def benchmark_latency(
    input_cache: Mapping[str, Any],
    feature_cache: Mapping[str, Any],
    *,
    extractor: VJEPA21FeatureExtractor,
    device: torch.device,
    replica: int,
    block_logger: Callable[[Mapping[str, Any]], None] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Benchmark M0-M3 with one fixed batch and randomized paired schedules."""

    validate_sun_development_input_cache(input_cache)
    validate_sun_development_feature_cache(feature_cache)
    if input_cache["samples"] != feature_cache["samples"]:
        raise ValueError("latency input/feature cache row identities differ")
    indices = rotation_indices(feature_cache, "R0")
    normalization = fit_rotation_normalization(feature_cache["ordinary_features"], indices["train"])
    sample_ids = input_cache["samples"]["sample_ids"]
    chosen = min(
        range(len(sample_ids)), key=lambda index: __import__("hashlib").sha256(sample_ids[index].encode()).hexdigest()
    )
    raw_feature = feature_cache["ordinary_features"][chosen, 0:1]
    feature = _normalize(raw_feature, normalization, device)
    camera = input_cache["ordinary_inputs"]["intrinsics_384"][chosen, 0:1].to(device).float()
    image = input_cache["ordinary_inputs"]["images_384_uint8"][chosen, 0].float().div(255)
    image_batch = _single_image_batch(image)

    models: dict[str, Phase2fScaleGeometryProbe] = {}
    for registry_arm, config in phase2f_arm_configs(768).items():
        torch.manual_seed(INITIALIZATION_SEED)
        torch.cuda.manual_seed_all(INITIALIZATION_SEED)
        models[registry_arm] = Phase2fScaleGeometryProbe(config).to(device).eval()

    def head(arm: str, timing_hook: Any | None = None) -> Any:
        model = models[arm]
        use_camera = arm in {"M2", "M3"}
        return model(
            feature,
            intrinsics=camera if use_camera else None,
            intrinsics_image_size=(384, 384) if use_camera else None,
            timing_hook=timing_hook,
        )

    def encode() -> torch.Tensor:
        bundle = extractor(image_batch)
        grid = bundle.dense_tokens[:, 0, 0].reshape(1, 24, 24, 768).permute(0, 3, 1, 2)
        return grid.float()

    raw_feature_device = raw_feature.to(device=device, dtype=torch.float32)

    def normalize_only() -> torch.Tensor:
        return _normalize(raw_feature_device, normalization, device)

    def end_to_end(arm: str) -> Any:
        grid = _normalize(encode(), normalization, device)
        model = models[arm]
        use_camera = arm in {"M2", "M3"}
        return model(
            grid,
            intrinsics=camera if use_camera else None,
            intrinsics_image_size=(384, 384) if use_camera else None,
        )

    def head_callback(selected: str) -> Callable[[int], Any]:
        return lambda _index: head(selected)

    def end_to_end_callback(selected: str) -> Callable[[int], Any]:
        return lambda _index: end_to_end(selected)

    def operation_callback(operation: Callable[[], Any]) -> Callable[[int], Any]:
        return lambda _index: operation()

    with torch.inference_mode():
        for arm in ARMS:
            for _ in range(WARMUPS):
                head(arm)
                end_to_end(arm)
        for _ in range(WARMUPS):
            encode()
            normalize_only()
    torch.cuda.synchronize()

    rng = random.Random(INITIALIZATION_SEED + replica * 1009)
    rows: list[dict[str, Any]] = []
    arm_values: dict[str, dict[str, Any]] = {
        arm: {
            "parameter_count": models[arm].trainable_parameter_count,
            "parameter_counts": models[arm].parameter_counts(),
            "complete_head_wall_ms": [],
            "complete_head_cuda_ms": [],
            "encoder_plus_head_wall_ms": [],
            "encoder_plus_head_cuda_ms": [],
            "components": {},
            "peak_allocation_bytes": 0,
        }
        for arm in ARMS
    }
    with torch.inference_mode():
        for block in range(BLOCKS):
            schedule = list(ARMS)
            rng.shuffle(schedule)
            for position, arm in enumerate(schedule):
                torch.cuda.reset_peak_memory_stats(device)
                wall, cuda = _time_cuda_block(head_callback(arm), ITERATIONS)
                arm_values[arm]["complete_head_wall_ms"].append(wall)
                arm_values[arm]["complete_head_cuda_ms"].append(cuda)
                arm_values[arm]["peak_allocation_bytes"] = max(
                    arm_values[arm]["peak_allocation_bytes"], torch.cuda.max_memory_allocated(device)
                )
                row = {
                    "path": "complete_head",
                    "block": block,
                    "position": position,
                    "arm": arm,
                    "iterations": ITERATIONS,
                    "wall_ms_per_iteration": wall,
                    "cuda_ms_per_iteration": cuda,
                }
                rows.append(row)
                if block_logger is not None:
                    block_logger(row)
        shared_components: dict[str, Any] = {
            "vjepa_encoding": {"wall_ms": [], "cuda_ms": []},
            "feature_normalization": {"wall_ms": [], "cuda_ms": []},
        }
        for block in range(BLOCKS):
            order = ["vjepa_encoding", "feature_normalization"]
            rng.shuffle(order)
            for position, name in enumerate(order):
                operation: Callable[[], Any] = encode if name == "vjepa_encoding" else normalize_only
                wall, cuda = _time_cuda_block(operation_callback(operation), ITERATIONS)
                shared_components[name]["wall_ms"].append(wall)
                shared_components[name]["cuda_ms"].append(cuda)
                row = {
                    "path": name,
                    "block": block,
                    "position": position,
                    "arm": "shared",
                    "iterations": ITERATIONS,
                    "wall_ms_per_iteration": wall,
                    "cuda_ms_per_iteration": cuda,
                }
                rows.append(row)
                if block_logger is not None:
                    block_logger(row)
        for block in range(BLOCKS):
            schedule = list(ARMS)
            rng.shuffle(schedule)
            for position, arm in enumerate(schedule):
                wall, cuda = _time_cuda_block(end_to_end_callback(arm), ITERATIONS)
                arm_values[arm]["encoder_plus_head_wall_ms"].append(wall)
                arm_values[arm]["encoder_plus_head_cuda_ms"].append(cuda)
                row = {
                    "path": "encoder_plus_head",
                    "block": block,
                    "position": position,
                    "arm": arm,
                    "iterations": ITERATIONS,
                    "wall_ms_per_iteration": wall,
                    "cuda_ms_per_iteration": cuda,
                }
                rows.append(row)
                if block_logger is not None:
                    block_logger(row)
        for arm in ARMS:
            hook = _ComponentHook()
            for _ in range(BLOCKS):
                for _ in range(ITERATIONS):
                    head(arm, hook)
            components: dict[str, Any] = {}
            for name in PHASE2F_COMPONENTS:
                wall_values = hook.wall_ms.get(name, [])
                cuda_values = hook.cuda_ms.get(name, [])
                components[name] = {
                    "calls": len(wall_values),
                    "wall_ms_mean": 0.0 if not wall_values else float(np.mean(wall_values)),
                    "cuda_ms_mean": 0.0 if not cuda_values else float(np.mean(cuda_values)),
                    "applicable": bool(wall_values),
                }
            arm_values[arm]["components"] = components
    return arm_values, shared_components, rows


def _plot(path: Path, arms: Mapping[str, Mapping[str, Any]]) -> None:
    image = Image.new("RGB", (980, 500), "#f7f8fb")
    draw = ImageDraw.Draw(image)
    draw.text((30, 20), "Phase 2f complete-head latency by arm", fill="#17202e")
    means = {arm: float(np.mean(arms[arm]["complete_head_wall_ms"])) for arm in ARMS}
    maximum = max(means.values())
    baseline = means["M0"]
    for index, arm in enumerate(ARMS):
        top = 90 + index * 90
        width = int(720 * means[arm] / maximum)
        color = "#238636" if means[arm] / baseline <= 1.10 else "#cf4a4a"
        draw.text((30, top), arm, fill="#17202e")
        draw.rectangle((100, top, 100 + width, top + 30), fill=color)
        draw.text((110, top + 8), f"{means[arm]:.4f} ms; {means[arm] / baseline:.3f}x", fill="white")
    image.save(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replica", type=int, choices=range(12), required=True)
    parser.add_argument("--input-cache", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--vjepa-checkpoint", type=Path, required=True)
    parser.add_argument("--vjepa-implementation", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wandb-entity", default="crlc112358")
    parser.add_argument("--wandb-project", default="jepa4d-worldmodel")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Phase 2f experiment CLIs may run only inside a Slurm job")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Phase 2f latency must run on a Slurm CUDA allocation")
    gpu_name = torch.cuda.get_device_name(device)
    if gpu_name != EXPECTED_GPU:
        raise RuntimeError(f"Phase 2f latency requires {EXPECTED_GPU}; allocated {gpu_name}")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    provenance = _validate_provenance(args.provenance)
    execution_id = str(provenance["execution_id"])
    slurm_id = str(provenance["slurm"].get("job_id", os.environ.get("SLURM_JOB_ID", "unknown")))
    import wandb

    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        group=f"phase2f-{execution_id}",
        job_type="latency-replica",
        name=f"{execution_id}-latency-r{args.replica:02d}-{slurm_id}",
        mode="online",
        reinit=True,
        config={
            "replica": args.replica,
            "initialization_seed": INITIALIZATION_SEED,
            "warmups_per_path": WARMUPS,
            "blocks": BLOCKS,
            "iterations_per_block": ITERATIONS,
            "git_commit": provenance["git_commit"],
        },
    )
    if run is None or run.offline:
        raise RuntimeError("Phase 2f latency requires online W&B")
    input_cache = _load_cache(args.input_cache)
    feature_cache = _load_cache(args.feature_cache)
    extractor = VJEPA21FeatureExtractor(
        checkpoint=args.vjepa_checkpoint,
        implementation_path=args.vjepa_implementation,
        backend="hf_compat",
        device=device,
        capture_layers=(),
    )
    started = time.perf_counter()

    def log_block(row: Mapping[str, Any]) -> None:
        run.log(
            {
                f"latency/{row['path']}/{row['arm']}/wall_ms": row["wall_ms_per_iteration"],
                f"latency/{row['path']}/{row['arm']}/cuda_ms": row["cuda_ms_per_iteration"],
                "schedule/block": row["block"],
                "schedule/position": row["position"],
            }
        )

    arms, shared_components, rows = benchmark_latency(
        input_cache,
        feature_cache,
        extractor=extractor,
        device=device,
        replica=args.replica,
        block_logger=log_block,
    )
    with (output / "schedule.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        output / "latency.npz",
        arms=np.asarray(ARMS),
        complete_head_wall_ms=np.asarray([arms[arm]["complete_head_wall_ms"] for arm in ARMS]),
        complete_head_cuda_ms=np.asarray([arms[arm]["complete_head_cuda_ms"] for arm in ARMS]),
        encoder_plus_head_wall_ms=np.asarray([arms[arm]["encoder_plus_head_wall_ms"] for arm in ARMS]),
    )
    figure = output / "latency.png"
    _plot(figure, arms)
    report = output / "report.html"
    report.write_text(
        self_contained_html(
            f"Phase 2f latency replica {args.replica}",
            {
                "GPU": gpu_name,
                "blocks": BLOCKS,
                "iterations per block": ITERATIONS,
                "M3/M0 head ratio": np.mean(arms["M3"]["complete_head_wall_ms"])
                / np.mean(arms["M0"]["complete_head_wall_ms"]),
            },
            images=(("Complete-head wall latency", figure),),
            claim_boundary="Development operation-graph qualification; model weights are untrained seed-260629 initializations.",
        ),
        encoding="utf-8",
    )
    summary_path = atomic_json(output / "latency_summary.json", {"arms": arms, "shared_components": shared_components})
    run.log(
        {
            "terminal/status": "success",
            "terminal/M1_to_M0_head_ratio": np.mean(arms["M1"]["complete_head_wall_ms"])
            / np.mean(arms["M0"]["complete_head_wall_ms"]),
        }
    )
    wandb_receipt = _finish_wandb(
        run,
        artifact_name=f"phase2f-latency-{execution_id}-r{args.replica:02d}",
        job_type="latency-replica",
        files=(output / "schedule.csv", output / "latency.npz", summary_path, figure, report),
    )
    receipt = {
        "schema_version": LATENCY_REPLICA_SCHEMA,
        "status": "success",
        "replica": args.replica,
        "config": {
            "initialization_seed": INITIALIZATION_SEED,
            "warmups_per_path": WARMUPS,
            "blocks": BLOCKS,
            "iterations_per_block": ITERATIONS,
            "batch_size": 1,
        },
        "hardware": {
            **cuda_hardware_identity(device),
            "peak_allocation_bytes": max(int(arms[arm]["peak_allocation_bytes"]) for arm in ARMS),
        },
        "arms": arms,
        "shared_components": shared_components,
        "cache_inputs": {
            "input": file_identity(args.input_cache),
            "feature": file_identity(args.feature_cache),
            "vjepa_checkpoint": file_identity(args.vjepa_checkpoint)
            if args.vjepa_checkpoint.is_file()
            else {"path": str(args.vjepa_checkpoint.resolve(strict=True))},
        },
        "elapsed_seconds": time.perf_counter() - started,
        "wandb": wandb_receipt,
        "execution_provenance": provenance,
    }
    require_finite_tree(receipt, "latency_receipt")
    atomic_json(output / "wandb_receipt.json", wandb_receipt)
    atomic_json(output / "latency_receipt.json", receipt)
    (output / "SUCCESS").write_text("success\n", encoding="utf-8")
    print(json.dumps({"status": "success", "replica": args.replica, "gpu": gpu_name}, sort_keys=True))


if __name__ == "__main__":
    main()
