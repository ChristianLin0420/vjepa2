"""Strict dataset, asset, real-model, optimizer, report, and W&B gate for Phase 2b."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from jepa4d.benchmarks.geometry.tum_rgbd import load_tum_indices, validate_archive
from jepa4d.models.geometry_belief import GeometryBeliefHead
from jepa4d.models.geometry_student import DenseGeometryProbe, geometry_probe_loss
from jepa4d.models.vjepa21_adapter import VJEPA21FeatureExtractor
from jepa4d.visualization.geometry_student_report import build_geometry_student_report
from scripts.run_phase2b_geometry_distillation import (
    _configure_determinism,
    _dataset_fingerprint,
    _evaluate_depths,
    _fit_metric_scale,
    _normalize,
    _single_image_batch,
    _targets,
    _valid,
)
from slurm.phase2b_gate import (
    asset_inventory,
    environment_fingerprint,
    repository_fingerprint,
    sha256,
)


def _tensor_summary(value: torch.Tensor) -> dict[str, Any]:
    detached = value.detach()
    finite = torch.isfinite(detached)
    finite_values = detached[finite].float()
    return {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "finite_fraction": float(finite.float().mean().item()),
        "mean": float(finite_values.mean().item()) if finite_values.numel() else None,
        "std": float(finite_values.std().item()) if finite_values.numel() > 1 else 0.0,
        "min": float(finite_values.min().item()) if finite_values.numel() else None,
        "max": float(finite_values.max().item()) if finite_values.numel() else None,
    }


def _require_finite(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise RuntimeError(f"{name} contains non-finite values")


def _assert_close(
    name: str,
    batched: torch.Tensor,
    separate: torch.Tensor,
    *,
    rtol: float,
    atol: float,
) -> dict[str, float]:
    if batched.shape != separate.shape:
        raise RuntimeError(f"{name} chunking changed shape: {tuple(batched.shape)} != {tuple(separate.shape)}")
    difference = (batched.float() - separate.float()).abs()
    maximum = float(difference.max().item())
    mean = float(difference.mean().item())
    rmse = float(difference.square().mean().sqrt().item())
    reference_rms = float(separate.float().square().mean().sqrt().item())
    relative_rmse = rmse / max(reference_rms, 1e-12)
    cosine_similarity = float(
        F.cosine_similarity(batched.float().reshape(1, -1), separate.float().reshape(1, -1)).item()
    )
    statistics = {
        "max_abs": maximum,
        "mean_abs": mean,
        "rmse": rmse,
        "reference_rms": reference_rms,
        "relative_rmse": relative_rmse,
        "cosine_similarity": cosine_similarity,
        "rtol": rtol,
        "atol": atol,
    }
    try:
        torch.testing.assert_close(batched.float(), separate.float(), rtol=rtol, atol=atol)
    except AssertionError as error:
        raise RuntimeError(f"{name} changes with chunk size: statistics={statistics}: {error}") from error
    return statistics


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _wandb_online_probe(args: argparse.Namespace, report: dict[str, Any], artifact_dir: Path) -> dict[str, Any]:
    import wandb

    run = None
    finished = False
    try:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            job_type="phase2b-preflight",
            mode="online",
            tags=["phase-2b", "preflight", "real-model-smoke", "slurm", "cuda"],
            config={
                "schema_version": report["schema_version"],
                "slurm_job_id": report["slurm_job_id"],
                "dataset_id": report["dataset"]["dataset_id"],
                "split_indices_sha256": report["dataset"]["split_indices_sha256"],
                "repository_sha256": report["authorization"]["repository"]["sha256"],
                "environment_sha256": report["authorization"]["environment"]["sha256"],
                "vjepa_sha256": report["assets"]["vjepa_checkpoint"]["sha256"],
                "vggt_sha256": report["assets"]["vggt_checkpoint"]["sha256"],
            },
        )
        if run.offline:
            raise RuntimeError("W&B preflight unexpectedly initialized offline")
        run.log(
            {
                "preflight/vjepa_chunk_max_abs": report["vjepa_smoke"]["chunk_invariance"]["dense"]["max_abs"],
                "preflight/vggt_chunk_max_abs": report["vggt_smoke"]["chunk_invariance"]["depth"]["max_abs"],
                "preflight/probe_loss": report["probe_smoke"]["loss"],
                "preflight/cuda_peak_memory_gb": torch.cuda.max_memory_allocated() / 1024**3,
            }
        )
        artifact = wandb.Artifact(f"{run.id}-phase2b-preflight", type="preflight-validation")
        artifact.add_dir(str(artifact_dir), name="preflight")
        logged_artifact = run.log_artifact(artifact)
        # Artifact uploads are asynchronous. Waiting here makes a successful
        # preflight evidence that the backend accepted both the run and files.
        logged_artifact.wait(timeout=900)
        result = {
            "mode": "online",
            "run_id": run.id,
            "run_url": run.url,
            "run_path": run.path,
            "artifact_name": logged_artifact.name,
            "artifact_version": logged_artifact.version,
            "artifact_digest": logged_artifact.digest,
        }
        run.summary.update({"preflight_status": "pass", **result})
        run.finish(exit_code=0)
        finished = True
        return result
    except Exception:
        if run is not None and not finished:
            try:
                run.summary.update({"preflight_status": "fail"})
                run.finish(exit_code=1)
            except Exception:
                pass
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--vjepa-checkpoint", type=Path, required=True)
    parser.add_argument("--vjepa-implementation", type=Path, required=True)
    parser.add_argument("--vggt-checkpoint", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--test-report", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wandb-project", default="jepa4d-worldmodel")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-name", default="phase2b-preflight")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _authorization(args: argparse.Namespace) -> dict[str, Any]:
    repository = repository_fingerprint(args.repo_root)
    environment = environment_fingerprint()
    receipt = json.loads(args.test_report.read_text())
    if receipt.get("schema_version") != "jepa4d-phase2b-tests-v1" or receipt.get("status") != "pass":
        raise RuntimeError("Phase 2b test receipt is missing or does not pass")
    if not receipt.get("slurm_job_id"):
        raise RuntimeError("Phase 2b tests must have run in a Slurm allocation")
    if receipt.get("repository") != repository:
        raise RuntimeError("repository content differs from the passing Slurm test job")
    if receipt.get("environment") != environment:
        raise RuntimeError("Python environment differs from the passing Slurm test job")
    cuda = receipt.get("cuda_report", {})
    cuda_path = Path(str(cuda.get("path", "")))
    if not cuda_path.is_file() or sha256(cuda_path) != cuda.get("sha256"):
        raise RuntimeError("Slurm test CUDA report is missing or has changed")
    if cuda.get("summary", {}).get("status") != "pass":
        raise RuntimeError("Slurm test CUDA report does not pass")
    return {
        "repository": repository,
        "environment": environment,
        "test_receipt": {
            "path": str(args.test_report.resolve()),
            "sha256": sha256(args.test_report),
            "slurm_job_id": receipt["slurm_job_id"],
            "cuda_report_sha256": cuda["sha256"],
        },
    }


def run(args: argparse.Namespace, report: dict[str, Any]) -> None:
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("the Phase 2b preflight requires an available CUDA device")
    device = torch.device(args.device)
    _configure_determinism()
    torch.cuda.set_device(device)
    properties = torch.cuda.get_device_properties(device)
    report["gpu"] = {
        "name": properties.name,
        "total_memory_bytes": properties.total_memory,
        "compute_capability": [properties.major, properties.minor],
    }

    stage_started = time.perf_counter()
    manifest = validate_archive(args.archive, args.manifest)
    split_indices = {
        split: [int(value) for value in manifest[f"{split}_indices"]] for split in ("train", "validation", "test")
    }
    flattened = [value for indices in split_indices.values() for value in indices]
    if len(flattened) != len(set(flattened)):
        raise RuntimeError("dataset split indices overlap")
    samples = {split: load_tum_indices(args.dataset_root, indices) for split, indices in split_indices.items()}
    counts = {key: len(value) for key, value in samples.items()}
    if counts != {"train": 64, "validation": 16, "test": 8}:
        raise RuntimeError(f"formal split must be 64/16/8, found {counts}")
    extraction_fingerprint = _dataset_fingerprint(args.dataset_root, samples, args.archive)
    report["dataset"] = {
        "root": str(args.dataset_root.resolve()),
        "archive": str(args.archive.resolve()),
        "archive_bytes": args.archive.stat().st_size,
        "archive_sha256": manifest["archive"]["sha256"],
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "dataset_id": manifest["dataset_id"],
        "version": str(manifest["version"]),
        "split_counts": counts,
        "split_indices_sha256": hashlib.sha256(
            json.dumps(split_indices, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "extraction_fingerprint": extraction_fingerprint,
    }
    report["timings_seconds"]["dataset_validation"] = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    report["assets"] = {
        "vjepa_checkpoint": asset_inventory(args.vjepa_checkpoint),
        "vjepa_implementation": asset_inventory(args.vjepa_implementation),
        "vggt_checkpoint": asset_inventory(args.vggt_checkpoint),
        "hash_mode": "full",
    }
    report["timings_seconds"]["asset_inventory"] = time.perf_counter() - stage_started

    smoke_samples = samples["train"][:8]
    batch = _single_image_batch(smoke_samples)
    if tuple(batch.images.shape[:3]) != (8, 1, 1):
        raise RuntimeError(f"formal batching must be B=8,V=1,T=1, found {tuple(batch.images.shape[:3])}")
    target_24 = _targets(smoke_samples, (24, 24))

    torch.cuda.reset_peak_memory_stats(device)
    stage_started = time.perf_counter()
    extractor = VJEPA21FeatureExtractor(
        checkpoint=args.vjepa_checkpoint,
        implementation_path=args.vjepa_implementation,
        backend="hf_compat",
        device=args.device,
    )
    load_seconds = time.perf_counter() - stage_started
    forward_started = time.perf_counter()
    with torch.inference_mode():
        batched_bundle = extractor(batch)
        separate_bundles = [extractor(_single_image_batch([sample])) for sample in smoke_samples]
    torch.cuda.synchronize(device)
    forward_seconds = time.perf_counter() - forward_started
    batched_dense = batched_bundle.dense_tokens[:, 0, 0]
    separate_dense = torch.cat([value.dense_tokens[:, 0, 0] for value in separate_bundles])
    _require_finite("V-JEPA dense tokens", batched_dense)
    if sorted(batched_bundle.layer_tokens) != [2, 5, 8, 11]:
        raise RuntimeError(f"unexpected V-JEPA intermediate layers: {sorted(batched_bundle.layer_tokens)}")
    layer_invariance = {}
    for layer, tokens in batched_bundle.layer_tokens.items():
        batched_layer = tokens[:, 0, 0]
        separate_layer = torch.cat([value.layer_tokens[layer][:, 0, 0] for value in separate_bundles])
        _require_finite(f"V-JEPA layer {layer}", batched_layer)
        layer_invariance[str(layer)] = _assert_close(
            f"V-JEPA layer {layer}", batched_layer, separate_layer, rtol=1e-2, atol=3e-3
        )
    probe_features = batched_dense.reshape(8, 24, 24, -1).permute(0, 3, 1, 2).contiguous().cpu()
    report["vjepa_smoke"] = {
        "sample_ids": [sample.sample_id for sample in smoke_samples],
        "input_shape_b_v_t_c_h_w": list(batch.images.shape),
        "model_config": extractor.model_config,
        "load_seconds": load_seconds,
        "forward_seconds_batched_plus_separate": forward_seconds,
        "dense_tokens": _tensor_summary(batched_dense),
        "global_tokens": _tensor_summary(batched_bundle.global_tokens),
        "layer_tokens": {str(key): _tensor_summary(value) for key, value in batched_bundle.layer_tokens.items()},
        "chunk_invariance": {
            "dense": _assert_close("V-JEPA dense tokens", batched_dense, separate_dense, rtol=1e-2, atol=3e-3),
            "layers": layer_invariance,
            "compared_chunk_sizes": [1, 8],
        },
        "peak_memory_bytes": torch.cuda.max_memory_allocated(device),
    }
    del batched_bundle, separate_bundles, batched_dense, separate_dense, extractor
    gc.collect()
    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats(device)
    stage_started = time.perf_counter()
    teacher = GeometryBeliefHead(
        backend="vggt", device=args.device, model_id=str(args.vggt_checkpoint), precision="bfloat16"
    )
    load_seconds = time.perf_counter() - stage_started
    forward_started = time.perf_counter()
    with torch.inference_mode():
        batched_belief = teacher(_single_image_batch(smoke_samples, size=518))
        separate_beliefs = [teacher(_single_image_batch([sample], size=518)) for sample in smoke_samples]
    torch.cuda.synchronize(device)
    forward_seconds = time.perf_counter() - forward_started
    if batched_belief.depth_mean is None or batched_belief.depth_logvar is None:
        raise RuntimeError("VGGT smoke did not return required dense geometry")
    if batched_belief.pointmap_mean is None:
        raise RuntimeError("VGGT smoke did not return a point map")
    batched_depth = batched_belief.depth_mean[:, 0, 0]
    separate_depth = torch.cat(
        [value.depth_mean[:, 0, 0] for value in separate_beliefs if value.depth_mean is not None]
    )
    _require_finite("VGGT depth", batched_depth)
    _require_finite("VGGT depth log variance", batched_belief.depth_logvar)
    _require_finite("VGGT point map", batched_belief.pointmap_mean)
    teacher_24 = F.interpolate(
        batched_depth.float().unsqueeze(1), size=(24, 24), mode="bilinear", align_corners=False
    )[:, 0].cpu()
    teacher_scale = _fit_metric_scale(teacher_24, target_24)
    teacher_24 *= teacher_scale
    report["vggt_smoke"] = {
        "sample_ids": [sample.sample_id for sample in smoke_samples],
        "load_seconds": load_seconds,
        "forward_seconds_batched_plus_separate": forward_seconds,
        "depth_mean": _tensor_summary(batched_belief.depth_mean),
        "depth_logvar": _tensor_summary(batched_belief.depth_logvar),
        "pointmap_mean": _tensor_summary(batched_belief.pointmap_mean),
        "scale_confidence": batched_belief.scale_confidence.detach().cpu().tolist(),
        "pose_confidence": batched_belief.pose_confidence.detach().cpu().tolist(),
        "reconstruction_confidence": batched_belief.reconstruction_confidence.detach().cpu().tolist(),
        "chunk_invariance": {
            # VGGT executes BF16 transformer blocks; these are PyTorch's
            # standard BF16-scale tolerances and still reject scene mixing.
            "depth": _assert_close("VGGT depth", batched_depth, separate_depth, rtol=2e-2, atol=1e-2),
            "compared_chunk_sizes": [1, 8],
        },
        "smoke_training_scale": teacher_scale,
        "metadata": batched_belief.metadata,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(device),
    }
    del teacher, batched_belief, separate_beliefs, batched_depth, separate_depth
    gc.collect()
    torch.cuda.empty_cache()

    stage_started = time.perf_counter()
    artifact_dir = args.output.parent / f"preflight-artifacts-{os.getenv('SLURM_JOB_ID', str(os.getpid()))}"
    artifact_dir.mkdir(parents=True, exist_ok=False)
    normalized_features = _normalize(probe_features, probe_features, probe_features)[0]
    model = DenseGeometryProbe(normalized_features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    optimizer.zero_grad(set_to_none=True)
    log_depth, log_variance = model(normalized_features.to(device))
    loss, parts = geometry_probe_loss(
        log_depth,
        log_variance,
        target_24.to(device),
        _valid(target_24).to(device),
        teacher_depth=teacher_24.to(device),
    )
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
    checkpoint = artifact_dir / "probe-one-step.pt"
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save({"state_dict": state, "input_dim": normalized_features.shape[1]}, checkpoint)
    reloaded = DenseGeometryProbe(normalized_features.shape[1]).to(device)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    reloaded.load_state_dict(saved["state_dict"], strict=True)
    reloaded.eval()
    with torch.inference_mode():
        prediction = reloaded(normalized_features.to(device))[0].exp().cpu()
    _require_finite("one-step probe prediction", prediction)
    metrics = _evaluate_depths(prediction, target_24)
    history = {
        "variant": "vjepa_probe_one_step",
        "seed": 0,
        "epoch": 0,
        "loss": float(loss.detach()),
        "gradient_norm": float(gradient_norm.detach()),
        **{key: float(value) for key, value in parts.items()},
    }
    comparison = {
        "experiment_id": "phase2b-preflight-probe-smoke",
        "schema_version": "jepa4d-phase2b-preflight-probe-v1",
        "dataset_manifest": str(args.manifest.resolve()),
        "split_hash": report["dataset"]["split_indices_sha256"],
        "metric_policy": {"primary": "metric_abs_rel on two training-only smoke frames"},
        "variants": [
            {
                "variant_id": "vjepa_probe_one_step",
                "family": "vjepa",
                "role": "preflight_smoke_only",
                "seed": 0,
                "metrics": metrics,
                "runtime": {},
                "parameters": sum(value.numel() for value in reloaded.parameters()),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256(checkpoint),
                "notes": ["One optimizer step on two training frames; not a model-quality result."],
            }
        ],
        "failures": [],
        "aggregates": {},
        "wandb_url": None,
    }
    report_artifacts = build_geometry_student_report(
        comparison,
        artifact_dir / "probe-smoke-report.html",
        training_history=[history],
        static_png=False,
    )
    report["probe_smoke"] = {
        "loss": float(loss.detach()),
        "loss_components": {key: float(value) for key, value in parts.items()},
        "gradient_norm": float(gradient_norm.detach()),
        "metrics": metrics,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_reload": "pass",
        "report": str(report_artifacts.html_path.resolve()),
        "report_sha256": sha256(report_artifacts.html_path),
        "report_warnings": list(report_artifacts.warnings),
    }
    _atomic_json(
        artifact_dir / "model-smoke.json",
        {
            "vjepa_smoke": report["vjepa_smoke"],
            "vggt_smoke": report["vggt_smoke"],
            "probe_smoke": report["probe_smoke"],
        },
    )
    report["timings_seconds"]["model_optimizer_report_smoke"] = time.perf_counter() - stage_started
    report["wandb_probe"] = _wandb_online_probe(args, report, artifact_dir)


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    report: dict[str, Any] = {
        "schema_version": "jepa4d-phase2b-preflight-v3",
        "status": "running",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": socket.gethostname(),
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "python": sys.version,
        "torch_cuda_build": torch.version.cuda,
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "timings_seconds": {},
    }
    exit_code = 0
    try:
        report["authorization"] = _authorization(args)
        run(args, report)
        report["status"] = "pass"
    except Exception as error:
        report["status"] = "fail"
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
        exit_code = 1
    report["total_seconds"] = time.perf_counter() - started
    _atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
