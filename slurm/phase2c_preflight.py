"""Real-data, real-model, learned-fusion, report, and online W&B Phase 2c gate."""

from __future__ import annotations

import argparse
import gc
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

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from jepa4d.models.geometry_belief import GeometryBeliefHead  # noqa: E402
from jepa4d.models.geometry_student import ResidualFusionGeometryProbe, geometry_probe_loss  # noqa: E402
from jepa4d.models.vjepa21_adapter import VJEPA21FeatureExtractor  # noqa: E402
from jepa4d.visualization.geometry_student_report import build_geometry_student_report  # noqa: E402
from scripts.run_phase2b_geometry_distillation import (  # noqa: E402
    _configure_determinism,
    _environment_snapshot,
    _evaluate_depths,
    _fit_metric_scale,
    _normalize_phase2c_layers,
    _profile_vjepa_probe_end_to_end,
    _single_image_batch,
    _targets,
    _valid,
)
from slurm.phase2b_gate import (  # noqa: E402
    asset_inventory,
    environment_fingerprint,
    repository_fingerprint,
    sha256,
)
from slurm.phase2c_gate import (  # noqa: E402
    atomic_json,
    bundle_identity,
    protocol_contract,
    validated_bundle,
)


def _tensor_summary(value: torch.Tensor) -> dict[str, Any]:
    detached = value.detach()
    finite = torch.isfinite(detached)
    finite_values = detached[finite].float()
    return {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "finite_fraction": float(finite.float().mean()),
        "mean": float(finite_values.mean()) if finite_values.numel() else None,
        "std": float(finite_values.std()) if finite_values.numel() > 1 else 0.0,
        "min": float(finite_values.min()) if finite_values.numel() else None,
        "max": float(finite_values.max()) if finite_values.numel() else None,
    }


def _require_finite(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        raise RuntimeError(f"{name} contains non-finite values")


def _assert_close(
    name: str,
    batched: torch.Tensor,
    separate: torch.Tensor,
    *,
    rtol: float,
    atol: float,
    max_outlier_fraction: float,
    max_relative_rmse: float,
    min_cosine_similarity: float,
) -> dict[str, float]:
    if batched.shape != separate.shape:
        raise RuntimeError(f"{name} chunking changed shape: {tuple(batched.shape)} != {tuple(separate.shape)}")
    difference = (batched.float() - separate.float()).abs()
    rmse = float(difference.square().mean().sqrt())
    reference_rms = float(separate.float().square().mean().sqrt())
    cosine = float(F.cosine_similarity(batched.float().reshape(1, -1), separate.float().reshape(1, -1)))
    close = torch.isclose(batched.float(), separate.float(), rtol=rtol, atol=atol)
    outlier_count = int((~close).sum())
    statistics = {
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "rmse": rmse,
        "reference_rms": reference_rms,
        "relative_rmse": rmse / max(reference_rms, 1e-12),
        "cosine_similarity": cosine,
        "outlier_count": float(outlier_count),
        "outlier_fraction": outlier_count / close.numel(),
        "rtol": rtol,
        "atol": atol,
    }
    if (
        statistics["outlier_fraction"] > max_outlier_fraction
        or statistics["relative_rmse"] > max_relative_rmse
        or cosine < min_cosine_similarity
    ):
        raise RuntimeError(f"{name} changes with chunk size: {statistics}")
    return statistics


def _authorization(args: argparse.Namespace) -> dict[str, Any]:
    repository = repository_fingerprint(args.repo_root)
    environment = environment_fingerprint()
    receipt = json.loads(args.test_report.read_text())
    if receipt.get("schema_version") != "jepa4d-phase2c-tests-v1" or receipt.get("status") != "pass":
        raise RuntimeError("Phase 2c test receipt is missing or does not pass")
    if receipt.get("protocol") != protocol_contract():
        raise RuntimeError("Phase 2c test receipt has a different protocol contract")
    if receipt.get("repository") != repository or receipt.get("environment") != environment:
        raise RuntimeError("code or Python environment differs from the passing Phase 2c tests")
    cuda = receipt.get("cuda_report", {})
    cuda_path = Path(str(cuda.get("path", "")))
    if not cuda_path.is_file() or sha256(cuda_path) != cuda.get("sha256"):
        raise RuntimeError("passing-test CUDA receipt is missing or changed")
    if cuda.get("summary", {}).get("status") != "pass":
        raise RuntimeError("passing-test CUDA receipt does not pass")
    return {
        "repository": repository,
        "environment": environment,
        "test_receipt": {
            "path": str(args.test_report.resolve(strict=True)),
            "sha256": sha256(args.test_report),
            "slurm_job_id": receipt.get("slurm_job_id"),
            "cuda_report_sha256": cuda.get("sha256"),
        },
    }


def _wandb_probe(args: argparse.Namespace, report: dict[str, Any], artifact_dir: Path) -> dict[str, Any]:
    import wandb

    run = None
    finished = False
    try:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            job_type="phase2c-preflight",
            mode="online",
            tags=["phase-2c", "preflight", "cross-sequence", "learned-fusion", "slurm", "cuda"],
            config={
                "schema_version": report["schema_version"],
                "slurm_job_id": report["slurm_job_id"],
                "protocol_sha256": report["protocol"]["sha256"],
                "bundle_sha256": report["dataset"]["sha256"],
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
                "preflight/fusion_loss": report["fusion_smoke"]["loss"],
                "preflight/fusion_gate_gradient_norm": report["fusion_smoke"]["gate_gradient_norm"],
                "preflight/cuda_peak_memory_gb": torch.cuda.max_memory_allocated() / 1024**3,
            }
        )
        artifact = wandb.Artifact(f"{run.id}-phase2c-preflight", type="preflight-validation")
        artifact.add_dir(str(artifact_dir), name="preflight")
        logged = run.log_artifact(artifact)
        logged.wait(timeout=900)
        result = {
            "mode": "online",
            "run_id": run.id,
            "run_url": run.url,
            "run_path": run.path,
            "artifact_name": logged.name,
            "artifact_version": logged.version,
            "artifact_digest": logged.digest,
        }
        if any(
            not result[key] for key in ("run_id", "run_url", "artifact_name", "artifact_version", "artifact_digest")
        ):
            raise RuntimeError(f"W&B returned an incomplete Phase 2c preflight receipt: {result}")
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
    parser.add_argument("--dataset-parent", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--vjepa-checkpoint", type=Path, required=True)
    parser.add_argument("--vjepa-implementation", type=Path, required=True)
    parser.add_argument("--vggt-checkpoint", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--test-report", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wandb-project", default="jepa4d-worldmodel")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-name", default="phase2c-preflight")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def run(args: argparse.Namespace, report: dict[str, Any]) -> None:
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Phase 2c preflight requires CUDA")
    device = torch.device(args.device)
    _configure_determinism()
    torch.cuda.set_device(device)
    properties = torch.cuda.get_device_properties(device)
    report["gpu"] = {
        "name": properties.name,
        "total_memory_bytes": properties.total_memory,
        "compute_capability": [properties.major, properties.minor],
    }
    runner_environment = _environment_snapshot(args.device)
    json.dumps(runner_environment, allow_nan=False)
    report["runner_environment_smoke"] = runner_environment

    started = time.perf_counter()
    bundle = validated_bundle(args.dataset_parent, args.manifest)
    report["dataset"] = bundle_identity(bundle)
    report["dataset"]["sequence_roles"] = {
        split: [selection.sequence_id for selection in bundle.selections if selection.split == split]
        for split in ("train", "validation", "test")
    }
    report["timings_seconds"]["bundle_validation"] = time.perf_counter() - started

    started = time.perf_counter()
    report["assets"] = {
        "vjepa_checkpoint": asset_inventory(args.vjepa_checkpoint),
        "vjepa_implementation": asset_inventory(args.vjepa_implementation),
        "vggt_checkpoint": asset_inventory(args.vggt_checkpoint),
        "hash_mode": "full",
    }
    report["timings_seconds"]["asset_inventory"] = time.perf_counter() - started

    train_selections = [selection for selection in bundle.selections if selection.split == "train"]
    smoke_samples = [*train_selections[0].samples[:4], *train_selections[1].samples[:4]]
    if len({sample.sequence_id for sample in smoke_samples}) != 2:
        raise RuntimeError("Phase 2c smoke batch must span both training sequences")
    batch = _single_image_batch(smoke_samples)
    if tuple(batch.images.shape[:3]) != (8, 1, 1):
        raise RuntimeError(f"formal batching must be B=8,V=1,T=1, found {tuple(batch.images.shape[:3])}")
    target_24 = _targets(smoke_samples, (24, 24))

    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    extractor = VJEPA21FeatureExtractor(
        checkpoint=args.vjepa_checkpoint,
        implementation_path=args.vjepa_implementation,
        backend="hf_compat",
        device=args.device,
        capture_layers=(2, 5, 8),
    )
    load_seconds = time.perf_counter() - started
    forward_started = time.perf_counter()
    with torch.inference_mode():
        batched_bundle = extractor(batch)
        separate_bundles = [extractor(_single_image_batch([sample])) for sample in smoke_samples]
    torch.cuda.synchronize(device)
    forward_seconds = time.perf_counter() - forward_started
    batched_dense = batched_bundle.dense_tokens[:, 0, 0]
    separate_dense = torch.cat([value.dense_tokens[:, 0, 0] for value in separate_bundles])
    _require_finite("V-JEPA dense tokens", batched_dense)
    if sorted(batched_bundle.layer_tokens) != [2, 5, 8]:
        raise RuntimeError(f"unexpected V-JEPA intermediate layers: {sorted(batched_bundle.layer_tokens)}")
    layer_invariance = {}
    for layer, tokens in batched_bundle.layer_tokens.items():
        batched_layer = tokens[:, 0, 0]
        separate_layer = torch.cat([value.layer_tokens[layer][:, 0, 0] for value in separate_bundles])
        _require_finite(f"V-JEPA layer {layer}", batched_layer)
        layer_invariance[str(layer)] = _assert_close(
            f"V-JEPA layer {layer}",
            batched_layer,
            separate_layer,
            rtol=1e-2,
            atol=3e-3,
            max_outlier_fraction=1e-4,
            max_relative_rmse=1e-4,
            min_cosine_similarity=0.99999,
        )
    final_grid = batched_dense.reshape(8, 24, 24, -1).permute(0, 3, 1, 2).contiguous().cpu()
    layer_grids = [
        batched_bundle.layer_tokens[layer][:, 0, 0].reshape(8, 24, 24, -1).permute(0, 3, 1, 2).contiguous().cpu()
        for layer in (2, 5, 8)
    ]
    report["vjepa_smoke"] = {
        "sample_ids": [sample.sample_id for sample in smoke_samples],
        "sequence_ids": [sample.sequence_id for sample in smoke_samples],
        "input_shape_b_v_t_c_h_w": list(batch.images.shape),
        "model_config": extractor.model_config,
        "load_seconds": load_seconds,
        "forward_seconds_batched_plus_separate": forward_seconds,
        "dense_tokens": _tensor_summary(batched_dense),
        "layer_tokens": {str(key): _tensor_summary(value) for key, value in batched_bundle.layer_tokens.items()},
        "chunk_invariance": {
            "dense": _assert_close(
                "V-JEPA dense tokens",
                batched_dense,
                separate_dense,
                rtol=1e-2,
                atol=3e-3,
                max_outlier_fraction=1e-4,
                max_relative_rmse=1e-4,
                min_cosine_similarity=0.99999,
            ),
            "layers": layer_invariance,
            "compared_chunk_sizes": [1, 8],
        },
        "peak_memory_bytes": torch.cuda.max_memory_allocated(device),
    }
    del extractor, batched_bundle, separate_bundles, batched_dense, separate_dense
    gc.collect()
    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    teacher = GeometryBeliefHead(
        backend="vggt", device=args.device, model_id=str(args.vggt_checkpoint), precision="bfloat16"
    )
    teacher_load_seconds = time.perf_counter() - started
    forward_started = time.perf_counter()
    with torch.inference_mode():
        batched_belief = teacher(_single_image_batch(smoke_samples, size=518))
        separate_beliefs = [teacher(_single_image_batch([sample], size=518)) for sample in smoke_samples]
    torch.cuda.synchronize(device)
    teacher_forward_seconds = time.perf_counter() - forward_started
    if batched_belief.depth_mean is None or batched_belief.depth_logvar is None:
        raise RuntimeError("VGGT smoke did not return dense geometry")
    if batched_belief.pointmap_mean is None:
        raise RuntimeError("VGGT smoke did not return a point map")
    batched_depth = batched_belief.depth_mean[:, 0, 0]
    separate_depth = torch.cat(
        [value.depth_mean[:, 0, 0] for value in separate_beliefs if value.depth_mean is not None]
    )
    _require_finite("VGGT depth", batched_depth)
    _require_finite("VGGT log variance", batched_belief.depth_logvar)
    _require_finite("VGGT point map", batched_belief.pointmap_mean)
    teacher_24 = F.interpolate(
        batched_depth.float().unsqueeze(1), size=(24, 24), mode="bilinear", align_corners=False
    )[:, 0].cpu()
    teacher_scale = _fit_metric_scale(teacher_24, target_24)
    teacher_24 *= teacher_scale
    report["vggt_smoke"] = {
        "sample_ids": [sample.sample_id for sample in smoke_samples],
        "load_seconds": teacher_load_seconds,
        "forward_seconds_batched_plus_separate": teacher_forward_seconds,
        "depth_mean": _tensor_summary(batched_belief.depth_mean),
        "depth_logvar": _tensor_summary(batched_belief.depth_logvar),
        "pointmap_mean": _tensor_summary(batched_belief.pointmap_mean),
        "chunk_invariance": {
            "depth": _assert_close(
                "VGGT depth",
                batched_depth,
                separate_depth,
                rtol=2e-2,
                atol=1e-2,
                max_outlier_fraction=1e-3,
                max_relative_rmse=5e-3,
                min_cosine_similarity=0.9999,
            ),
            "compared_chunk_sizes": [1, 8],
        },
        "smoke_training_scale": teacher_scale,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(device),
    }
    del teacher, batched_belief, separate_beliefs, batched_depth, separate_depth
    gc.collect()
    torch.cuda.empty_cache()

    started = time.perf_counter()
    artifact_dir = args.output.parent / f"phase2c-preflight-artifacts-{os.getenv('SLURM_JOB_ID', os.getpid())}"
    artifact_dir.mkdir(parents=True, exist_ok=False)
    smoke_features = {
        "vjepa_final": final_grid.half(),
        **{f"vjepa_layer_{layer}": value.half() for layer, value in zip((2, 5, 8), layer_grids, strict=True)},
    }
    normalized_variants, normalization_statistics = _normalize_phase2c_layers(
        smoke_features,
        smoke_features,
        smoke_features,
    )
    feature_stack = normalized_variants["vjepa_learned_fusion"][0]
    normalization_path = artifact_dir / "fusion-normalization.pt"
    torch.save(normalization_statistics, normalization_path)
    model = ResidualFusionGeometryProbe(final_grid.shape[1]).to(device)
    with torch.inference_mode():
        initial_fused = model.fusion(feature_stack[:1, 0].to(device), feature_stack[:1, 1:].to(device))
    if not torch.equal(initial_fused.cpu(), feature_stack[:1, 0].float()):
        raise RuntimeError("learned fusion does not initialize exactly as the final layer")
    optimizer = torch.optim.AdamW(
        [
            {"params": model.probe.parameters(), "weight_decay": 1e-4},
            {"params": model.fusion.parameters(), "weight_decay": 1e-4},
        ],
        lr=2e-3,
    )
    optimizer.zero_grad(set_to_none=True)
    log_depth, log_variance = model(feature_stack.to(device))
    loss, parts = geometry_probe_loss(
        log_depth,
        log_variance,
        target_24.to(device),
        _valid(target_24).to(device),
        teacher_depth=teacher_24.to(device),
    )
    loss.backward()
    gate_gradients = [parameter.grad for parameter in model.fusion.parameters()]
    if any(value is None or not torch.isfinite(value).all() for value in gate_gradients):
        raise RuntimeError("fusion gate gradients are missing or non-finite")
    gate_gradient_norm = float(
        torch.stack([value.float().square().sum() for value in gate_gradients if value is not None]).sum().sqrt()
    )
    if gate_gradient_norm <= 0:
        raise RuntimeError("fusion gate gradient norm is zero")
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0))
    optimizer.step()
    checkpoint = artifact_dir / "fusion-one-step.pt"
    torch.save(
        {
            "model_type": type(model).__name__,
            "input_dim": final_grid.shape[1],
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "fusion_state": model.fusion_state(),
        },
        checkpoint,
    )
    reloaded = ResidualFusionGeometryProbe(final_grid.shape[1]).to(device)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    reloaded.load_state_dict(saved["state_dict"], strict=True)
    model.eval()
    reloaded.eval()
    with torch.inference_mode():
        original = model(feature_stack.to(device))[0]
        restored = reloaded(feature_stack.to(device))[0]
    if not torch.equal(original, restored):
        raise RuntimeError("fusion checkpoint reload changed predictions")
    profile_extractor = VJEPA21FeatureExtractor(
        checkpoint=args.vjepa_checkpoint,
        implementation_path=args.vjepa_implementation,
        backend="hf_compat",
        device=args.device,
        capture_layers=(),
    )
    profile_smoke = {
        "vjepa_final": _profile_vjepa_probe_end_to_end(
            profile_extractor,
            reloaded.probe,
            smoke_samples,
            "vjepa_final",
            normalization_statistics,
            args.device,
            warmup_iterations=1,
            measured_iterations=2,
            repetitions=1,
        ),
        "vjepa_learned_fusion": _profile_vjepa_probe_end_to_end(
            profile_extractor,
            reloaded,
            smoke_samples,
            "vjepa_learned_fusion",
            normalization_statistics,
            args.device,
            warmup_iterations=1,
            measured_iterations=2,
            repetitions=1,
        ),
    }
    if profile_smoke["vjepa_final"]["capture_layers"] != []:
        raise RuntimeError("final-layer profile smoke captured intermediate layers")
    if profile_smoke["vjepa_learned_fusion"]["capture_layers"] != [2, 5, 8]:
        raise RuntimeError("learned-fusion profile smoke omitted intermediate layers")
    del profile_extractor
    gc.collect()
    torch.cuda.empty_cache()
    prediction = restored.exp().cpu()
    _require_finite("fusion smoke prediction", prediction)
    metrics = _evaluate_depths(prediction, target_24)
    history = {
        "variant": "vjepa_learned_fusion_preflight",
        "seed": 0,
        "epoch": 0,
        "loss": float(loss.detach()),
        "gradient_norm": gradient_norm,
        "gate_gradient_norm": gate_gradient_norm,
        **{key: float(value) for key, value in parts.items()},
        **model.fusion_state(),
    }
    comparison = {
        "experiment_id": "phase2c-preflight-fusion-smoke",
        "schema_version": "jepa4d-phase2c-preflight-fusion-v1",
        "dataset_manifest": str(args.manifest.resolve()),
        "split_hash": bundle.split_hash,
        "metric_policy": {"primary": "training-only smoke; not a model-quality result"},
        "variants": [
            {
                "variant_id": "vjepa_learned_fusion_preflight",
                "family": "vjepa",
                "role": "preflight_smoke_only",
                "seed": 0,
                "metrics": metrics,
                "runtime": {},
                "parameters": sum(value.numel() for value in reloaded.parameters()),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256(checkpoint),
                "notes": ["One optimizer step on training-only frames spanning both training sequences."],
            }
        ],
        "failures": [],
        "aggregates": {},
        "wandb_url": None,
    }
    report_artifacts = build_geometry_student_report(
        comparison,
        artifact_dir / "fusion-smoke-report.html",
        training_history=[history],
        static_png=False,
    )
    report["fusion_smoke"] = {
        "initialization": "exact_final_layer",
        "loss": float(loss.detach()),
        "loss_components": {key: float(value) for key, value in parts.items()},
        "gradient_norm": gradient_norm,
        "gate_gradient_norm": gate_gradient_norm,
        "fusion_state": model.fusion_state(),
        "normalization": str(normalization_path.resolve()),
        "normalization_sha256": sha256(normalization_path),
        "normalized_feature_shape": list(feature_stack.shape),
        "metrics": metrics,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_reload": "pass",
        "end_to_end_profile_smoke": profile_smoke,
        "report": str(report_artifacts.html_path.resolve()),
        "report_sha256": sha256(report_artifacts.html_path),
    }
    atomic_json(
        artifact_dir / "model-smoke.json",
        {
            "vjepa_smoke": report["vjepa_smoke"],
            "vggt_smoke": report["vggt_smoke"],
            "fusion_smoke": report["fusion_smoke"],
        },
    )
    report["timings_seconds"]["model_optimizer_report_smoke"] = time.perf_counter() - started
    report["wandb_probe"] = _wandb_probe(args, report, artifact_dir)


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    report: dict[str, Any] = {
        "schema_version": "jepa4d-phase2c-preflight-v1",
        "status": "running",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": socket.gethostname(),
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "python": sys.version,
        "torch_cuda_build": torch.version.cuda,
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "protocol": protocol_contract(),
        "timings_seconds": {},
    }
    exit_code = 0
    try:
        if not report["slurm_job_id"]:
            raise RuntimeError("Phase 2c preflight must run in Slurm")
        report["authorization"] = _authorization(args)
        run(args, report)
        report["status"] = "pass"
    except Exception as error:
        report["status"] = "fail"
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
        exit_code = 1
    report["total_seconds"] = time.perf_counter() - started
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
