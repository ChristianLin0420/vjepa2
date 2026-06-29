# Phase 2 geometry experiment — 2026-06-29

## Objective

Validate the complete geometry-belief path with deterministic mock data and the official VGGT-1B checkpoint, while
ensuring uncalibrated single/multi-view outputs remain explicitly uncertain.

## Environment

- CPU: 16 logical cores, Intel Xeon Silver 4208;
- RAM: 62 GiB, no swap;
- CUDA: unavailable due host-driver/PyTorch mismatch;
- VGGT source: official repository commit `a288dd0f14786c93483e45524328726ab7b1b4ce`;
- model: official `facebook/VGGT-1B`, local Safetensors, approximately 5.03 GB;
- checkpoint SHA-256: `f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e`;
- input: generated RGB gradient views; no intrinsics, extrinsics, or metric-scale prior.

## Contract and mock results

The deterministic backend generated `[1,3,1,112,112]` depth, XYZ points, camera priors, uncertainty, and 64 grid tracks.
NPZ, PLY, metadata, Markdown, and interactive Plotly reports opened successfully. The geometry smoke benchmark produced
finite fraction 1.0 and positive two-view confidence gain 0.08.

## Official single-image result

- output depth: `[1,1,1,518,518]`;
- output point map: `[1,1,1,518,518,3]`;
- 2D/3D tracks: 64;
- finite fraction: 1.0;
- depth mean/std: 0.95113 / 0.01198 in teacher-relative units;
- scale/pose/reconstruction confidence: 0.08 / 0.05 / 0.1761;
- total adapter runtime: 14.99 s, including 11.23 s model loading.

The low scale/pose confidence is correct policy for an uncalibrated single image. Depth units must not be interpreted as
meters.

## Official three-view result

- output depth: `[1,3,1,518,518]`;
- output point map: `[1,3,1,518,518,3]`;
- tracks: `[1,3,64,2]` and `[1,3,64,3]`;
- finite fraction: 1.0;
- depth mean/std: 0.69320 / 0.02492 in teacher-relative units;
- scale/pose/reconstruction confidence: 0.24 / 0.50 / 0.1950;
- runtime: 47.06 s including 11.57 s load;
- W&B: <https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/l6nfxczi>.

Artifacts:

- `outputs/jepa4d_phase2/real_vggt_multiview_wandb/geometry_belief.npz`;
- `outputs/jepa4d_phase2/real_vggt_multiview_wandb/pointcloud.ply`;
- `outputs/jepa4d_phase2/real_vggt_multiview_wandb/report.html`;
- `outputs/jepa4d_phase2/real_vggt_multiview_wandb/EXPERIMENT.md`.

## Interpretation

The experiment establishes software correctness, official-weight loading, tensor alignment, finite geometry, track
extraction, persistence, reporting, and observability. It does not establish accuracy because the synthetic gradients have
no ground-truth cameras or depth. Multi-view confidence rises but remains below metric-planning thresholds.

## Limitations and next action

Run DTU/ScanNet++ or a small calibrated fixture, verify camera convention numerically, report pose/depth/point errors, and
fit confidence calibration on a separate split. Re-run on compatible CUDA hardware to measure deployable latency and
memory. Then begin JEPA-conditioned geometry distillation rather than adding more unmeasured teachers.
