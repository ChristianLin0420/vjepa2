# Phase 2 geometry experiment — 2026-06-29

## Experiment metadata

| Field | Value |
|---|---|
| Experiment ID | `phase2-tum-rgbd-vggt-a100` |
| Legacy integration ID | `2026-06-29-geometry-vggt-three-view-v1` |
| Stage / status | `geometry teacher / complete` |
| Evidence level | `official mini subset` |
| Promoted W&B run | [rcpsxq6g](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/rcpsxq6g) |
| Legacy integration run | [l6nfxczi](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/l6nfxczi) |
| Decision | Freeze VGGT-1B as the measured optional teacher and move learned geometry comparisons to Phase 2b. |

## W&B dashboard reading guide

| Panel | What it answers | Observation | Insight / decision |
|---|---|---|---|
| `geometry/depth_map`, `geometry/depth_histogram` | Is the predicted depth spatially structured and numerically plausible? | Multi-view depth and its distribution are visible. | Inspect both; a plausible image can hide an implausible numeric range. |
| `geometry/depth_uncertainty`, depth log-variance | Where does the adapter express uncertainty? | Uncertainty artifacts are persisted with the mean. | Treat uncertainty as uncalibrated until held-out NLL/ECE evaluation. |
| scale/pose/reconstruction confidence | Does confidence respond to input mode and available views? | Three-view confidence is reported separately by type. | Never collapse these into one generic confidence score. |
| `geometry/point_extent_xyz` | Does the point map occupy a plausible coordinate extent? | Axis extents are logged. | Use as an export/sanity diagnostic, not reconstruction accuracy. |
| track count and runtime | What geometry products exist, and at what cost? | Tracks and end-to-end geometry time are explicit. | Later full-pipeline evidence identifies VGGT as the main latency target. |

## Stage insights and decisions

| Stage | Evidence | Insight | Decision |
|---|---|---|---|
| Single-image geometry | Low scale confidence without a prior | A monocular output is a belief, not a metric map. | Keep scale uncertainty explicit. |
| Multi-view geometry | Camera/depth/point-map/track exports | The adapter contract supports downstream 3D attachment. | Integrate with object slots and memory. |
| Calibration | No ground-truth split in this run | Confidence is introspection, not measured correctness. | Add dataset-level pose/depth/calibration evaluation before accuracy claims. |

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

## Original follow-up target

Run DTU/ScanNet++ or a small calibrated fixture, verify camera convention numerically, report pose/depth/point errors, and
fit confidence calibration on a separate split. Re-run on compatible CUDA hardware to measure deployable latency and
memory. The official TUM/A100 run below completes this target; JEPA-conditioned geometry distillation is the next phase.

## Downloaded W&B record snapshot

The original promoted CPU run record was downloaded through the W&B API on 2026-06-29 and verified in `finished` state.
Its persisted summary reports three input views, depth mean/std 0.693195/0.024921, finite point fraction 1.0, 64 tracks,
scale/pose/reconstruction confidence 0.24/0.50/0.195004, and geometry runtime 47.058953 s. W&B lists five logged
artifacts spanning the geometry belief, point cloud, interactive report, extent table, and run history. These are the
remote values for run `l6nfxczi`; the validated A100 quality run is recorded separately below.

## Official TUM RGB-D A100 completion run

Run [rcpsxq6g](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/rcpsxq6g) closes the Phase-2 teacher evaluation path on
an NVIDIA A100 80 GB PCIe. It uses eight deterministic frames from the official TUM RGB-D Freiburg1 XYZ sequence, split
into four calibration and four test frames. The source archive is CC BY 4.0, 448,204,271 bytes, and pinned by SHA-256
`a0236d97b8c30cd93b653656d2b6c293ff7c982a4130ef2a1a8beecdb124ef98` in
`jepa4d/config/benchmarks/manifests/tum_rgbd_freiburg1_xyz_mini.yaml`.

Held-out test results:

| Metric | Result |
|---|---:|
| Median-scale-aligned depth AbsRel | 0.043210 |
| Median-scale-aligned depth RMSE | 0.116805 m |
| Delta < 1.25 | 0.957320 |
| Aligned mean 3D point error | 0.052100 m |
| Aligned point F-score at 10 cm | 0.916100 |
| Sim(3)-aligned camera ATE RMSE | 0.016874 m |
| Sim(3)-aligned mean rotation error | 6.559037 degrees |
| Raw / calibrated Gaussian NLL | -0.025298 / -1.778085 |
| Uncertainty AUSE | 0.010390 |
| Per-sample failures | 0 |

The fitted variance multiplier is 0.008144. It was estimated only from calibration frames and frozen before test-frame
evaluation. The negative confidence/error correlation (-0.476503) has the expected direction: higher teacher confidence
is associated with lower aligned error on this subset.

CUDA profile (all outputs finite):

| Precision | Frames | Runtime | Peak allocated memory |
|---|---:|---:|---:|
| FP32 | 1 | 2.187 s | 5.385 GiB |
| FP32 | 2 | 3.506 s | 5.569 GiB |
| FP32 | 4 | 6.520 s | 6.120 GiB |
| FP32 | 8 | 13.679 s | 7.480 GiB |
| BF16 autocast | 1 | 1.336 s | 5.449 GiB |
| BF16 autocast | 2 | 1.331 s | 5.570 GiB |
| BF16 autocast | 4 | 1.819 s | 6.120 GiB |
| BF16 autocast | 8 | 3.143 s | 7.480 GiB |

The separately timed eight-frame FP32 quality pass took 15.163 s and peaked at 7.384 GiB. W&B verified the promoted run
in `finished` state with result `success`, zero failures, ten logged artifacts, per-sample and CUDA-profile tables, the
geometry belief/point cloud, COLMAP cameras/poses, JSON report, and experiment record.

### Claim boundary

This is one official sequence, not an independent-scene benchmark. Depth and point results are median-scale aligned and
must not be presented as metric-scale performance. Pose is Sim(3)-aligned. Calibration uses only four held-out-from-test
frames, so it validates the calibration mechanism rather than safety-grade probability coverage. Cross-dataset expansion
belongs to Phase 6; student and non-JEPA comparisons belong to Phase 2b.
