# Design record: Phase 2 geometry belief and VGGT adapter

## Status

Implemented as a research substrate on 2026-06-29. Official VGGT single- and three-view inference has been executed.
Dataset accuracy and uncertainty calibration remain follow-up work.

## Goal

Convert one or more RGB observations into an explicit belief over cameras, depth, world points, tracks, scale, pose, and
reconstruction quality. The API must accept missing calibration, preserve view/time identity, run in deterministic mock
mode for CI, and expose uncertainty rather than hallucinated certainty.

## Why a geometry teacher is necessary

V-JEPA tokens encode visual and temporal structure but do not by themselves define camera intrinsics, camera pose, metric
scale, or 3D coordinates. Those quantities require geometric assumptions and supervision. VGGT directly predicts cameras,
depth, points, and tracks for one or many images, making it a suitable teacher/adapter. It is not treated as infallible.

## Official integration boundary

The adapter imports `vggt.models.vggt.VGGT` only when `backend="vggt"`. Weights load through the official
`PyTorchModelHubMixin` interface from `facebook/VGGT-1B` or a local snapshot. Source and weights are not vendored because
they carry their own Meta license and are large. The tested source commit was `a288dd0f14786c93483e45524328726ab7b1b4ce`.

VGGT input is `[B,S,3,518,518]`, where `S=V*T`. Flattening is internal; outputs are reshaped back to `[B,V,T]`.

## Output decisions

- Camera extrinsics are homogeneous 4×4 camera-from-world transforms.
- Intrinsics are 3×3 matrices at the geometry output resolution.
- Depth has scalar mean and log-variance.
- Point maps have XYZ mean and per-axis repeated log-variance.
- Tracks retain flattened scene-sequence order because a track crosses view/time observations.
- Three scalar confidences separate scale, pose, and dense reconstruction.

`pointmap_logvar` currently repeats point-branch confidence over XYZ. A future calibrated student may predict anisotropic
covariance.

## Confidence conversion

VGGT head confidence uses a positive activation and is not a probability. The adapter applies `p=s/(s+1)` and stores
`-log(p)` as a monotonic uncertainty proxy. This avoids invalid probability interpretation while retaining ordering.

Scene-level confidence additionally uses view count and known calibration. It is a policy heuristic, not empirical
coverage. Specifically:

- uncalibrated single image receives low scale and pose confidence;
- added views modestly improve relative pose/reconstruction belief;
- known intrinsics improve scale belief only modestly;
- known extrinsics make pose confidence high;
- `known_scale_prior` raises scale belief only when supplied by a legitimate external prior.

Calibration will use NLL, ECE, risk-coverage, and sparsification on held-out data.

## Deterministic mock design

The mock derives depth from luminance and vertical position, assigns higher border uncertainty, constructs camera priors,
unprojects to points, and samples grid tracks. It is fast, finite, input-dependent, and shape-correct. Its camera motion and
depth have no learned or metric validity. `synthetic_geometry=True` is mandatory metadata.

## Point cloud export

PLY uses world points and RGB resized to geometry resolution. Invalid points are removed. Sampling is deterministic and
bounded by `max_points`. An optional log-variance threshold supports uncertainty-aware export. ASCII was selected for
portability and inspection; binary PLY may be added for scale.

NPZ contains all numeric belief fields and is the authoritative Phase 2 exchange artifact. Metadata JSON carries summaries
and semantics without duplicating arrays.

## Interactive report

The report includes depth, depth log-variance, rotatable world points colored by uncertainty, input/configuration metadata,
and an explicit warning about single-image ambiguity. Plotly is embedded for offline use, increasing file size but avoiding
a hosted dependency.

## W&B schema

Geometry runs log input view/time counts, runtime, depth moments/histogram/map, depth uncertainty and p95, scale/pose/
reconstruction confidence, point finite fraction and XYZ extents, track count, NPZ, PLY, and HTML. Run names encode backend
and input mode. Local Markdown is always written.

## Real smoke results

On a 16-thread Xeon Silver 4208 CPU with 62 GiB RAM:

- single image: 14.99 s including 11.23 s load, finite 518×518 depth/points, 64 tracks;
- three views: 47.06 s including 11.57 s load, finite 3×518×518 depth/points, 64 tracks;
- checkpoint: official `facebook/VGGT-1B`, approximately 5.03 GB;
- checkpoint SHA-256: `f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e`;
- promoted W&B run: `l6nfxczi`.

These are correctness measurements. GPU latency must be profiled before robotics use.

## Alternatives considered

- using only V-JEPA geometry probes: insufficient before training a geometry head;
- making VGGT mandatory: breaks offline CPU development and CI;
- returning only point clouds: discards cameras, uncertainty, and dense alignment;
- claiming metric single-image output: violates observability and project principles;
- using confidence directly as probability: unsupported by the teacher output semantics;
- integrating bundle adjustment immediately: obscures feed-forward baseline and adds dependencies.

## Known limitations

- confidence is uncalibrated;
- input resize currently stretches to square rather than reproducing every official crop/pad mode;
- provided input calibration is not yet fed into VGGT, only reflected in output confidence/mock geometry;
- real tracks are queried on a fixed grid;
- track visibility/confidence are not yet preserved in the public belief;
- no COLMAP export or bundle adjustment;
- no metric frame registration to robot odometry;
- no batched variable-resolution scene support;
- no JEPA geometry student training yet.

## Phase 2 completion gate

The adapter implementation is complete when contracts, mock, official single/multi-view smoke, exports, reports, W&B,
tests, and docs pass. Scientific completion additionally requires dataset metrics, calibration, coordinate validation,
CUDA profiling, and student ablations.
