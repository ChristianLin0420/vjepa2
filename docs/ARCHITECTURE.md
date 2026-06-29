# JEPA-4D WorldModel architecture

## 1. Purpose and system boundary

JEPA-4D is an RGB-first substrate for robot perception, geometric belief, persistent memory, and verified planning. It
extends the V-JEPA 2 repository without relocating or modifying the original training stacks. Upstream code remains under
`src/`, `app/`, `evals/`, and `configs/`; all new work lives under `jepa4d/` and calls upstream components through adapters.

The product is not a mesh, a point cloud, or a raw feature tensor. The product is a queryable belief over a changing
world, including geometry, objects, regions, observations, time, uncertainty, and task evidence. Reconstruction quality
is one input to that belief and is benchmarked independently.

Implemented phases:

- Phase 0: package structure, contracts, deterministic mocks, query boundary, tests, CI, and experiment records.
- Phase 1: real and mock V-JEPA 2.1 dense feature extraction, multi-layer tokens, serialization, and observability.
- Phase 2: deterministic geometry mock, optional official VGGT backend, camera/depth/point/track beliefs, uncertainty,
  NPZ/PLY export, interactive visualization, and geometry metrics.
- Phase 3: deterministic and GroundingDINO observations, box/SAM2 mask boundary, JEPA/geometry evidence, persistent
  cross-view slots, SQLite/scene-graph records, object reports, W&B diagnostics, and stagewise smoke metrics.
- Phase 4: bounded active local map, temporal global object graph, episodic events, vector retrieval, atomic SQLite
  records/event log/snapshots, reload/replay, LOD compression, memory queries, reports, and revision metrics.

Not yet implemented as production systems: calibrated object permanence and identity repair, metric map/region fusion,
trainable latent dynamics, behavior-tree robot execution, and full dataset benchmark adapters.

## 2. End-to-end information flow

```text
RGB sources
  single image | unordered view set | synchronized cameras | short video
        │
        ▼
RGBInputBatch [B,V,T,3,H,W]
  timestamps, camera IDs, optional K/Tcw, robot state, validity mask
        │
        ├───────────────┐
        ▼               ▼
ViewSetTokenizer     GeometryBeliefHead
  mode/view/time       mock | official VGGT
  identity             cameras, depth, points, tracks, uncertainty
        │               │
        ▼               │
VJEPA21FeatureExtractor │
  dense/global/layers   │
        │               │
        └───────┬───────┘
                ▼
ObjectSlotGrounder
  masks, identity, category, state, language, affordances
                │
                ▼
FourDMemoryCore (Phase 4)
  active local map + hierarchical graph + episodic history
                │
                ▼
WorldModelQueryAPI
  typed spatial, semantic, temporal, uncertainty, and route queries
                │
                ▼
Latent dynamics + task graph + behavior tree (Phase 5)
  propose → execute → observe → verify → update → replan
```

The feature and geometry branches are separate in Phase 2. This allows independent benchmarking and avoids claiming that
VGGT geometry is already distilled into V-JEPA tokens. Phase 2 training design joins them through explicit distillation
losses, described in `DESIGN_PHASE02_GEOMETRY.md`.

Phase 3 joins the branches only at an evidence adapter: boxes pool V-JEPA patch tokens and masks pool point-map geometry.
This is not end-to-end training yet. Association preserves source view/time references so Phase 4 can replay, revise, or
reject a slot without losing its observations. Full rationale and failure modes are in `DESIGN_PHASE03_OBJECT_SLOTS.md`.

## 3. Canonical input contract

`RGBInputBatch.images` always uses `[B,V,T,3,H,W]`:

| Mode | Views | Time | Meaning |
|---|---:|---:|---|
| `single_image` | 1 | 1 | one monocular observation |
| `multi_view` | >1 | 1 | one timestamp or weakly ordered view set |
| `video` | 1 | >1 | monocular temporal clip |
| `multiview_video` | >1 | >1 | multiple camera streams over time |

The contract also carries `[B,V,T]` timestamps, camera IDs, optional `[B,V,3,3]` intrinsics, optional `[B,V,4,4]`
camera-from-world extrinsics, robot state, action history, source references, and a validity mask. Missing calibration is
ordinary input—not an exception. Modules must either infer a belief or return unavailable fields with uncertainty.

Variable-size examples are padded during collation. The validity mask distinguishes observations from padding. View/time
identity is never encoded by flattening and forgetting axes; flattening is only an internal implementation detail.

## 4. View and time identity

`ViewSetTokenizer` supplies three additive identity signals:

1. learned input-mode embedding;
2. learned view-index embedding;
3. continuous sinusoidal timestamp encoding.

It returns these separately from pixels so that upstream V-JEPA preprocessing remains faithful. Future adapters can fuse
identity tokens into geometry, slots, and memory without changing `RGBInputBatch`. Camera IDs are retained as stable
references even though Phase 1 only learns view-index embeddings.

## 5. V-JEPA 2.1 feature substrate

At 384×384 resolution and patch size 16, each spatial frame has a 24×24 grid, or 576 patch tokens. Video uses tubelets of
two frames, so `T' = ceil(T/2)`. Odd clips duplicate only their last frame to complete a tubelet and propagate the correct
validity mask.

`JEPATokenBundle` exposes:

- dense tokens `[B,V,T',576,C]`;
- global mean-pooled tokens `[B,V,T',C]`;
- selected layer tokens keyed by block index;
- patch grid and feature scale;
- image/video modality;
- validity mask and model/runtime metadata.

Model widths and hierarchy layers:

| Model | Width | Blocks | Exposed hierarchy |
|---|---:|---:|---|
| ViT-B | 768 | 12 | 2, 5, 8, 11 |
| ViT-L | 1024 | 24 | 5, 11, 17, 23 |
| ViT-g | 1408 | 40 | 9, 19, 29, 39 |
| ViT-G | 1664 | 48 | 11, 23, 37, 47 |

The real adapter supports official native `.pt` checkpoints and a local HF conversion compatibility path. It rejects
missing weights that participate in execution. The deterministic mock is a shape- and identity-correct CI backend, not a
learned representation baseline.

## 6. Geometry belief architecture

`GeometryBeliefHead` has two backends:

### 6.1 Deterministic mock

The mock produces image-conditioned, finite tensors quickly on CPU. It creates a luminance/spatial depth surface, camera
priors, unprojected world points, grid tracks, and structured uncertainty. This backend validates contracts, exporters,
memory integration, tests, and reports. Its output must never be reported as model accuracy.

### 6.2 Official VGGT

The real backend uses the official `facebookresearch/vggt` package and `facebook/VGGT-1B` weights. RGB observations are
flattened internally to a scene sequence and resized to 518×518. VGGT returns pose encoding, depth and confidence, world
points and confidence, and tracks. Pose encoding is converted to OpenCV camera-from-world extrinsics and intrinsics. A
homogeneous row is added so all JEPA-4D extrinsics are 4×4.

The output is reshaped back to explicit `[B,V,T,...]` axes:

- extrinsics `[B,V,T,4,4]`;
- intrinsics `[B,V,T,3,3]`;
- depth mean/log-variance `[B,V,T,Hg,Wg]`;
- point mean/log-variance `[B,V,T,Hg,Wg,3]`;
- 2D tracks `[B,V*T,N,2]`;
- sampled 3D tracks `[B,V*T,N,3]`.

VGGT confidence scores are positive scores, not calibrated probabilities. The adapter maps `s` to `s/(s+1)` before
forming `-log(p)` uncertainty. This is monotonic and numerically stable, but it is not a substitute for held-out
calibration.

## 7. Epistemic uncertainty policy

A single RGB image cannot uniquely determine metric scale, hidden geometry, or object permanence. JEPA-4D therefore
separates three confidence dimensions:

- `scale_confidence`: whether numeric distances are task-usable in a metric frame;
- `pose_confidence`: whether camera transforms are reliable;
- `reconstruction_confidence`: aggregate dense geometry confidence.

Uncalibrated single-image scale is capped at 0.15. Additional views raise belief modestly; known intrinsics, known
extrinsics, or an explicit robot-size/metric prior raise only their relevant confidence. None of these values are accuracy
claims until calibrated on benchmark data. Planners must compare confidence with task-specific thresholds and invoke
`suggest_verification_action` when the threshold is not met.

## 8. Coordinate conventions

- Images use pixel coordinates with origin at the top-left.
- Intrinsics are standard 3×3 pinhole matrices.
- Extrinsics are 4×4 camera-from-world transforms following the VGGT/OpenCV convention.
- Point maps are world-frame XYZ for the real backend.
- The mock uses the same convention but its translation magnitudes are synthetic and non-metric without a scale prior.
- Phase 4 distinguishes `map` and `base_link` at its public boundary; a complete `odom`/camera frame graph remains open.

Every conversion between frames must carry its frame ID in production memory. Phase 2 arrays are scene-local and record
their convention in documentation and metadata.

## 9. Geometry persistence and visualization

NPZ stores complete numeric beliefs, including cameras, depth, log-variance, points, tracks, and scalar confidence. PLY
stores a bounded, finite, optionally uncertainty-filtered colored point cloud. The self-contained Plotly report includes
depth, uncertainty, a rotatable 3D point cloud, complete metadata, and an epistemic warning.

Phase 4 stores structured current records, append-only update rows, and snapshots in SQLite. Geometry artifacts remain
external experimental records; large-array lifecycle and DuckDB/vector backends remain future work.

## 10. Memory and query boundary

The Phase 4 memory substrate contains:

- bounded robot-centric local objects and observation summaries;
- temporal objects, histories, regions, relations, and connected-region routes;
- deterministic episodic events with evidence references;
- task status;
- atomic SQLite current records, event log, snapshots, reload, and replay;
- task-aware LOD compression;
- an in-memory cosine index boundary.

Planners do not receive `JEPATokenBundle` or dense geometry directly. They call object, region, route, observation,
affordance, uncertainty, verification, and task-state methods. Durable identity repair and hierarchical place inference
will extend these containers without exposing raw tensors.

## 11. Observability

W&B feature runs log token moments, finite fraction, norm histograms, PCA, temporal consistency, multi-layer statistics,
runtime, throughput, system versions, and artifacts. Geometry runs log depth moments/histograms, depth uncertainty,
scale/pose/reconstruction confidence, finite point fraction, XYZ extents, track count, runtime, NPZ, PLY, and interactive
reports. Every online run has a local `EXPERIMENT.md`; W&B is never the sole record.

Memory runs log per-revision insert/update counts, local/global object counts, event and history growth, persistence writes,
confidence, final object/event tables, run-scoped artifacts, and an interactive trajectory/confidence report.

The training logger reserves a stable namespace for component losses, learning rate, gradients, weights, throughput,
memory, and task-specific metrics. No training curves are fabricated before a training loop exists.

## 12. Failure boundaries and invariants

- A missing optional model produces an actionable installation error.
- A partially loaded V-JEPA model fails rather than silently reinitializing used weights.
- Missing calibration lowers confidence instead of crashing.
- Invalid shapes and confidence outside `[0,1]` fail at the contract boundary.
- Heavy models and W&B are absent from CPU CI.
- Mocks are labeled in metadata and reports.
- Geometry is not inserted into persistent memory until its frame and uncertainty are known.
- Real VGGT is practical on CUDA; CPU runs are correctness tests, not latency claims.

## 13. Security and licensing

Credentials are injected by environment or hidden terminal input and are excluded from tracked files. Checkpoints,
outputs, and W&B local state are ignored by Git. VGGT code and weights are governed by their own Meta license and
acceptable-use policy; users must review those terms before redistribution or deployment. JEPA-4D does not vendor VGGT
source or weights into the repository.
