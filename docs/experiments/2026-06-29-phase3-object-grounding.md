# Phase 3 initial object-grounding experiment

## Experiment metadata

| Field | Value |
|---|---|
| Experiment ID | `2026-06-29-grounding-full-real-pipeline-v2` |
| Stage / status | `grounding + pipeline / complete` |
| Evidence level | `integration` |
| Promoted W&B run | [wvljbqlv](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/wvljbqlv) |
| Supersedes | `bojfn58h` for logging completeness; `4b1xse80` remains component evidence. |
| Decision | Keep the integrated pipeline and prioritize geometry latency plus sequence-level identity evaluation. |

## W&B dashboard reading guide

| Panel | What it answers | Observation | Insight / decision |
|---|---|---|---|
| `features/*` | Did the real JEPA substrate remain healthy inside the full pipeline? | Feature moments, distributions, and media are logged before downstream stages. | Full runs must not hide upstream regressions. |
| `geometry/*` | What geometry and uncertainty were attached to detections? | Depth, uncertainty, confidence, extents, and track diagnostics are present. | Slot 3D attachment inherits geometry uncertainty. |
| `objects/mask_and_box`, slot/query/observation tables | What was detected, localized, and converted into slots? | Per-object qualitative and tabular evidence is inspectable. | A detector output is an observation, not verified object truth. |
| `pipeline/stage_latency_s`, cumulative latency | Where is end-to-end time spent? | VGGT used 26.659 s of the 38.707 s run; V-JEPA used 3.630 s. | Geometry is the first performance optimization target. |
| artifact inventory | Can the exact result be reviewed outside the dashboard? | JSON, NPZ, SQLite, scene graph, HTML, and model/run artifacts were uploaded. | Keep local artifacts authoritative and W&B as the comparison surface. |

## Stage insights and decisions

| Stage | Evidence | Insight | Decision |
|---|---|---|---|
| V-JEPA | Real feature telemetry | Representation extraction is not the pipeline bottleneck. | Preserve feature diagnostics; avoid premature encoder optimization. |
| VGGT | 26.659 s stage latency | Geometry dominates this CPU/full-real configuration. | Profile device placement, resolution, and caching. |
| GroundingDINO | Load and inference measured separately | Model initialization materially affects one-shot runtime. | Report warm and cold latency independently in benchmarks. |
| Persistence | 0.492 s and queryable artifacts | Structured output overhead is small relative to learned adapters. | Continue into incremental memory. |
| Identity | No temporal benchmark in this run | Stable physical identity is not proven by one view set. | Evaluate appearance/IoU/mask association separately. |

## Question

Can the new object-slot boundary run end to end with both deterministic CPU mocks and a real open-vocabulary teacher,
attach JEPA/geometry evidence, persist queryable records, render an interactive report, and produce a meaningful W&B run?

## Code state

- baseline committed and pushed first: `5f8fa8b`;
- Phase 3 changes were uncommitted during execution and are committed after final regression;
- workspace: `facebookresearch/vjepa2` extended under `jepa4d/`;
- date: 2026-06-29 UTC.

## Inputs

Mock integration used two generated 384×256 views containing colored rectangles and queries `red mug` and
`wooden table`. The real teacher integration used `assets/architecture_vjepa2_1.jpg` and queries `person` and `robot`.
The latter is a repository illustration and therefore suitable only for integration evidence.

## Models and backends

- feature evidence: deterministic V-JEPA adapter, 64-dimensional dense token mock;
- geometry evidence: deterministic geometry belief at 112×112;
- real detector: `IDEA-Research/grounding-dino-tiny` through Transformers 5.12.1;
- masks: box-raster baseline;
- association: category-gated weighted appearance/IoU/geometry heuristic;
- device: CPU;
- SAM2 was not installed or exercised in this experiment.

## Mock results

- observations: 4;
- persistent slots: 2;
- observations per slot: 2;
- slots with geometry: 2;
- mask validity: 1.0;
- unique-ID fraction: 1.0;
- memory query `mug`: one matching persisted object.

These are expected deterministic contract results and are not accuracy metrics.

## Real GroundingDINO results

- model load: approximately 9.7 seconds on the first CPU process;
- detector/association runtime: 5.28 seconds;
- input mode: single image;
- queries: 2;
- detections/slots: 1/1;
- grounded label: `person`;
- detector score: 0.35783;
- geometry-attached slots: 1;
- result: complete JSON, NPZ, SQLite, scene graph, interactive HTML, and W&B artifacts.

The absence of a `robot` detection is not treated as an implementation failure. The threshold and the illustration are
not an evaluation protocol.

## W&B

Successful promoted run:

- name: `phase3-groundingdino-real-cpu-v2`;
- run ID: `4b1xse80`;
- URL: <https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/4b1xse80>;
- logged panels: runtime/count scalars, score statistics and histogram, geometry attachment fraction, slot table, semantic
  mask overlay, full configuration, summary, and four artifact groups.

An earlier run `bojfn58h` failed during W&B media construction because the semantic mask payload lacked a named layer.
Inference and local persistence had completed, but the run was not promoted. The schema was corrected to
`masks={"predictions": {"mask_data": ..., "class_labels": ...}}`, validated offline, and superseded by `4b1xse80`.

## Stagewise benchmark

Configuration: `jepa4d/config/benchmarks/object_grounding_smoke.yaml`.

- representation finite fraction: 1.0 across single image, multi-view, and video;
- geometry finite fraction: 1.0;
- multiview geometry confidence gain: 0.08;
- object association recall on deterministic fixture: 1.0;
- valid mask fraction: 1.0;
- unique ID fraction: 1.0;
- expected slot count: 2.

Smoke values establish regression invariants only. They cannot substitute for detection AP, segmentation IoU, HOTA,
IDF1, calibration, or task success.

## Artifacts

The successful local real run is under ignored `outputs/phase3_grounding_real/`:

- `objects.json`;
- `masks.npz`;
- `memory.db`;
- `scene_graph.json`;
- `report.html`;
- `EXPERIMENT.md`.

Artifacts are also versioned in W&B. Generated outputs and local W&B state remain git-ignored; this tracked Markdown is
the durable experiment narrative.

## Verification performed

- Ruff format and lint;
- mypy over the JEPA-4D package;
- 27 JEPA-4D unit tests before final additions;
- mock object demo and SQLite query;
- stagewise representation/geometry/object smoke benchmark;
- real GroundingDINO CPU run without W&B;
- W&B offline mask/media test;
- real GroundingDINO CPU run with online W&B and artifact upload;
- complete JEPA-4D and upstream regression suite after documentation/code finalization.

## Interpretation

The Phase 3 substrate is usable for research iteration: real text grounding reaches typed observations, JEPA tokens and
geometry enrich them, repeated observations produce slots, and results are inspectable locally and remotely. The result
is not yet robust object memory. Identity assignment is batch-local, confidence is heuristic, masks are box baselines,
and the real smoke has no labeled ground truth.

## Next experiment

Install and pin official SAM2, run prompted masks on a versioned real image/video fixture, measure mask IoU and temporal
identity, then build incremental durable track updates in Phase 4. In parallel, establish a small labeled multi-view
fixture with two same-category instances to quantify false merges and ID switches.

## Logging remediation and full real pipeline

After review, the initial run was judged incomplete because W&B started after feature extraction, geometry inference, and
teacher construction. The logger was moved before all heavyweight stages and expanded with feature, geometry, object,
pipeline, hardware, table, distribution, overlay, and artifact panels. An offline serialization test passed before the
replacement real run.

Replacement full real run:

- V-JEPA2.1 ViT-B: real local checkpoint;
- geometry: real local VGGT-1B checkpoint;
- detector: real GroundingDINO-tiny;
- device: CPU, explicitly labeled because the A100 PCI device was unavailable;
- end-to-end measured time: 38.707 seconds;
- V-JEPA: 3.630 seconds;
- VGGT geometry: 26.659 seconds;
- GroundingDINO load/inference: 2.593/5.260 seconds;
- persistence/report: 0.492 seconds;
- W&B run: <https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/wvljbqlv>.

The promoted v4 run synced 14 media files and 26 artifact files, exposes more than 70 scalar/summary fields, adds repeated
stage/cumulative latency scalar series, uses run-scoped artifact names, and includes scene-graph and Markdown records.
The earlier complete runs `2my1vsxu` and `7yy18id3` remain valid but are superseded for dashboard review. GPU diagnosis
and the dashboard contract are documented in `docs/GPU_AND_OBSERVABILITY.md`.
