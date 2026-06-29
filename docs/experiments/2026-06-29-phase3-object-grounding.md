# Phase 3 initial object-grounding experiment

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
