# Phase 3 design: open-vocabulary observations and persistent object slots

## Scope and outcome

Phase 3 turns text-conditioned detections into explicit, serializable object observations and cross-view/time object
slots. It connects three existing substrates: RGB observations, dense V-JEPA 2.1 tokens, and geometry belief. The output is
the first planner-safe semantic boundary: downstream code receives object IDs, evidence references, confidence fields,
poses, and affordance priors instead of detector tensors.

This phase does not claim general object permanence, calibrated detection probability, or manipulation readiness. A slot
is a hypothesis supported by observations. Occlusion reasoning, transactional history, confidence decay, and robust loop
closure belong to Phase 4 memory work.

## Runtime pipeline

```text
RGBInputBatch + text queries
        │
        ├── GroundingDINO or deterministic mock → boxes, labels, scores
        ├── box mask or optional SAM2           → per-observation mask
        ├── V-JEPA dense token pooling          → normalized visual embedding
        └── GeometryBelief point map            → masked 3D centroid
                         │
                         ▼
             greedy evidence association
                         │
                         ▼
      ObjectSlot + observation references + confidence
                         │
                         ▼
          SceneGraph + SQLite + JSON/NPZ + HTML/W&B
```

Teacher models are lazy optional dependencies. CPU CI uses the deterministic detector and box masks; the real detector
path uses the Hugging Face Transformers implementation of GroundingDINO. SAM2 follows the official image-predictor API
and is loaded only when selected.

The current grounder processes one canonical sample (`B=1`) per call and rejects collated batches explicitly. Training
code should iterate/unbatch until the Phase 3b head introduces a batch-native output contract; silently merging identities
across samples is prohibited.

## Contracts

`ObjectObservation` records evidence tied to the source axes:

- `observation_id`, batch/view/time indices, and camera ID;
- canonical category and detector score;
- pixel-coordinate XYXY box and full-resolution boolean mask;
- normalized visual embedding;
- optional map-frame 3D centroid.

`ObjectSlot` records the associated hypothesis:

- deterministic UUID5 object ID;
- category, description, latest mask/box, optional pose;
- aggregate normalized embedding;
- affordance/state dictionaries and factorized confidence;
- last-seen timestamp and complete observation references;
- in-memory observation objects for reports and debugging.

`ObjectGroundingResult` contains slots, raw observations, normalized queries, backend metadata, runtime, input shape, and
explicit mock flags. JSON stores mask/embedding summaries; NPZ stores lossless masks separately. This avoids enormous
JSON while keeping artifacts independently inspectable.

## Query normalization and detection

Queries are stripped, lower-cased, deduplicated, and sorted. Sorting makes query order irrelevant to mock outputs and
stable IDs. Empty query sets fail before any model call.

The GroundingDINO prompt joins terms with periods, matching the model's phrase-grounding convention. The processor
returns absolute XYXY boxes after target-size postprocessing. `text_labels` is preferred over numeric labels. Detector
and text thresholds remain explicit constructor settings; they are experiment configuration, not hidden constants.

The mock emits one deterministic box per query/view/time. Boxes shift slightly by view/time and scores derive from the
query hash. Its purpose is exercising association and persistence—not estimating accuracy.

## Masks

The `box` backend rasterizes each box into a boolean mask. It is deliberately named as a baseline: mask quality equals
box extent and must not be interpreted as segmentation quality.

The `sam2` backend supplies each box as an image prompt, requests multiple masks, and selects the highest predicted
score. This initial implementation handles image frames independently. Stateful video propagation and mask-memory
reuse are planned after the object history layer can represent splits, merges, and re-identification events.

## V-JEPA evidence

When a `JEPATokenBundle` is present, each box is projected onto the patch grid. Enclosed dense tokens are mean pooled and
L2 normalized. Video observation time is mapped to its tubelet bin. Without tokens, the fallback combines crop channel
statistics with a query hash so tests retain deterministic association behavior.

The current embedding is evidence for association, not a separately trained retrieval representation. Phase 3b should
train an object-slot projection head with teacher labels, multi-view positives, hard negatives, and temporal consistency.

## Geometry attachment

The full-resolution mask is nearest-neighbor resized to the point-map grid. Finite points under the mask are averaged to
produce an optional 3D centroid. Missing geometry produces `None` rather than a fabricated pose. The centroid inherits
the geometry belief's frame, scale ambiguity, and uncertainty; Phase 4 must retain that provenance instead of treating
it as ground truth.

## Association and stable identity

Observations are ordered by category, view, and time. A candidate compares against the latest observation in each
same-category cluster:

```text
score = 0.65 × cosine(visual embeddings)
      + 0.20 × box IoU
      + 0.15 × exp(-3D centroid distance), when both poses exist
```

The default acceptance threshold is 0.55. A new cluster is created otherwise. IDs are UUID5 values derived from category,
cluster order, and quantized mean pose. This is deterministic for a fixed evidence set, but it is not yet invariant to
arbitrary incremental insertion order. Phase 4 will move identity ownership into a transactional track manager.

## Confidence semantics

Confidence is intentionally factorized:

- `detection`: mean teacher score;
- `association`: observations divided by possible view/time observations, capped at one;
- `geometry`: zero when absent and a conservative placeholder when present;
- `overall`: a documented heuristic combining detection and repeated support.

These numbers are ranking signals, not calibrated probabilities or safety guarantees. Benchmark work must measure
reliability diagrams, expected calibration error, identity switches, false merges, and false splits before policy use.

## Affordances and states

Initial affordances are transparent lexical priors for `graspable`, `openable`, and `support_surface`. Visibility is the
only initial state. These fields prove the interface and enable query tests; learned affordance/state heads and active
verification must replace them before manipulation claims.

## Persistence and reports

`build_memory` writes:

- `objects.json`: readable slots, evidence summaries, and backend metadata;
- `masks.npz`: lossless per-observation masks;
- `scene_graph.json`: object nodes suitable for inspection;
- `memory.db`: SQLite records queried by `query_memory`;
- `report.html`: interactive Plotly boxes and slot table plus complete metadata;
- `EXPERIMENT.md`: timestamped local run record.

W&B receives scalar counts/runtime, detection-score distribution, geometry attachment rate, a slot table, semantic mask
overlay, and versioned artifacts. Credentials are read only from the environment. The HTML image is downsampled for
display while all boxes are rescaled consistently and original masks remain in NPZ.

## Testing strategy

CPU tests assert query-order-invariant IDs, one cross-view track per query, valid masks, normalized embeddings, JEPA token
usage, geometry centroids, JSON/NPZ serialization, and clear invalid-input failures. The stagewise smoke benchmark checks
association recall, valid masks, unique IDs, and slot count after representation and geometry smoke stages.

The real integration smoke loads `IDEA-Research/grounding-dino-tiny`, performs CPU inference, persists the result, and
logs W&B artifacts. This validates wiring only; the repository architecture illustration is not an accuracy dataset.

## Failure modes and mitigations

- Similar same-category instances may merge: add pairwise geometry, motion, and assignment constraints.
- Camera motion may reduce IoU: rely more on reprojection and learned embeddings.
- Single-view centroids inherit scale ambiguity: expose geometry confidence and request another view.
- Detector phrase variants may fragment categories: add ontology and language-embedding normalization.
- Independent SAM2 frames may drift: use official video propagation with identity-aware correction.
- Deterministic IDs may change after cluster reorder: make Phase 4 track IDs durable database entities.
- Large images inflate reports: downsample display only and retain lossless machine artifacts.

## Phase 3b training proposal

Freeze V-JEPA initially and train a compact slot projection, objectness, box/mask, association, state, affordance, and
uncertainty head. Supervision combines teacher pseudo-labels with licensed ground truth. Log total and component losses,
positive/negative similarity distributions, mask IoU, box AP, identity F1, ID switches, state/affordance metrics,
calibration, throughput, memory, gradients, parameter norms, and qualitative failure grids.

Every training run must name the dataset manifest, teacher revisions, checkpoint hashes, seed, preprocessing, trainable
parameter set, and evaluation policy in both W&B config and a corresponding Markdown record.

## Exit criteria before Phase 4 completion

1. Static single/multi-view and short-video paths pass with missing geometry and calibration.
2. Teacher packages remain optional and mock mode remains deterministic/offline.
3. Identity metrics are reported on an external labeled fixture, not inferred from smoke tests.
4. Confidence calibration is measured and used by verification policy.
5. Incremental updates preserve durable IDs under replay.
6. SAM2 image and video paths have pinned revisions and integration tests.
7. Planner/query code consumes slots and evidence references, never raw detector tensors.
