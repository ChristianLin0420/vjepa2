# Phase 1 initial feature experiment — 2026-06-28

## Experiment metadata

| Field | Value |
|---|---|
| Experiment ID | `2026-06-28-representation-real-video-v1` |
| Stage / status | `representation / complete` |
| Evidence level | `integration` |
| Promoted W&B run | [gisjdqvx](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/gisjdqvx) |
| Decision | Retain real multi-layer V-JEPA 2.1 tokens as the common downstream substrate. |

## W&B dashboard reading guide

| Panel | What it answers | Observation | Insight / decision |
|---|---|---|---|
| `visualizations/input`, `visualizations/pca_rgb` | Are the input and dense spatial structure recognizable and aligned? | Both artifacts render for the promoted video. | Qualitative inspection is viable; PCA color is diagnostic, not semantic truth. |
| `features/value_histogram`, `features/norm_histogram` | Are tokens finite, non-constant, or dominated by outliers? | Finite, non-degenerate distributions were recorded. | Preserve histogram logging alongside aggregate moments. |
| `visualizations/temporal_consistency` | Do adjacent frames retain latent similarity? | A per-transition trace is available rather than one mean. | Use this as a diagnostic baseline, not a tracking metric. |
| `features/layer_summary` and per-layer mean/std/norm | Which layers expose useful scale and variation? | Multiple requested layers were captured and tabulated. | Downstream ablations should name the source layer instead of assuming the final layer. |
| `inference/*`, `system/*` | Is the run operationally usable? | Runtime, throughput, and system context were logged. | Compare later adapters against this feature-only floor. |

## Stage insights and decisions

| Stage | Evidence | Insight | Decision |
|---|---|---|---|
| RGB normalization | Valid video batch and input media | Unified input contracts survive real video preprocessing. | Reuse the same contract for geometry and grounding. |
| V-JEPA inference | Finite dense/global/intermediate tokens | The real checkpoint path is functional. | Proceed to downstream adapters. |
| Observability | Scalars, histograms, PCA, temporal trace, table, artifacts | No single plot is sufficient; the panel set exposes collapse, spatial structure, time, and layer choice. | Keep the full panel taxonomy for future representation experiments. |

## Objective

Validate the Phase 0/1 contracts for single-image, multi-view, and short-video inputs, then execute a real local V-JEPA
2.1 ViT-B checkpoint with multi-layer capture, feature persistence, interactive diagnostics, and W&B telemetry.

## Questions

1. Are all RGB modes shape-safe and serializable?
2. Does video retain tubelet time instead of flattening it away?
3. Are mock outputs deterministic enough for CPU CI?
4. Does the real conversion load every encoder parameter used by forward execution?
5. Can intermediate layers be captured without modifying upstream model code?
6. Do local and W&B artifacts contain enough information to reproduce the result?

## Environment

- Python 3.13, PyTorch 2.12.1, Transformers 5.12.1;
- 16 logical-core Xeon Silver 4208 CPU;
- CUDA unavailable because installed PyTorch CUDA requires a newer host driver;
- checkpoint: local `davevanveen/vjepa2.1-vitb-fpc64-384` Safetensors;
- compatibility implementation: inspected `Dev-Jahn/vjepa2.1-vitl-fpc64-384` implementation;
- deterministic RGB gradients, avoiding dataset licensing and download variability.

## Pre-registered success criteria

- JEPA-4D and upstream CPU tests pass;
- mock single image returns `[1,1,1,576,64]`;
- mock three-view input returns `[1,3,1,576,64]`;
- odd and even videos return `ceil(T/2)` bins;
- real ViT-B returns width 768 and finite output;
- layers 2, 5, 8, and 11 are persisted;
- PyTorch/Zarr, JSON, HTML, and Markdown artifacts are produced;
- W&B includes scalars, histograms, PCA, temporal plot, hierarchy table, and artifacts.

## Test results

The initial suite passed 18 JEPA-4D tests. The 2026-06-29 regression additionally passed 23 upstream tests, with three
CUDA-only skips. Ruff and mypy passed. Mock single-image, multi-view, and video-memory scripts completed.

## Mock results

- single image: `[1,1,1,576,64]`;
- three views: `[1,3,1,576,64]`;
- all finite and deterministic;
- reports contained PCA and full metadata;
- no semantic or accuracy claim is attached to mock feature values.

## Real single-image result

The local base checkpoint emitted `[1,1,1,576,768]` finite tokens. The adapter mapped the conversion's trained final
distillation norm to the correct hierarchy norm and rejected other unexpected or missing execution weights.

## Promoted real video result

- input: eight generated 384×384 RGB frames;
- temporal bins: four, reflecting tubelet size two;
- dense output: `[1,1,4,576,768]`;
- intermediate layers: 2, 5, 8, and normalized final layer 11;
- finite fraction: 1.0;
- feature mean/std: -0.03255 / 1.11923;
- adjacent temporal cosine mean/min: 0.99551 / 0.99371;
- forward: 4.125 s on CPU; instrumented total: 8.768 s;
- W&B: <https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/gisjdqvx>.

Artifacts are under `outputs/jepa4d_phase1/real_vitb_video_wandb_v3/`: `features.zarr`, `metadata.json`, `report.html`,
and `EXPERIMENT.md`.

## Observability result

The promoted run contains feature moments, finite fraction, token-norm histograms, input and PCA images, adjacent-bin
temporal consistency, per-layer mean/std/norm, layer shapes, system versions, throughput, feature artifact, and HTML report.

## Failed attempt and remediation

The first online attempt (`kp19zosi`) logged metrics but crashed because Zarr was treated as a file. Artifact handling was
changed to add directories correctly, verified offline, and rerun. `r0gomnkz` validated upload; `gisjdqvx` superseded it by
adding every hierarchy layer and per-layer scalars.

## Interpretation and limitations

This validates extraction and instrumentation. High temporal cosine on a smooth gradient is a sanity check, not evidence
of benchmark-level temporal understanding. CPU timing is correctness evidence, not robot-rate performance. Only ViT-B was
executed; global tokens are mean pooling; no task labels or geometry were involved; stock Transformers does not directly
support the chosen conversion.

## Next action

Proceed to explicit geometry belief with official VGGT, preserve single-image uncertainty, validate cameras/depth/points/
tracks and artifacts, and benchmark geometry independently before distilling it into V-JEPA features.
