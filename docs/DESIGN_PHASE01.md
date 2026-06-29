# Design record: Phase 0 and Phase 1 substrate

## Status

Implemented and regression-tested on 2026-06-29. This document records decisions rather than aspirational architecture.

## Context

The upstream repository contains production research code for V-JEPA 2, V-JEPA 2.1, and V-JEPA 2-AC, but its public
interfaces are training/evaluation oriented. JEPA-4D needs a stable application contract for heterogeneous RGB, explicit
view/time identity, serializable dense features, optional heavy dependencies, query boundaries, and offline CI.

## Decision 1: additive package

All new implementation resides in `jepa4d/`; upstream modules are imported but not moved. This minimizes merge conflict
with future upstream changes and preserves original scripts. New configs are namespaced under `jepa4d/config`.

Consequences:

- upstream tests remain meaningful regression coverage;
- adapters sometimes accommodate upstream checkpoint conventions;
- duplicate-looking utilities are acceptable when their contract differs;
- no upstream API is silently repurposed.

## Decision 2: six-axis RGB

Canonical pixels are `[B,V,T,3,H,W]`. Alternatives considered:

- list of arbitrary observations: flexible but difficult to batch and type;
- `[B,T,C,H,W]` with camera folded into batch: loses synchronized view identity;
- flattening `V*T`: simple for models but unsafe for tracks and memory references.

The selected form makes every observation addressable and enables padding through `[B,V,T]` validity.

## Decision 3: explicit modes

Mode is validated against dimensions. This prevents a single image accidentally entering a video tubelet path and makes
uncertainty policy mode-aware. Mixed-mode batches are padded to a maximum shape and currently receive the maximum-shape
mode; production training may add per-sample modes if mixed batches become necessary.

## Decision 4: separate identity tokens

Mode/view/time embeddings are returned beside pixels, not injected into V-JEPA pixels or upstream tokens. This preserves
pretrained behavior while giving later heads an explicit identity signal. Camera IDs remain strings for stable references.

## Decision 5: backend-independent feature bundle

Mocks, native checkpoints, and HF conversions return `JEPATokenBundle`. Dense, global, and intermediate tokens have
explicit axes. The adapter exposes raw intermediate block states and the normalized final output. Metadata records backend,
checkpoint, input/output time, patch grid, and runtime.

## Decision 6: tubelet semantics

Video is processed as a clip, not independent frames. V-JEPA tubelet size two yields `ceil(T/2)` output bins. Odd clips
duplicate the last frame only for model compatibility. Validity is duplicated and reduced with logical any, so the final
bin remains associated with the true last observation.

## Decision 7: strict real loading

Native `.pt` checkpoints use the upstream architecture. The HF conversion path uses an inspected compatibility
implementation because stock Transformers currently interprets conversion-only fields incorrectly. Missing parameters
used during forward execution cause an error; unused historical norms are tolerated only with an explicit mapping.

## Decision 8: deterministic mocks

Mocks are image-conditioned and deterministic, preserve all output shapes, and expose all hierarchy layers. They are
designed for contracts, CI, demos, and downstream development. Metadata and tags label every mock; no accuracy claim may
include mock output.

## Decision 9: dual persistence

PyTorch is the simplest complete local artifact. Zarr supports chunked arrays and later partial retrieval. JSON metadata
contains shapes and configuration but not millions of tensor values. Every CLI run also writes Markdown and interactive
HTML.

## Decision 10: observability without coupling

W&B is optional and imported lazily. Online/offline/disabled modes share metric names. Feature runs log moments,
histograms, PCA, temporal consistency, hierarchy tables, runtime, throughput, and artifacts. Future training has a reserved
schema for component losses, LR, gradients, weights, throughput, and memory. CI never authenticates or calls W&B.

## Decision 11: planner query boundary early

Phase 0 introduced memory/query interfaces before full memory exists. This prevents later planning code from normalizing
around raw tensors. The current implementations are intentionally simple, but method semantics are explicit and tested.

## Tooling and gates

- Ruff covers formatting/import/error rules.
- Mypy checks JEPA-4D source while treating upstream packages as external.
- Pytest covers contracts, modes, odd clips, real local checkpoint when present, geometry mock, memory, queries, dynamics,
  benchmark registry, and server health.
- CI uses only mocks and CPU.
- Pre-commit checks Python, YAML, TOML, whitespace, and EOF.

## Verified results

The Phase 0/1 gate on 2026-06-29 passed 18 JEPA-4D tests, 23 upstream tests, with three upstream CUDA-only skips. Mock
single-image, multi-view, and video-memory demos passed. Real ViT-B single-image output was `[1,1,1,576,768]`. The promoted
eight-frame run emitted four temporal bins and hierarchy layers 2, 5, 8, and 11.

## Known debt

- no dependency lock file yet;
- stock Transformers does not natively load the chosen V-JEPA 2.1 conversion;
- CUDA driver on the test host is incompatible with installed PyTorch CUDA;
- source references and mixed-mode collation need richer production semantics;
- global token is mean pooling, not a learned task token;
- no training integration is claimed despite logging hooks.

## Migration policy

Any future change to tensor axes, coordinate conventions, confidence semantics, persistence schema, or query return type
requires a new design record, migration note, version bump, and compatibility tests.
