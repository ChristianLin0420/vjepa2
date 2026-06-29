# JEPA-4D experiment ledger policy

## Purpose

W&B dashboards are useful for comparison but mutable and account-dependent. Every experiment therefore writes a local
Markdown record and machine-readable artifacts. Promoted conclusions are copied into this tracked directory after review.

## Run naming

Feature runs:

```text
<mock|real>-<model>-<mode>-<views>v<timesteps>t[-purpose]
```

Geometry runs:

```text
geometry-<mock|vggt|student>-<mode>-<views>v<timesteps>t[-purpose]
```

Training runs additionally include dataset, objective, and seed. Names should identify meaning, not implementation ticket
numbers.

## Mandatory fields

1. objective and hypothesis;
2. timestamp, Git commit, and dirty state;
3. model source, revision, checkpoint path/hash, and license note;
4. exact inputs or versioned manifest;
5. preprocessing and calibration availability;
6. device, precision, dependency environment, and random seed;
7. tensor shapes and finite checks;
8. metrics with definitions and alignment/calibration policy;
9. runtime, memory, and artifact sizes;
10. W&B URL if enabled;
11. artifact paths and checksums for promoted results;
12. interpretation, limitations, failures, and next action.

## Promotion criteria

A run is promoted only if it completed, artifacts open, metrics are internally consistent, no secret is embedded, and the
result supports or rejects a stated question. Mocks may be promoted as infrastructure evidence but never model-quality
evidence. Failed runs remain documented when they reveal a real defect.

## Artifact conventions

- features: `.pt` or `.zarr`;
- geometry: `geometry_belief.npz` and `pointcloud.ply`;
- metadata and aggregate metrics: JSON;
- human report: self-contained HTML;
- experiment narrative: Markdown;
- checkpoints: ignored local directory with recorded SHA-256.

## W&B conventions

Projects use `jepa4d-worldmodel`; tags include phase, stage, backend, model, and input mode. Scalar namespaces are
`features/`, `geometry/`, `inference/`, and `training/`. Media and artifacts are supplemental; local outputs remain the
reproducible source. Credentials come only from environment variables.

## Failure records

Failed experiments state the last completed stage, exception, whether partial metrics are trustworthy, remediation, and
whether a replacement run supersedes them. Deleting failed evidence without explanation is discouraged.

## Current promoted runs

- Phase 1 multi-layer V-JEPA 2.1 video: `gisjdqvx`.
- Phase 2 official VGGT three-view geometry: `l6nfxczi`.
