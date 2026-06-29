# JEPA-4D experiment ledger

## Purpose

W&B dashboards are useful for comparison but mutable and account-dependent. Every experiment therefore writes a local
Markdown record and machine-readable artifacts. Promoted conclusions are copied into this tracked directory after review.

- [INDEX.md](INDEX.md) is the stage-by-stage evidence map and decision log.
- [TEMPLATE.md](TEMPLATE.md) is the stable schema for promoted and generated records.
- Dated files are immutable narratives for individual promoted experiments; amendments must be labeled.

The organization is intentionally append-only: new stages, datasets, seeds, or ablations add rows and records without
rewriting the meaning of older evidence.

## Run naming

Feature runs:

```text
<mock|real>-<model>-<mode>-<views>v<timesteps>t[-purpose]
```

Geometry runs:

```text
geometry-<mock|vggt|student>-<mode>-<views>v<timesteps>t[-purpose]
```

Object runs:

```text
objects-<mock|grounding_dino|student>-<box|sam2>-<mode>[-purpose]
```

Memory runs:

```text
memory-<mock|real>-<episode-or-dataset>-<updates>u[-purpose]
```

Identity runs:

```text
identity-<mock|real-vjepa>-<fixture-or-dataset>-<sequence>[-ablation]
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

Projects use `jepa4d-worldmodel`; tags include phase, stage, backend, model, input mode, dataset/split, evidence level,
and seed where applicable. Stable namespaces are `features/`, `geometry/`, `objects/`, `memory/`, `identity/`,
`dynamics/`, `planning/`, `training/`, `system/`, and `pipeline/`. Each promoted record includes a dashboard reading
guide that maps panels to questions, observations, insights, and decisions. Media and artifacts are supplemental; local
outputs remain the reproducible source. Credentials come only from environment variables.

## Required narrative structure

Every record uses the canonical order in [TEMPLATE.md](TEMPLATE.md): metadata; question and decision; stage results and
insights; reproduction configuration; W&B guide; numerical results; artifacts; failures; claim boundary; next
experiments. Stage-specific content belongs in optional appendices, keeping automated indexing possible.

Use one evidence label from [INDEX.md](INDEX.md). In particular, mocks are `contract-only`, unscored real-model demos are
`integration`, and a single named sequence is `sequence-level`; none should be presented as benchmark evidence.

## Failure records

Failed experiments state the last completed stage, exception, whether partial metrics are trustworthy, remediation, and
whether a replacement run supersedes them. Deleting failed evidence without explanation is discouraged.

## Current promoted runs

The authoritative, extensible list is [INDEX.md](INDEX.md). The earlier Phase 3 run `4b1xse80` remains useful component
evidence, while `wvljbqlv` is the promoted full-pipeline observability run.

The failed Phase 3 logging run `bojfn58h` is superseded by `4b1xse80`; its failure and remediation are recorded in
`2026-06-29-phase3-object-grounding.md`.
