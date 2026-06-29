# JEPA-4D experiment index

This is the entry point for promoted evidence. Read rows left-to-right as a chain of increasingly structured claims;
do not treat an integration run as benchmark evidence. Detailed records contain the reproduction command, artifacts,
W&B panel guide, limitations, and next decision.

## Evidence map

| Stage | Promoted record | W&B run | Evidence level | Key result | Decision enabled |
|---|---|---|---|---|---|
| 1 · representation | [V-JEPA 2.1 features](2026-06-28-phase1-initial.md) | [gisjdqvx](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/gisjdqvx) | integration | Real multi-layer video tokens are finite, persisted, and temporally diagnosable. | Use real V-JEPA features as the common substrate. |
| 2 · geometry | [VGGT geometry](2026-06-29-phase2-geometry.md) | [l6nfxczi](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/l6nfxczi) | integration | Three RGB views produce camera, depth, point-map, track, and uncertainty artifacts. | Attach geometry as an optional teacher belief, not calibrated truth. |
| 3 · grounding | [Object grounding](2026-06-29-phase3-object-grounding.md) | [wvljbqlv](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/wvljbqlv) | integration | Real V-JEPA + VGGT + GroundingDINO completes with stagewise observability and persistence. | Optimize geometry latency and test association separately. |
| 4 · memory | [Persistent 4D memory](2026-06-29-phase4-memory.md) | [fa9r6n1c](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/fa9r6n1c) | contract-only | Incremental history, occlusion, SQLite reload, event replay, queries, and LOD compression agree. | Move from fixture observations to real sequence updates. |
| 4D identity | [Identity ablation](2026-06-29-identity-ablation.md) | [fw4rj25e](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/fw4rj25e) | sequence-level | V-JEPA appearance beats RGB appearance on DAVIS `dogs-scale`, but IoU remains stronger. | Learn/project appearance features; retain geometry/IoU fusion. |

## Evidence levels

| Level | Meaning | Permitted claim |
|---|---|---|
| `contract-only` | Mock or controlled fixture validates schemas, control flow, persistence, or observability. | The implementation contract works. |
| `integration` | Real models run on a small unscored input. | Components interoperate and outputs are inspectable. |
| `sequence-level` | Metrics are computed on named real sequences, without broad held-out coverage. | A result holds for the reported sequence and operating point. |
| `benchmark` | Versioned dataset split, metric protocol, and held-out selection are reported. | Comparative model-quality evidence for that protocol. |
| `training` | A reproducible optimization run includes curves, checkpoints, and validation. | Optimization behavior and held-out checkpoint selection. |
| `closed-loop` | Repeated task execution includes failures, safety, latency, and recovery. | System-level task performance in the stated environment. |

## Reading order

1. Start with the key result and decision in the table above.
2. Open the phase record and use its W&B dashboard guide to understand each panel rather than reading charts in isolation.
3. Check the claim boundary before comparing runs.
4. Follow artifact paths for machine-readable values; W&B is the comparison surface, not the sole source of record.

New stages add a row without changing prior records. Use [TEMPLATE.md](TEMPLATE.md) for every promoted experiment.
