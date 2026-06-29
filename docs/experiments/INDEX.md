# JEPA-4D experiment index

This is the entry point for promoted evidence. Read rows left-to-right as a chain of increasingly structured claims;
do not treat an integration run as benchmark evidence. Detailed records contain the reproduction command, artifacts,
W&B panel guide, limitations, and next decision.

The living cross-phase synthesis is [INSIGHTS.md](INSIGHTS.md). It records conclusions, rejected shortcuts, and the next
gate without duplicating every numerical result.

## Evidence map

| Stage | Promoted record | W&B run | Evidence level | Key result | Decision enabled |
|---|---|---|---|---|---|
| 1 · representation | [V-JEPA 2.1 features](2026-06-28-phase1-initial.md) | [gisjdqvx](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/gisjdqvx) | integration | Real multi-layer video tokens are finite, persisted, and temporally diagnosable. | Use real V-JEPA features as the common substrate. |
| 2 · geometry | [VGGT geometry](2026-06-29-phase2-geometry.md) | [rcpsxq6g](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/rcpsxq6g) | official mini subset | TUM RGB-D held-out aligned depth/point metrics, pose validation, variance calibration, and A100 profiles complete with zero failures. | Freeze VGGT-1B as the measured teacher and begin Phase 2b distillation. |
| 2b · geometry student | [Geometry distillation](2026-06-29-phase2b-prepared-blocked.md) | [ikh4ptrb](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/ikh4ptrb) | sequence-level | Final-layer V-JEPA reaches 0.07523 ± 0.00384 AbsRel versus RGB 0.19417 and VGGT 0.12034, while running 8.30× faster with 14.44× lower encoder peak memory than VGGT. | Use the final layer by default; do not promote the fixed four-layer average, which is 4.44% worse on the primary metric despite mixed secondary gains. |
| 2c · cross-family geometry | [Cross-sequence learned fusion](2026-06-29-phase2c-cross-sequence.md) | [mfquwgbw](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/mfquwgbw) | sequence-level | Learned fusion improves final-layer macro AbsRel by 4.58% on both Freiburg-3 sequence means, but reaches 1.1655× final latency; fixed averaging and RGB are stronger raw-metric baselines. | Retain final by the frozen gate; target scale transfer, fresh camera families, and a separately registered latency confirmation. |
| 2d · causal diagnostics | [Fusion, scale, and latency audit](2026-06-29-phase2d-diagnostics.md) | [q1m52wi1](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/q1m52wi1) | sequence-level | Zeroing the learned residual gates changes raw AbsRel by only `0.000081`; a target-fitted per-image scalar oracle cuts raw AbsRel by `61.61%`; independently repeated learned/final latency is `1.02262×` (95% CI `[1.02196, 1.02332]`). | Stop attributing Phase 2c behavior to the learned gates, retain final-layer operationally, and make metric-scale transfer the next modeling target. |
| 2e · fresh sensor family | [SUNRGBD factorized geometry](2026-06-29-phase2e-sunrgbd.md) | [89ugevtp](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/89ugevtp) | benchmark | On untouched kv2, the registered candidate improves raw/aligned AbsRel by `2.67%/6.22%`, but scale error is `13.82%` worse, calibrated NLL is worse, and head latency is `9.3578×`; the operational gate fails. | Do not promote the candidate. Repair camera controls, detach scale learning from shape, and qualify component latency before another formal run. |
| 3 · grounding | [Object grounding](2026-06-29-phase3-object-grounding.md) | [wvljbqlv](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/wvljbqlv) | integration | Real V-JEPA + VGGT + GroundingDINO completes with stagewise observability and persistence. | Optimize geometry latency and test association separately. |
| 4 · memory | [Persistent 4D memory](2026-06-29-phase4-memory.md) | [fa9r6n1c](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/fa9r6n1c) | contract-only | Incremental history, occlusion, SQLite reload, event replay, queries, and LOD compression agree. | Move from fixture observations to real sequence updates. |
| 4D identity | [Identity ablation](2026-06-29-identity-ablation.md) | [fw4rj25e](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/fw4rj25e) | sequence-level | V-JEPA appearance beats RGB appearance on DAVIS `dogs-scale`, but IoU remains stronger. | Learn/project appearance features; retain geometry/IoU fusion. |
| 5 · planning | [Verified recovery](2026-06-29-phase5-planning.md) | [8kctk4mt](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/8kctk4mt) | contract-only | Explicit evidence, safe uncertainty rejection, failure attribution, and bounded recovery pass; real V-JEPA→CEM handoff ran on A100 before it became unavailable. | Integrate learned dynamics and a named simulator. |
| 6 · benchmarking | [Versioned benchmark harness](2026-06-29-phase6-benchmark-harness.md) | [63j8m3cp](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/63j8m3cp) | contract-only | Six stages × five repetitions produce validated manifests, bootstrap intervals, typed failures, and JSON/HTML/Markdown/W&B artifacts. | Add one official licensed mini subset per stage. |

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
