# Phase 2f detached scale and camera conditioning — completed, no survivor

## Experiment metadata

| Field | Value |
|---|---|
| Experiment ID | `043d074d-20260630T043658Z` |
| Stage / status | `formal development training complete; external final skipped by design` |
| Evidence level | preregistered development benchmark and training evidence |
| Execution commit | `043d074dba2d6d757cb92295dd041407977b9ff5` |
| Hardware | 12 independent NVIDIA A100-SXM4-80GB latency allocations plus gated A100 Slurm jobs |
| Frozen reference | `M0_monolithic` |
| Candidate arms | `M1_detached_global`, `M2_canonical_k`, `M3_coarse_scale_field` |
| Scientific decision | **No enhanced arm passed the frozen latency gate; retain M0.** |
| External-final state | **Not opened. DIODE remains sealed and unconsumed.** |
| Postflight W&B run | [c5c5z4v3](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/c5c5z4v3) |
| Visual postflight | [report.html](../../outputs/jepa4d_phase2f/043d074d-20260630T043658Z/postflight/report.html) |
| Frozen protocol | [preregistration](2026-06-29-phase2f-scale-camera-preregistered.md) |
| Scheduler amendment | [eight-allocation array amendment](2026-06-30-phase2f-scheduler-amendment.md) |

This execution is technically complete: all 73 logical jobs completed, online W&B logging and artifacts succeeded, and
strict postflight passed every dependency and content-integrity check. Its scientific gate did not pass because no
enhanced arm survived the preregistered head-latency screen. That is a valid null selection result, not an execution
failure and not an external-final score.

## Decision at a glance

| Question | Observation | Decision |
|---|---|---|
| Do the compact arms fit the parameter budget? | M0/M1/M2/M3 contain 86,402 / 92,820 / 92,916 / 93,685 parameters; all are below the 95,042 hard cap. | Parameter count is not the blocker. |
| Does detached global scale meet the head-latency gate? | M1 is `1.6807x` M0, 95% CI `[1.6786, 1.6829]`, above the `1.10x` cap. | Do not pilot or formally train this implementation. |
| Does canonical-K conditioning meet the gate? | M2 is `3.6064x` M0, 95% CI `[3.5996, 3.6139]`. Its camera transform alone averages about `0.717` CUDA ms. | Optimize/precompute the camera path before testing its quality hypothesis. |
| Does the coarse scale field meet the gate? | M3 is `4.3621x` M0, 95% CI `[4.3540, 4.3712]`. | Optimize the shared camera path first, then profile the field separately. |
| What quality evidence was produced? | M0 completed all 12 camera-family rotation/seed runs. M1-M3 performed zero optimizer steps by frozen skip semantics. | Preserve the M0 development baseline; make no candidate-quality or camera-causality claim. |
| Was the fresh external test consumed? | Selector returned `eligible_arms=[]`, `survivor=null`, and `final_authorized=false`; the guard recorded `skipped_no_survivor`. | Keep DIODE sealed for a future preregistered survivor. |
| Did the scheduler run safely? | 12 base submissions represented 73 logical jobs; accounting peak was exactly 8 concurrent allocations. | The amended array design satisfies the administrative cap. |

The experimental funnel makes the outcome explicit:

```text
4 arms pass parameter audit
          |
12 x A100 interleaved latency replicas
          |
          +-- M0: PASS ----------------> 10-epoch pilot --> 12 formal runs
          +-- M1: 1.681x -- FAIL ------> audited skip ----> 12 formal skips
          +-- M2: 3.606x -- FAIL ------> audited skip ----> 12 formal skips
          +-- M3: 4.362x -- FAIL ------> audited skip ----> 12 formal skips
                                                       |
                                      no enhanced eligible arm
                                                       |
                                    DIODE final remains unopened
```

## Frozen protocol and six-stage execution

| Stage | Frozen purpose | Result | Evidence / visualization |
|---|---|---|---|
| 1. Data and opacity | Build four-family SUN RGB-D development cache while auditing only DIODE compressed bytes. | 512 SUN samples cached; DIODE content never listed, extracted, loaded, summarized, or previewed. | [cache report](../../outputs/jepa4d_phase2f/043d074d-20260630T043658Z/cache/cache_receipt.report.html), [asset receipt](../../outputs/jepa4d_phase2f/043d074d-20260630T043658Z/assets/asset_receipt.json) |
| 2. Model audit | Instantiate M0-M3 and enforce the `1.10x` parameter ceiling. | All four arms pass; totals are 86,402 to 93,685. | [static receipt](../../outputs/jepa4d_phase2f/043d074d-20260630T043658Z/static/static_receipt.json) |
| 3. Latency qualification | Run 12 independent randomized/interleaved A100 profiles and a 100,000-resample paired cluster bootstrap. | Only M0 passes the frozen complete-head wall-time gate. | [latency report](../../outputs/jepa4d_phase2f/043d074d-20260630T043658Z/latency-aggregate/report.html), [PNG](../../outputs/jepa4d_phase2f/043d074d-20260630T043658Z/latency-aggregate/qualification.png) |
| 4. Pilot and causal controls | Pilot only the latency allowlist; require finite optimization, exact reload, and zero forbidden cross-gradient. | M0 passes a 10-epoch pilot. M1-M3 write validated `skipped_not_qualified` receipts; K controls are therefore not executed. | [pilot-gate report](../../outputs/jepa4d_phase2f/043d074d-20260630T043658Z/pilot-gate/report.html) |
| 5. Formal development | Train admitted arms for four rotations x three seeds x 60 epochs. | M0 completes 12/12; the 36 candidate cells correctly skip with zero optimizer steps. | [selection report](../../outputs/jepa4d_phase2f/043d074d-20260630T043658Z/selection/report.html), [PNG](../../outputs/jepa4d_phase2f/043d074d-20260630T043658Z/selection/selection.png) |
| 6. One-shot final and postflight | Open DIODE only if exactly one enhanced survivor is authorized; validate the complete graph afterward. | No survivor, so final guard skips without target access. Postflight passes 72 predecessors and 790 hash references. | [final guard](../../outputs/jepa4d_phase2f/043d074d-20260630T043658Z/final/report.html), [postflight report](../../outputs/jepa4d_phase2f/043d074d-20260630T043658Z/postflight/report.html) |

The cache contains 128 samples from each of `kv1`, `kv2`, `realsense`, and `xtion`. Input, frozen-feature, and target
caches are 2.294 GB, 4.530 GB, and 14.79 MB respectively. The paired-transform validator produced at least eight distinct
updated intrinsics matrices per source; both assignment-change and matrix-change fractions are 1.0. These facts show that
the Phase-2e constant-K control defect was repaired in the protocol. They do not establish camera causality because M2/M3
never entered the pilot.

## Static and latency qualification

### Frozen primary gate

Complete-head host wall time is the preregistered efficiency estimand. Each of 12 A100 jobs used the same immutable batch,
30 warmups, then 30 independently randomized blocks of 100 batch-one iterations. The independent allocation is the
bootstrap unit; lower is better.

| Arm | Parameters | Parameter gate | Head wall mean (ms) | Ratio to M0 | Paired 95% CI | Latency gate | Qualified |
|---|---:|---:|---:|---:|---:|---|---|
| M0 monolithic | 86,402 | PASS | 0.352336 | 1.0000 | `[1.0000, 1.0000]` | PASS | **YES** |
| M1 detached global | 92,820 | PASS | 0.592176 | 1.6807 | `[1.6786, 1.6829]` | **FAIL** | NO |
| M2 canonical K | 92,916 | PASS | 1.270675 | 3.6064 | `[3.5996, 3.6139]` | **FAIL** | NO |
| M3 coarse scale field | 93,685 | PASS | 1.536928 | 4.3621 | `[4.3540, 4.3712]` | **FAIL** | NO |

The intervals are far from the `1.10x` boundary; this is not a noisy threshold decision. All four arms satisfy the hard
parameter ceiling, demonstrating again that trainable parameter count is not a runtime proxy for small GPU heads.

### Secondary end-to-end diagnostic

Encoder-plus-head wall time was frozen as a secondary descriptive quantity and cannot replace the primary gate after
results are seen.

| Arm | Encoder + head mean (ms) | Ratio to M0 | Interpretation |
|---|---:|---:|---|
| M0 | 25.665797 | 1.0000 | Reference |
| M1 | 25.972878 | 1.0120 | Frozen encoder hides most head overhead |
| M2 | 26.678843 | 1.0395 | System path remains under 1.10x descriptively |
| M3 | 26.933100 | 1.0494 | System path remains under 1.10x descriptively |

This contrast is operationally useful for designing Phase 2g, but it does not revise Phase 2f. The experiment intentionally
gated the head so that expensive conditioning could not be hidden behind a much larger frozen encoder.

### Component diagnosis

CUDA component means are synchronized but are not algebraically additive because paths and event scopes overlap.

| Arm | Dense decoder (ms) | Pooling (ms) | Scale head (ms) | Camera transform (ms) | Coarse field (ms) | Composition (ms) | Dominant issue |
|---|---:|---:|---:|---:|---:|---:|---|
| M0 | 0.2598 | - | - | - | - | 0.0526 | Dense baseline |
| M1 | 0.2656 | 0.0378 | 0.2034 | - | - | 0.0989 | Small scale-head and composition launches |
| M2 | 0.2703 | 0.0386 | 0.1924 | **0.7170** | - | 0.1000 | Camera summary path |
| M3 | 0.2717 | 0.0388 | 0.1935 | **0.7197** | 0.2983 | Camera path, then coarse field |

The shared V-JEPA encoding component averages 24.8928 CUDA ms and normalization 0.1185 ms. The candidate heads are small
relative to the encoder but consist of many tiny operations. This makes launch/capture/fusion behavior a more plausible
engineering target than reducing a few hundred parameters.

## Pilot, formal training, and selector

The M0 pilot completed 10 epochs and 320 optimizer steps. Its selected epoch is 8; reload is bitwise exact, every value is
finite, and the maximum forbidden gradient is exactly zero. Pilot development-test metrics were raw AbsRel `0.196857`,
aligned AbsRel `0.152308`, absolute log-scale error `0.092158`, NLL `-0.931912`, and AUSE `0.074243`.

Formal execution contains all 48 preregistered arm/rotation/seed cells:

| Outcome | Cells | Epochs / optimizer steps | Integrity consequence |
|---|---:|---:|---|
| M0 trained successfully | 12 | 60 epochs and 1,920 steps each; 23,040 steps total | 12 exact-reload checkpoints, finite metrics, zero forbidden gradient |
| M1-M3 skipped as disqualified | 36 | 0 epochs and 0 optimizer steps | Validated receipt, SUCCESS marker, and online W&B artifact for every skip |
| Pilot plus formal optimization | - | 23,360 total optimizer steps | No candidate optimization occurred |

Selector output is `eligible_arms=[]`, `survivor=null`, and `final_authorized=false`. M1-M3 are ineligible because they
were not pilot-qualified, not because of an observed development-quality loss. The external-final guard therefore wrote
`skipped_no_survivor` and did not construct an external target cache.

## M0 development baseline

These metrics are equal-weight development-family aggregates over the 12 M0 runs. They are a reusable baseline, not a
candidate comparison and not an external-final result.

| Metric | 12-run mean | Sample SD across runs |
|---|---:|---:|
| Raw AbsRel | 0.202083 | 0.029534 |
| Aligned AbsRel | 0.146821 | 0.035152 |
| Absolute log-scale error | 0.118882 | 0.034392 |
| Validation-calibrated NLL | -0.781757 | 0.185742 |
| AUSE | 0.072549 | 0.013865 |
| Variance multiplier | 6.3955 | 2.6048 |
| Selected epoch | 35.17 | 14.78 |

The SD is descriptive optimizer variation nested within four development families. It is not a population confidence
interval or a significance test.

### Camera-family rotations

| Rotation / held-out family | Raw AbsRel | Scale error | Aligned AbsRel | NLL | AUSE | Best epoch |
|---|---:|---:|---:|---:|---:|---:|
| R0 / kv2 | 0.161760 +/- 0.005068 | **0.087421 +/- 0.008037** | 0.129755 +/- 0.005152 | **-1.025747 +/- 0.045907** | **0.051650 +/- 0.002146** | 26.00 +/- 13.75 |
| R1 / kv1 | 0.210724 +/- 0.001610 | 0.140603 +/- 0.005706 | **0.123097 +/- 0.001714** | -0.550913 +/- 0.096017 | 0.079970 +/- 0.000882 | 39.00 +/- 9.54 |
| R2 / xtion | 0.196890 +/- 0.007241 | 0.159503 +/- 0.017305 | 0.129868 +/- 0.004455 | -0.751828 +/- 0.075954 | 0.072471 +/- 0.006124 | 24.00 +/- 8.54 |
| R3 / RealSense | 0.238956 +/- 0.009270 | 0.088003 +/- 0.003314 | 0.204565 +/- 0.005762 | -0.798542 +/- 0.040727 | 0.086103 +/- 0.001082 | 51.67 +/- 10.21 |

Values are mean +/- sample SD over three seeds. The pattern matters more than the pooled mean:

- kv2 is strongest on raw error, NLL, and AUSE.
- kv1 has the best aligned shape but substantially worse scale and NLL than kv2.
- xtion has shape comparable to kv2 but the worst scale error, indicating a scale-transfer problem.
- RealSense has scale error close to kv2 but the weakest raw/aligned geometry and AUSE, indicating a shape/domain problem.

Between-family behavior is much larger and more structured than seed variation. Future validation should prioritize
independent camera/domain coverage over adding more seeds to the same family.

### Exact per-run checkpoint results

| Rotation | Seed | Epoch | Raw AbsRel | Scale error | Aligned AbsRel | NLL | AUSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| R0 | 0 | 41 | 0.165510 | 0.095828 | 0.129280 | -1.003380 | 0.049541 |
| R0 | 1 | 23 | 0.155995 | 0.086620 | 0.124858 | -0.995310 | 0.053832 |
| R0 | 2 | 14 | 0.163776 | 0.079814 | 0.135128 | -1.078550 | 0.051578 |
| R1 | 0 | 44 | 0.209706 | 0.137983 | 0.124414 | -0.586855 | 0.079791 |
| R1 | 1 | 45 | 0.212580 | 0.147148 | 0.121159 | -0.442110 | 0.080928 |
| R1 | 2 | 28 | 0.209886 | 0.136677 | 0.123718 | -0.623773 | 0.079192 |
| R2 | 0 | 23 | 0.192270 | 0.151149 | 0.125061 | -0.794467 | 0.065401 |
| R2 | 1 | 33 | 0.193166 | 0.147960 | 0.133859 | -0.796882 | 0.076093 |
| R2 | 2 | 16 | 0.205235 | 0.179400 | 0.130684 | -0.664136 | 0.075920 |
| R3 | 0 | 59 | 0.235380 | 0.090886 | 0.204198 | -0.761781 | 0.085019 |
| R3 | 1 | 56 | 0.232007 | 0.084383 | 0.198996 | -0.842323 | 0.086106 |
| R3 | 2 | 40 | 0.249481 | 0.088741 | 0.210501 | -0.791522 | 0.087184 |

## Camera-causality result boundary

The development cache and transform algebra validate a non-degenerate updated/stale/wrong/permuted-K suite. However,
M2 and M3 failed latency before pilot training. The preregistration correctly prevented camera controls from being run on
untrained arms. Therefore Phase 2f supports neither of these claims:

- camera conditioning improves or worsens depth quality;
- detached global-scale learning improves or worsens cross-family transfer.

The supported claim is narrower: the current eager M1/M2/M3 implementations are ineligible under the frozen head-runtime
budget. This is an architecture-implementation screen, not a scientific rejection of their underlying hypotheses.

## External-final opacity

The asset job streamed only the DIODE validation archive's compressed bytes. It verified 2,774,625,282 bytes and SHA-256
`8e847e0923c57c221533c0040a49fc37a547af08f0a78ab235fdbf91dc362374`. It did not list or extract the archive and did not
load, summarize, visualize, or cache target values. Development jobs recorded zero external-archive reads/opens.

Because the selector found no survivor:

- `fresh_final_opened=false` in final and postflight receipts;
- no `FRESH_FINAL_OPENED.json` sentinel exists;
- no DIODE target or feature cache exists;
- the scientific gate is `{passed: false, reason: no_survivor}`;
- DIODE remains available for a future, separately preregistered one-shot comparison.

## Slurm execution and administrative cap

The final DAG uses distinct `p2f8-*` names and only 12 base `sbatch` submissions. Array task names expand to the semantic
logical labels recorded in the dependency graph.

| Base stage | Slurm ID | Array / cap | Logical work |
|---|---:|---:|---|
| T / A / C / Q | `29629524`-`29629527` | scalar | tests, sealed asset audit, development cache, static audit |
| L | `29629528` | `0-11%8` | 12 latency replicas, at most 8 running |
| LA | `29629571` | scalar | latency aggregation |
| P | `29629572` | `0-3%4` | four pilot cells |
| PG | `29629573` | scalar | pilot gate |
| F | `29629574` | `0-47%8` | 48 formal cells, at most 8 running |
| S / E / Z | `29629576`-`29629578` | scalar | selector, guarded final, postflight |

Accounting over every allocation attempt shows a peak of exactly eight concurrent `RUNNING` allocations. Slurm briefly
displayed up to ten rows when `RUNNING` and `COMPLETING` were counted together during array refill; accounting boundaries
show those extra rows were completion-display lag rather than more than eight overlapping allocations. First T start to Z
completion took 54 minutes 45 seconds.

Four formal skip cells (`F-M1-R1-S1`, `F-M1-R3-S0`, `F-M1-R3-S1`, and `F-M1-R3-S2`) stalled in nested-`srun` step
lifecycle on busy shared nodes. They had produced no receipt, SUCCESS marker, scientific output, or optimizer step. The
operator used `scontrol requeuehold`/`release` on the same logical IDs, removed only their empty output directories, and
submitted no new job. Each has `Restarts=1` and then completed its deterministic skip in 33-46 seconds; every other row has
`Restarts=0`. This was transparent same-ID administrative recovery, not a scientific retry.

## Integrity and online W&B

Strict postflight reports:

| Check | Result |
|---|---:|
| Logical jobs | 73 expected / 73 completed `0:0` |
| Validated predecessor receipts | 72 / 72 |
| Referenced file hashes | 790 checked |
| Receipt status counts | 3 `pass`, 29 `success`, 39 `skipped_not_qualified`, 1 `skipped_no_survivor` |
| Dependency graph SHA-256 | `a8fe58144474a4de662dd539afdc49cb1ac8e2716d014867e44a4d512b8920ec` |
| Preregistration SHA-256 | `1f6515790d3a7772fad131fca5d00b54a30371ad088e36dab66004b6987c2d64` |
| Scheduler amendment SHA-256 | `ccfa78b0b68119deca7a346b8bf5ba7fd5c6f5cef97c6bf1fc5bffb28c53489d` |
| Test receipt | 256 tests passed; Ruff, mypy, Bash syntax, CUDA health, and W&B checks passed |
| Postflight | `status=pass`, `integrity_status=pass` |

All 73 logical jobs have complete, unique online W&B run and artifact identities. High-level dashboards are:

| Stage | W&B run | What to inspect |
|---|---|---|
| Tests | [qshl5u4k](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/qshl5u4k) | command outcomes, CUDA health, source and graph identity |
| Asset seal | [7cj9nqq4](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/7cj9nqq4) | compressed-byte identity and opacity flags |
| Development cache | [e1bixetx](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/e1bixetx) | family counts, cache hashes, transform/K validity |
| Static audit | [a2c73dwy](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/a2c73dwy) | component and total parameter counts |
| Latency aggregate | [s3ppsosv](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/s3ppsosv) | arm ratios, confidence intervals, qualification cards |
| M0 pilot | [r3xa0n29](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/r3xa0n29) | losses, validation metrics, gradients, checkpoint selection |
| Pilot gate | [bsti5voz](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/bsti5voz) | frozen allowlist and skip reasons |
| Development selector | [i35wgtbz](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/i35wgtbz) | formal matrix completeness and no-survivor decision |
| External-final guard | [h1efznca](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/h1efznca) | sealed skip and no-open state |
| Strict postflight | [c5c5z4v3](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/c5c5z4v3) | final integrity, status distribution, artifact inventory |

The durable record contains 70 self-contained HTML reports and 29 PNG visualizations: cache; 12 per-replica latency
reports; aggregate latency; four pilot reports and gate; 48 formal reports, with curves for the 12 trained M0 cells;
selection; final guard; and postflight. Raw SUN/DIODE targets, large caches, and credentials were not uploaded.

Primary machine-readable files:

| Role | Artifact |
|---|---|
| Full immutable DAG | [dependency-graph.json](../../outputs/phase2f-gates/043d074d-20260630T043658Z/dependency-graph.json) |
| Test receipt | [tests.json](../../outputs/phase2f-gates/043d074d-20260630T043658Z/tests.json) |
| Static qualification | [static_receipt.json](../../outputs/jepa4d_phase2f/043d074d-20260630T043658Z/static/static_receipt.json) |
| Latency qualification | [qualification.json](../../outputs/jepa4d_phase2f/043d074d-20260630T043658Z/latency-aggregate/qualification.json) |
| Frozen development selector | [selector.json](../../outputs/jepa4d_phase2f/043d074d-20260630T043658Z/selection/selector.json) |
| External-final guard | [final_receipt.json](../../outputs/jepa4d_phase2f/043d074d-20260630T043658Z/final/final_receipt.json) |
| Strict terminal receipt | [postflight_receipt.json](../../outputs/jepa4d_phase2f/043d074d-20260630T043658Z/postflight/postflight_receipt.json) |

## Retry and supersession history

Only execution `043d074d-20260630T043658Z` is promoted. Earlier attempts remain under `outputs/failed_attempts` and are
not pooled with this result.

| Lineage | Outcome | Correction / disposition |
|---|---|---|
| `10d49d5e` | Submit bootstrap did not expose the repository on `PYTHONPATH` before worker launch. | Fixed bootstrap; no scientific result used. |
| `a130546c` | CPU bookkeeping jobs were rejected because every approved partition requires a GPU request. | All Slurm stages now request an allocation-compatible GPU; no experiment moved to the login node. |
| `348749da` | Test job failed Ruff formatting. | Formatting fixed and full gate rerun. |
| `85ef22d2` | Cache found a tiny antialias RGB overshoot outside `[0,1]`. | Clamp added with regression coverage; full graph rerun. |
| `edea008f` | T through E completed, but Z rejected a historical W&B snapshot against the finalized local receipt self-hash. | Snapshot validator corrected; this lineage is not promoted. DIODE remained sealed. |
| `05dd7e7e` | Corrected graph was intentionally canceled before cache receipt/SUCCESS when the scheduler design exceeded the requested submission/concurrency policy. | Partial caches invalidated and archived; no result used. |
| `mock-p2f8-043d074` | Dry-run with fake IDs proved exactly 12 base submissions, 73 logical tasks, and hash-bound amendment behavior. | Non-scientific validation only. |
| `043d074d` | Final 12-submission, max-eight graph completed and passed strict postflight. | **Only promoted Phase 2f execution.** |

These corrections address infrastructure, validation, and scheduler behavior. They do not change the frozen arms, losses,
data rotations, primary latency estimand, qualification threshold, or selector after seeing scientific results.

## Interpretation and comprehensive insights

1. **The null selection is useful.** The latency-first design prevented 36 expensive candidate training runs and a one-shot
   external-final opening after the implementations had already violated a hard operational constraint.
2. **The result is about implementation cost, not model quality.** M1-M3 never trained, so Phase 2f cannot answer whether
   detaching scale gradients, adding canonical K, or using a coarse scale field improves depth.
3. **The estimand controls the conclusion.** Candidate end-to-end ratios look modest because V-JEPA dominates the system,
   while head-only ratios fail decisively. Both views should be retained, but changing the gate post hoc would invalidate
   the preregistration.
4. **M2/M3 share an obvious optimization target.** Roughly 0.717-0.720 ms sits in a deterministic four-value camera
   transform, much more than its arithmetic should require. Precomputation or fusion is higher leverage than changing the
   camera representation again.
5. **M1 also needs systems work.** Even without camera conditioning, several tiny pooling, scale-head, and composition
   operations increase head time by 68%. This points to launch overhead and graph structure, not parameter capacity.
6. **Family effects are not one-dimensional.** xtion mainly exposes scale transfer; RealSense exposes shape/domain error;
   kv1 has strong aligned shape but weak NLL/scale. A single pooled score would hide the failure taxonomy.
7. **The opacity contract worked.** No survivor means no external score. Preserving the unopened final is more valuable
   than generating an un-actionable number after the decision was already fixed.
8. **Operational reproducibility includes scheduler behavior.** The graph records arrays, semantic task names, dependency
   hashes, the eight-allocation cap, and the four same-ID administrative requeues rather than presenting only model metrics.

## Claim boundary and limitations

- This result establishes runtime qualification and an M0 baseline on four fixed SUN RGB-D camera-family rotations. It is
  not an external benchmark result.
- No candidate-quality, detached-scale benefit, camera-causality benefit, or universal-camera claim is supported because
  M1-M3 performed no optimizer step.
- Head wall time is a deliberately isolated batch-one cached-feature metric. Encoder-plus-head timing is secondary and
  neither measurement is a complete application deployment benchmark.
- The 12 latency allocations all used A100-SXM4-80GB GPUs. Conclusions may change with compilation mode, software stack,
  batch size, GPU generation, or integration into a persistent service.
- Three seeds within each held-out family measure optimization variability, not independent-scene population uncertainty.
- SUN camera family is confounded with scene/source composition. The rotations diagnose transfer but do not isolate a pure
  optical-intrinsics effect.
- DIODE is still fresh for this project. Nothing in this record may be described as DIODE performance.

## Recommended Phase 2g

The next stage should be an **implementation-only latency salvage**, registered before profiling. Do not change the model
quality hypothesis or open DIODE yet.

| Priority | Experiment | Frozen comparison | Promotion criterion |
|---|---|---|---|
| P0 | Precompute M2/M3 normalized-K features in the cache or CPU input pipeline. | Exact output parity with current eager camera summary. | Remove the approximately 0.717 ms camera-transform path without changing predictions. |
| P0 | Compare eager with `torch.compile(fullgraph=True, mode="reduce-overhead")` and explicit CUDA Graph capture. | Identical inputs, weights, output tolerances, warmup, blocks, and 12-allocation resampling unit. | Upper 95% CI passes a newly frozen runtime gate; record compile/capture amortization separately. |
| P0 | Fuse pooling, scale projection/head, and composition; profile the M3 coarse field separately. | M1 first, then add exact precomputed K, then field. | Identify a minimal qualified arm rather than hiding a slow component in the encoder total. |
| P1 | Freeze two latency views before results: head-relative and encoder-plus-head/system. | Report both, state which one governs qualification, and include absolute tail latency. | No post-result switching between estimands. |
| P1 | Only after runtime qualification, run the existing pilot and updated/stale/wrong/permuted-K controls. | Same gradient firewall, rotations, seeds, and selector. | Candidate passes runtime, reload, gradients, and identifiable camera controls before 60-epoch training. |
| P2 | Open DIODE exactly once only for one selected survivor versus M0. | Existing sealed archive identity and final gate. | Selector authorizes exactly one survivor; otherwise preserve the archive again. |

This direction is consistent with current systems guidance: PyTorch documents `reduce-overhead` mode as using CUDA graphs
to reduce Python overhead for small batches, while NVIDIA notes that networks made of many small kernels can become
enqueue-bound and benefit from CUDA Graphs. The modeling direction remains well grounded—Metric3D v2 uses canonical camera
space, CAM-Convs exposes calibration to the network, UniDepthV2 uses a compact camera representation, and Depth Pro
jointly estimates depth and focal length—but Phase 2f says the next uncertainty to reduce is execution efficiency, not
another unprofiled camera architecture.

Research and implementation references:

- [PyTorch `torch.compile`](https://docs.pytorch.org/docs/2.12/generated/torch.compile.html)
- [NVIDIA TensorRT performance best practices](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/optimization.html)
- [Metric3D v2](https://arxiv.org/abs/2404.15506)
- [CAM-Convs](https://arxiv.org/abs/1904.02028)
- [UniDepthV2](https://arxiv.org/abs/2502.20110)
- [Depth Pro](https://arxiv.org/abs/2410.02073)

The immediate stop condition is simple: if exact precomputation, compilation/capture, and fusion cannot produce a
candidate that passes the newly frozen implementation screen, retain M0 and redirect effort toward data/supervision rather
than spending formal-training or external-final budget.
