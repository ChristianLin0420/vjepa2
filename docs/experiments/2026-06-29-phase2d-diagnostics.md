# Phase 2d causal attribution, calibration audit, and latency confirmation — completed on Slurm

## Experiment metadata

| Field | Value |
|---|---|
| Status | complete; strict postflight passed with zero failures |
| Evidence level | post-hoc mechanism diagnostic and profiling confirmation; no fresh generalization evidence |
| Frozen protocol | [Phase 2d preregistration](2026-06-29-phase2d-diagnostics-preregistered.md) |
| Source result | [completed Phase 2c](2026-06-29-phase2c-cross-sequence.md), decision `retain_final_layer` |
| Execution commit / dirty | `160207418112bb18c8d6d1c4c6c8b7082ea8d114` / clean |
| Source Phase 2c commit | `9a8f8f0cb5fbe8aa55b609845d9204df885f0d95` |
| Source split hash | `e5fbb372b1858d1a1783b78a3bee8948d1a6f40a6926d689729437b99ed14862` |
| Dataset manifest SHA-256 | `9d3c892454b4036710b6604d278ad37fdcd6a8e5cee91ac8586411d8de213615` |
| Formal execution | `2026-06-29 09:55:53`–`10:08:09` PDT (`16:55:53`–`17:08:09` UTC) |
| Hardware | 12 distinct NVIDIA A100-SXM4-80GB devices for latency; A100 for attribution and calibration |
| Slurm partition | allocated on `polar4`; submission used the approved fallback list `polar4,polar3,polar,batch_block1,grizzly,batch_block2,batch_block3` |
| Test receipt | job `29595964`; `194 passed`, one warning; ruff, mypy, and finite CUDA stress passed |
| Claim boundary | post-hoc diagnostic only; Phase 2c decision remains `retain_final_layer` |

The final [aggregate JSON](../../outputs/jepa4d_phase2d/aggregate/phase2d_diagnostics.json) has status `complete`, the
[strict postflight](../../outputs/jepa4d_phase2d/aggregate/postflight.json) has status `pass`, and both contain an empty
`failures` list. The source identity, three selected checkpoints, train-only normalization, dataset split, public model
assets, test receipt, and Slurm dependency graph were hash-verified before aggregation.

## Executive result

| Frozen question | Result | Interpretation | Action |
|---|---|---|---|
| Did the learned gates themselves cause the Phase 2c quality change? | Original-minus-zero-gate raw AbsRel is exactly `-8.123016838607056e-05`; removing the gates changes prediction magnitude by only `0.001099514576101986` on average. | The direct gate effect is negligible. The Phase 2c learned-versus-final difference is more plausibly probe co-adaptation and validation checkpoint selection than useful residual mixing. | Stop treating learned gate weights as the main quality mechanism; keep final-only as the deployment default. |
| Is the cross-camera gap mainly scale? | Target-fitted per-sequence scalar reduces raw AbsRel by `47.989923250764255%`; per-image scalar by `61.61259501735166%`; bounded spatial scale by `78.2708710325924%`. | A large scale component exists, but the spatial oracle's further gain shows that one global camera scalar is not the whole problem. | Prioritize trainable scale/factorized geometry on fresh data, with all oracle results labeled diagnostic upper bounds. |
| Can the intrinsics explanation be tested causally? | Crop/resize K and FoV are internally consistent, but distortion, RGB-depth registration, and upstream depth-correction provenance are all `unknown_not_declared`. K controls were generated but not executed because the stored model is K-agnostic. | No camera-intrinsics causal claim is supported. | Make provenance fields mandatory and execute correct/wrong/shuffled-K controls only with a K-conditioned model. |
| Was Phase 2c's `1.1655×` latency ratio structural? | 12-job learned/final mean ratio `1.0226172872312669`; 95% cluster-bootstrap CI `[1.0219610402803287, 1.0233169519676384]`; label `within_1.10x`. | The original latency failure was not reproduced by the stronger protocol; the discrepancy is consistent with earlier measurement setup/noise. The confirmed end-to-end overhead here is about 2.26%. | Use the randomized/interleaved protocol for future profiling, but do not retroactively promote the Phase 2c candidate. |

The combined diagnosis is therefore: the fusion gate mechanism is not compelling, the representation still contains much
better relative geometry than metric depth, and scale transfer—not extra residual layers—is the highest-leverage modeling
problem. The latency confirmation removes efficiency as the dominant scientific concern, but Phase 2d is explicitly not
authorized to overturn the frozen Phase 2c decision.

## Experiment A — same-checkpoint fusion attribution

For every Phase 2c learned-fusion seed, this experiment reloaded the exact selected checkpoint, probe, normalization, and
validation-only uncertainty calibration. It changed only `fusion.raw_gates` and reevaluated the full 128-frame Freiburg-3
test set. In total it executed original, zero, fixed-average-equivalent, five non-identity layer permutations, and seven
non-empty sign-flip controls. The primary causal contrast is original versus zero; fixed-average-equivalent uses the learned
probe and must not be confused with the separately trained Phase 2c fixed-fusion baseline.

### Core control results

Values are exact mean ± sample standard deviation over checkpoint seeds 0, 1, and 2. Lower is better for all accuracy,
scale, and NLL columns. Prediction change is measured against the same-seed original-gate prediction.

| Same learned probe/checkpoint | Raw AbsRel ↓ | Aligned AbsRel ↓ | Abs log-scale error ↓ | Raw log-depth NLL ↓ | Calibrated NLL ↓ | Relative prediction change | Residual/final feature norm |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original learned gates | `0.41800999800519395 ± 0.022847705207569284` | `0.16046324259756753 ± 0.005867893997949621` | `0.5563621858102706 ± 0.04753663400539538` | `32.12822558482488 ± 27.383588639887375` | `2.149193468814095 ± 0.12139966298791986` | `0 ± 0` | `0.017162689783920843 ± 0.011988265844203733` |
| Zero gates | `0.41809122817358 ± 0.02293102667993177` | `0.16053597735784328 ± 0.006004639235839503` | `0.5565782315074544 ± 0.047789596581844195` | `32.22352588176727 ± 27.509940978373468` | `2.150864176141719 ± 0.11752836443764994` | `0.001099514576101986 ± 0.0007079418315572803` | `0 ± 0` |
| Fixed-average-equivalent gates | `0.40255242807324976 ± 0.022483681454275428` | `0.1714754190761596 ± 0.0018541442862430976` | `0.5066313853403395 ± 0.04620411804103959` | `18.8871595064799 ± 14.355666731521499` | `0.758327431200693 ± 0.28821576046935066` | `0.11458476384480794 ± 0.020971521186806483` | `0.7332897782325745 ± 0` |

Original gates beat zero gates by only `0.00008123016838607056` raw AbsRel, or `0.0194288%` relative to zero. Their
aligned difference is similarly negligible (`-0.00007273476027575`). The learned residual contribution is only 1.72% of
the final-feature norm on average and is seed-dependent. This is direct evidence against attributing the Phase 2c quality
gain to the learned gates themselves.

The fixed-average-equivalent intervention is different: it changes predictions by 11.46%, lowers raw AbsRel by
`0.01545756993194419` (`3.697895%`) versus the original learned gates, and improves scale error and NLL, but worsens aligned
AbsRel by `0.01101217647859207` (`6.862741%`). This suggests a substantial scale/co-adaptation effect. It is not evidence
that the separately trained fixed-fusion model should be promoted, because the probe and checkpoint selection are held at
the learned-fusion values here.

### Sequence split

Effective `(final, L2, L5, L8)` coefficients are seed 0 `(0.9829455561911523, +0.009727363420077673,
+0.013278220724177387, -0.005951140335407373)`, seed 1 `(0.9756817403518815, +0.009767619223273826,
+0.00888467134816987, +0.005665969076674867)`, and seed 2 `(1.0014664400304247, +0.002464449918086167,
-0.002919213096920376, -0.0010116768515904917)`. The tiny values and sign changes do not support a stable hierarchy.

Per-sequence values below are exact mean ± sample SD over the same three checkpoints.

| Intervention | Long office raw / aligned AbsRel ↓ | Structure/texture far raw / aligned AbsRel ↓ |
|---|---:|---:|
| Original | `0.3095560397487134 ± 0.014307761860726848` / `0.17836571290778616 ± 0.007520197467437306` | `0.5264639562616745 ± 0.038197307103109354` / `0.1425607722873489 ± 0.007917298425210826` |
| Zero | `0.30951327927565825 ± 0.014228151650042568` / `0.17822336168804517 ± 0.007661233586810762` | `0.5266691770715018 ± 0.03841923929542314` / `0.14284859302764139 ± 0.008253721240581553` |
| Fixed-average-equivalent | `0.327202647846813 ± 0.012510434089614853` / `0.21049672035345188 ± 0.004246188256218012` | `0.4779022082996865 ± 0.03536439134621741` / `0.13245411779886732 ± 0.0028096786764732575` |

Fixed-equivalent mixing trades worse long-office shape for much better far-sequence raw scale and aligned shape. That
sequence interaction is another reason not to collapse this diagnostic into a single pooled-frame claim. Original-gate
calibrated NLL also splits sharply: `0.10126031065980594` on long office versus `4.197126626968384` on structure/texture
far, so validation-only uncertainty calibration does not transfer uniformly across camera/scene families.

The full prediction handoff has shape `[9, 128, 518, 518]`: three controls for all three seeds; four deterministic,
sequence-balanced examples are stored separately for bounded qualitative inspection.

## Experiment B — calibration, scale-oracle, and camera audit

The oracle ladder consumes the exact full Phase 2c predictions. Every correction is fitted to test targets and is therefore
a mechanism probe/upper bound, not a deployable method or an unbiased estimate of future performance. Aggregate values are
the mean across the three original learned-gate checkpoint predictions.

| Diagnostic correction | Raw AbsRel ↓ | Aligned AbsRel ↓ | Abs log-scale error ↓ | Raw AbsRel reduction from uncorrected |
|---|---:|---:|---:|---:|
| Uncorrected | `0.41800999681828865` | `0.16046319034659193` | `0.5563617207727461` | — |
| Target-fitted per-sequence scalar | `0.21740732016466982` | `0.16046319034659193` | `0.13756717022850395` | `47.989923250764255%` |
| Target-fitted per-image scalar | `0.16046319034659193` | `0.16046319034659193` | `9.830099697201907e-18` | `61.61259501735166%` |
| Target-fitted per-sequence affine | `0.23118775614634549` | `0.18056017962831325` | `0.1303134934118606` | `44.69324707398227%` |
| Bounded target-fitted per-image low-resolution spatial scale | `0.09082993130530233` | `0.09247367887870434` | `0.020417758260152703` | `78.2708710325924%` |

The per-sequence scalar nearly halves raw error without changing aligned shape, confirming a large family/sequence scale
component. Per-sequence affine is worse than scalar in both raw and aligned AbsRel; a simple offset does not solve the
problem. The spatial oracle's additional improvement shows meaningful local structure remains after global rescaling.
Because every oracle sees the target, none of these numbers may be compared to a deployable model as though it had access
to the same information.

### Crop/resize intrinsics audit

All five sequence selections have matching 480×640 RGB/depth dimensions. The center-square crop is `[x=80, y=0,
w=480, h=480]`, and resize uses the half-pixel / `align_corners=False` convention. Identical camera-family rows are grouped
below; K is shown as `(fx, fy, cx, cy)`.

| Camera family / sequences | 384×384 transformed K | 518×518 transformed K | Horizontal / vertical FoV |
|---|---|---|---|
| Freiburg-1: `xyz`, `floor` | `(413.84, 413.20000000000005, 190.78000000000003, 204.14000000000001)` | `(558.2529166666666, 557.3895833333333, 257.52875, 275.55083333333334)` | `49.777533292958196° / 49.811698448042165°` |
| Freiburg-2: `xyz` | `(416.72, 416.8, 195.98000000000005, 199.66)` | `(562.1379166666666, 562.2458333333333, 264.54333333333335, 269.5075)` | `49.4707740761982° / 49.45280023367454°` |
| Freiburg-3: both test sequences | `(428.32, 431.36000000000007, 191.98000000000005, 197.98000000000002)` | `(577.7858333333332, 581.8866666666667, 259.1475, 267.24125)` | `48.28981010393426° / 47.98004430482398°` |

The arithmetic audit supports consistent K propagation through the implemented crop/resize. It does not validate the
physical camera model: distortion model/coefficients, undistortion state, RGB-depth registration, upstream depth correction,
and duplicate-correction status are all `unknown_not_declared`. The loader converts uint16 depth to float32 and divides once
by the PNG integer divisor `5000`, but that divisor is storage semantics, not proof that device-specific correction is right.

Correct-K, wrong-K, and shuffled-K control tensors were generated, but status is `generated_not_executed` with evaluation
`not_executed_no_K_conditioned_model_callback`. Stored Phase 2c predictions do not depend on K, so executing those controls
would be degenerate. No camera-intrinsics causal effect is claimed.

## Experiment C — independent latency confirmation

Each of 12 independent Slurm allocations used a distinct A100 GPU UUID, 30 warmups per path, 30 randomized path-order
blocks, and 100 serial batch-one iterations per block. The aggregate contains 1,440 end-to-end blocks and 1,800 head-only
blocks. The confidence interval resamples the independent Slurm job, not inner timing iterations.

### Frozen confirmation result

| Statistic | Exact value |
|---|---:|
| Mean learned/final paired wall-time ratio | `1.0226172872312669` |
| Median paired ratio | `1.0224870423556087` |
| 95% cluster-bootstrap lower bound | `1.0219610402803287` |
| 95% cluster-bootstrap upper bound | `1.0233169519676384` |
| Frozen label | `within_1.10x` |

### Aggregate path timing

| Path | Wall p50 ms | Wall p90 ms | Wall p95 ms | CUDA p50 ms |
|---|---:|---:|---:|---:|
| Final deployment | `25.90609585` | `26.259251997999996` | `26.4631416485` | `25.905797119140626` |
| Final probe with all layers captured | `26.143450889999997` | `26.480567127` | `26.670995079999997` | `26.143167724609377` |
| Fixed deployment | `26.188928325` | `26.541112187` | `26.730515669` | `26.188608398437502` |
| Learned deployment | `26.49150386` | `26.82122429` | `26.998797879500003` | `26.491182861328127` |

Head-only wall p50/p95 ms are final `0.26703915/0.2723193495`, fixed `0.282794835/0.2878193355`, learned
`0.52467475/0.53609137`, zero-gate `0.523351975/0.5337990105`, and fixed-equivalent
`0.523305175/0.5335526095`. The learned head is about `1.965×` final but only about `0.258` ms slower; encoder work
dominates, so capture-all and full learned deployment add roughly `0.92%` and `2.26%` end to end.

### Independent job-level evidence

Replicates 0–11 used jobs `29595967,29595968,29595969,29595970,29595971,29595972,29595973,29595974,29595975,29595976,29595977,29595979`. W&B runs: [`sea5ynxq`](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/sea5ynxq), [`zta8qcdh`](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/zta8qcdh), [`gghflkiu`](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/gghflkiu), [`r4o2sna4`](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/r4o2sna4), [`64hisxiv`](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/64hisxiv), [`wnh2zabf`](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/wnh2zabf), [`yigtnu9p`](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/yigtnu9p), [`votfbcdo`](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/votfbcdo), [`y0kvk0tu`](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/y0kvk0tu), [`pwhza5qh`](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/pwhza5qh), [`t3avdni9`](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/t3avdni9), [`p2nxazn0`](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/p2nxazn0). Raw files: [latency directory](../../outputs/jepa4d_phase2d/latency).

All 12 jobs have unique job IDs, unique GPU UUIDs, online W&B runs, local raw block schedules, CUDA/wall timing, HTML,
and GPU telemetry. Aggregate peak CUDA allocation/reservation is `0.41034841537475586` / `0.427734375` GiB. Telemetry
averages are `51.622283798576895%` GPU utilization, `837.7063149972632` MiB used memory, `173.84874274767378` W,
and `1246.4634646962234` MHz SM clock; maximum temperature is `67°C`.

This confirmation changes the understanding of the old latency measurement, not the old decision. Phase 2c still retains
the final layer because its preregistered gate was evaluated with its own frozen measurement and Phase 2d was frozen as
non-retroactive.

## Slurm execution and validation

The login node handled environment/assets/hashes/submission; tests and all real-model work ran in Slurm. Test `29595964`, attribution `29595965`, calibration `29595966`, latency replicas listed above, latency aggregate `29595980`, and final aggregate `29595994` all completed `0:0`. The [test receipt](../../outputs/phase2d-gates/tests.json) SHA-256 is `ddf7c38db6ebff8d89d3d39f7e3086e96553e79ad095342dd70e6533ccefe26d`; [dependency graph](../../outputs/phase2d-gates/dependency-graph.json) SHA-256 is `a792f8adcdac16d7e7857006b89040bc8a521e1589bc7f8e4352fe55e8ea6696`. Final audit checked 318 references and 268 unique hashes with zero failures.

## W&B runs and immutable artifacts

Every scientific GPU stage used online mode; credentials are absent from commands, receipts, and artifacts.

| Stage | Online run | Uploaded artifact | Backend digest |
|---|---|---|---|
| Attribution | [`tz3dy9vb`](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/tz3dy9vb) | `crlc112358/jepa4d-worldmodel/tz3dy9vb-phase2d-attribution:v0` | `3082a369b6957a0f3c3e0e9f96f35164` |
| Calibration/scale | [`4zpfqgqc`](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/4zpfqgqc) | `crlc112358/jepa4d-worldmodel/4zpfqgqc-phase2d-calibration:v0` | `7bca10447ebb337895051aaeac42de22` |
| Latency aggregate | [`313jtoyi`](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/313jtoyi) | `crlc112358/jepa4d-worldmodel/313jtoyi-phase2d-latency-aggregate:v0` | `3383a3d1115d81f6336185fb584239f1` |
| Final aggregate | [`q1m52wi1`](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/q1m52wi1) | `crlc112358/jepa4d-worldmodel/q1m52wi1-phase2d-diagnostics:v0` | `0d7caad363ad1a703b312409f50b3123` |

Each latency run in the job-level table also uploaded an immutable `phase2d-latency:v0` artifact. Exact qualified names,
digests, upload modes, and file hashes are in the adjacent per-replicate `wandb_receipt.json` files.

## Durable artifacts

| Stage | Canonical artifact and SHA-256 | Supporting files |
|---|---|---|
| Final | [JSON](../../outputs/jepa4d_phase2d/aggregate/phase2d_diagnostics.json), `d50c1dd7cca3ad902eb3913b035b863dbe8723af4d9cf5323b2859d06769a911` | [HTML](../../outputs/jepa4d_phase2d/aggregate/phase2d_diagnostics_report.html), [postflight](../../outputs/jepa4d_phase2d/aggregate/postflight.json), [W&B receipt](../../outputs/jepa4d_phase2d/aggregate/wandb_receipt.json) |
| Attribution | [JSON](../../outputs/jepa4d_phase2d/fusion-attribution/fusion_attribution.json), `ab15d2e68428f17667e3e476464d19239e117b85ebb1bdb3ff082bd2d7e460c8` | [HTML](../../outputs/jepa4d_phase2d/fusion-attribution/fusion_attribution_report.html), [predictions](../../outputs/jepa4d_phase2d/fusion-attribution/full_predictions.npz), [qualitative](../../outputs/jepa4d_phase2d/fusion-attribution/qualitative_examples.npz), [source identity](../../outputs/jepa4d_phase2d/fusion-attribution/source_identity.json), [receipt](../../outputs/jepa4d_phase2d/fusion-attribution/receipt.json), [W&B receipt](../../outputs/jepa4d_phase2d/fusion-attribution/wandb_receipt.json) |
| Calibration | [JSON](../../outputs/jepa4d_phase2d/calibration-scale-audit/phase2d_calibration_scale_audit.json), `c2add1194ba7de4c6456ad9a2455102e5b84eab6f332f66a3826d76108aa8243` | [HTML](../../outputs/jepa4d_phase2d/calibration-scale-audit/phase2d_calibration_scale_audit.html), [camera table](../../outputs/jepa4d_phase2d/calibration-scale-audit/phase2d_calibration_table.csv), [oracle table](../../outputs/jepa4d_phase2d/calibration-scale-audit/phase2d_oracle_summary.csv), [W&B receipt](../../outputs/jepa4d_phase2d/calibration-scale-audit/wandb_receipt.json) |
| Latency | [JSON](../../outputs/jepa4d_phase2d/latency-aggregate/latency_aggregate.json), `78e03b15d3362be8d9656093ede0032f525cd096a61b3ddab45c72ae6e9f53eb` | [HTML](../../outputs/jepa4d_phase2d/latency-aggregate/latency_aggregate_report.html), [postflight](../../outputs/jepa4d_phase2d/latency-aggregate/postflight.json), [W&B receipt](../../outputs/jepa4d_phase2d/latency-aggregate/wandb_receipt.json) |

## Failed attempts and clean supersession

These retries are relevant because they demonstrate that infrastructure failures were caught before scientific output and
that the accepted result was recomputed from a new test receipt rather than resumed from partial state.

| Attempt | Commit / jobs | Failure | Disposition |
|---|---|---|---|
| 1 | `291c47b7d0870d61e65b62fdb4975edf260b48f2`; test `29594930`, attribution `29594931`, latency `29594933`, `29594937`–`29594943`, `29594945`, `29594947`–`29594949` | Test job passed (`189 passed`, one skipped), but the validator treated the approved comma-separated partition fallback list as one literal partition. Attribution and all 12 latency jobs failed before model inference; dependents were canceled. | Commit `e5beb8e079f900483ee702bf45b281c4f6b281cc` fixed split/validation of the approved list. No checkpoints, W&B results, or inference artifacts were reused. |
| 2 | `e5beb8e079f900483ee702bf45b281c4f6b281cc`; test `29595898` | The real-checkpoint test found an inconsistent asset pair: a root model directory existed while its matching implementation did not. Result: `193 passed`, one failed. All downstream work was canceled. | Commit `160207418112bb18c8d6d1c4c6c8b7082ea8d114` selected the matched `phase2b_assets` model and implementation. No inference or W&B run had started. |
| 3 | `160207418112bb18c8d6d1c4c6c8b7082ea8d114`; jobs listed above | No scientific or infrastructure failures. | Fresh test receipt, outputs, 16 online W&B runs, aggregate, and strict postflight accepted. |

## Claim boundary and limitations

- Supported: for the exact three consumed Phase 2c learned-fusion checkpoints on the two named Freiburg-3 recordings,
  zeroing learned gates has a negligible direct effect, while fixed-equivalent mixing changes scale and shape materially.
- Supported: target-fitted diagnostics reveal a large global scale component and an additional spatially varying component.
  The oracle values are upper bounds and are not deployable performance.
- Supported: under 12 independent randomized/interleaved A100 allocations, the learned/final latency ratio has a
  job-clustered 95% CI of exactly `[1.0219610402803287, 1.0233169519676384]` and satisfies the frozen `1.10×` label.
- Not supported: a new generalization claim, population-level statistical significance, learned fusion promotion, causal
  camera-intrinsics effects, separation of camera family from scene content, or correctness of undeclared distortion,
  registration, and depth-correction metadata.
- The 128 Freiburg-3 frames are temporally correlated and were already consumed. Three checkpoint seeds describe
  optimization variation, not independent scene variation; frame-level intervals would be pseudoreplication.
- The latency jobs used unique GPU UUIDs, but several ran concurrently on shared nodes. The job-clustered interval is the
  frozen analysis for this cluster execution; it is not a universal A100 latency guarantee and may not capture node-level
  common shocks.
- The current repository HEAD may be newer, but all accepted numbers and hashes above belong to clean execution commit
  `160207418112bb18c8d6d1c4c6c8b7082ea8d114`.

## Actionable conclusions

| Priority | Next action | Why this follows from Phase 2d | Evidence required before changing the default |
|---|---|---|---|
| P0 | Keep `vjepa_final` as the deployment/default representation. | Learned gates contribute only `-8.123016838607056e-05` AbsRel and unstable, tiny residual coefficients. | A fresh, preregistered multi-sequence result showing a repeatable quality gain from the mechanism itself. |
| P0 | Shift modeling effort from gate search to scale-aware/factorized geometry. | A per-sequence scalar removes 47.99% of raw error, while spatial correction removes 78.27%; scale and local geometry both matter. | Train-only or validation-only estimation that improves raw scale, aligned shape, NLL, and coverage on fresh camera families without target fitting. |
| P0 | Use fresh independent sequences/camera families for the next quality decision. | Freiburg-3 is consumed, temporally correlated, and confounds scene with camera family. | Sequence-level macro metrics and uncertainty over genuinely independent recordings/datasets. |
| P1 | Make distortion, undistortion, registration, and depth-correction provenance mandatory manifest fields. | Arithmetic K propagation is correct, but physical camera/depth provenance is incomplete. | Fully declared metadata plus auditable loader transformations. |
| P1 | Add K conditioning only together with executable correct/wrong/shuffled-K controls. | The current K-agnostic predictions make the negative controls degenerate. | A preregistered causal effect on fresh cameras, with shuffled/wrong-K degradation and no leakage. |
| P1 | Standardize future efficiency decisions on this 12-job randomized/interleaved protocol. | It resolves the noisy Phase 2c ratio to a tight 2.20%–2.33% overhead interval. | Job-clustered CI, raw schedules, unique GPU UUIDs, telemetry, and local plus online artifacts. |

The immediate decision is therefore conservative but focused: retain final-only, treat Phase 2d as a completed diagnostic,
and spend the next experimental budget on independently testable scale transfer and camera-aware geometry rather than more
fine-grained residual-gate tuning.
