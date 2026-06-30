# JEPA-4D metric guide

This document is the common interpretation guide for JEPA-4D experiments. It defines what each reported metric measures,
how the repository computes it, why the metric is needed, what direction is better, and which claims it cannot support by
itself. Experiment-specific preregistrations remain authoritative when they intentionally narrow a definition, split,
aggregation rule, or decision gate.

The most important rule is that a scalar is not self-explanatory. Every result must be read together with:

1. the dataset and immutable split;
2. the valid-sample or valid-pixel rule;
3. whether ground-truth alignment was applied;
4. whether calibration was fitted, and on which split;
5. the aggregation unit: pixel, frame, sequence, camera family, scene, or episode;
6. the uncertainty unit: optimization seed, independent scene, family, or episode;
7. the baseline and direction of improvement;
8. the evidence level and claim boundary.

For example, median-aligned AbsRel measures relative shape after using target depth to remove global scale. It cannot be
presented as metric-scale performance. Likewise, three optimizer seeds on one camera family are not three independent
cameras or three population samples.

## 1. Evidence levels before metrics

| Evidence level | What was executed | What the metrics may establish |
|---|---|---|
| `contract-only` | Deterministic mock or controlled fixture | Schemas, invariants, control flow, persistence, or observability work as implemented |
| `integration` | Real model on a small unscored input | Components interoperate and outputs are inspectable |
| `sequence-level` | Named real sequence with a frozen protocol | The measured behavior holds for that sequence and operating point |
| `benchmark` | Versioned data, split, metrics, baselines, and held-out evaluation | Comparative quality under that exact protocol |
| `training` | Reproducible optimization, validation selection, curves, and checkpoints | Optimization behavior and checkpoint-selection evidence |
| `closed-loop` | Repeated actions, verification, failures, safety, and recovery | System task behavior in the named environment |

A perfect contract-only score is not a model-quality result. A benchmark metric on one camera family is not universal
generalization. A technically clean pipeline can also produce a scientifically negative result.

## 2. Notation and Phase 2 evaluation unit

For one valid pixel `i`:

- `d_i > 0` is target metric depth in metres;
- `l_i` is predicted log depth;
- `d_hat_i = exp(l_i)` is predicted metric depth;
- `v_i` is predicted log variance in log-depth space;
- `r_i = l_i - log(d_i)` is the log-depth residual.

Phase 2f requires finite predictions and at least one valid reduced target pixel per frame. Its SUN cache starts from finite
target depth in `0.1 < d < 10.0` metres, rejects source frames with fewer than 100 full-resolution valid pixels, and reduces
validity to 24x24 with mask-weighted area interpolation. When the shared metric function is called without an explicit
mask, its fallback is finite, strictly positive target depth. The TUM teacher protocol also uses `0.1 < d < 10.0` metres,
requires a positive prediction, and requires at least 100 evaluated valid pixels. The masks and spatial grids still differ,
so numbers from those protocols must not be pooled as if their denominators were identical.

Phase 2f computes each primary metric per frame. It then provides two aggregates:

- `frame_macro`: every frame has equal weight;
- `group_macro`: average frames within each camera family/domain first, then give every family/domain equal weight.

`group_macro` is the primary cross-family quantity. It prevents a large family from dominating merely because it has more
frames or valid pixels.

## 3. Metric-depth and shape metrics

### 3.1 Raw absolute relative error

```text
Raw AbsRel = mean_i(|d_hat_i - d_i| / d_i)
```

| Property | Interpretation |
|---|---|
| Direction | Lower is better; zero is exact |
| Unit | Dimensionless relative error |
| Measures | Combined relative-shape and absolute-scale quality |
| Sensitive to | Global scale bias, local shape error, near-depth errors, domain shift |
| Does not reveal alone | Whether failure came from scale or shape |

Raw AbsRel is the primary metric-depth quantity in Phases 2b-2f because robot geometry needs distances in the scene's
physical coordinate system. It must always be paired with aligned error and scale error to diagnose the mechanism.

### 3.2 Median-aligned absolute relative error

Phase 2f computes a robust per-frame log-scale offset:

```text
s = median_i(l_i - log(d_i))
d_aligned_i = exp(l_i - s)
Aligned AbsRel = mean_i(|d_aligned_i - d_i| / d_i)
```

| Property | Interpretation |
|---|---|
| Direction | Lower is better |
| Measures | Relative scene shape after removing one target-derived multiplicative scale per frame |
| Robustness | Median alignment has a 50% breakdown point for the scalar offset |
| Claim boundary | Diagnostic shape quality only; not metric-scale accuracy |

Earlier phases use a related but non-identical alignment. Phases 2b-2e compute
`a = median(target_depth) / median(predicted_depth)` and align as `a * predicted_depth`; their absolute log-scale error is
`|log(a)|`. The Phase 2f median of pixelwise log residuals is not generally equal to the log ratio of depth medians.
Compare candidates within one phase and definition; do not pool aligned or scale values across this boundary.

Reading raw and aligned AbsRel together:

| Raw AbsRel | Aligned AbsRel | Likely interpretation |
|---|---|---|
| Poor | Good | Shape is useful; global metric scale is wrong |
| Poor | Poor | Shape/domain transfer is also wrong |
| Better | Better | Genuine combined geometry improvement is plausible |
| Better | Worse | Scale compensation may be hiding a shape regression |

### 3.3 Signed and absolute log-scale error

```text
Signed log-scale error   = s
Absolute log-scale error = |s|
```

| Metric | Direction | Meaning |
|---|---:|---|
| Signed log-scale error | Toward zero | Positive means predicted depths are globally too large/far; negative means too small/near |
| Absolute log-scale error | Lower | Magnitude of the global multiplicative scale mistake |

The corresponding multiplicative factor is `exp(s)`. For intuition, an absolute log-scale error of `0.1` corresponds to
about a `1.105x` scale factor. Absolute error is suitable for ranking; the signed distribution is required to diagnose
systematic near/far bias.

### 3.4 RMSE and log RMSE

```text
RMSE     = sqrt(mean_i((d_hat_i - d_i)^2))
Log RMSE = sqrt(mean_i((log(d_hat_i) - log(d_i))^2))
```

| Metric | Unit | Direction | Why report it |
|---|---|---:|---|
| Raw RMSE | Metres | Lower | Strongly penalizes large physical-distance errors |
| Aligned RMSE | Metres | Lower | Large shape errors after global scale removal |
| Log RMSE | Log-depth | Lower | Treats multiplicative errors more symmetrically across depth range |

RMSE is outlier-sensitive by design. It complements AbsRel; neither should silently replace the other after results are
observed.

### 3.5 Delta accuracy

For each pixel define `q_i = max(d_hat_i / d_i, d_i / d_hat_i)`. Then:

```text
Delta-k = fraction_i(q_i < 1.25^k),  k in {1,2,3}
```

| Metric | Direction | Meaning |
|---|---:|---|
| Delta-1 | Higher | Fraction within a factor of 1.25 |
| Delta-2 | Higher | Fraction within a factor of 1.25 squared |
| Delta-3 | Higher | Fraction within a factor of 1.25 cubed |

Delta metrics are intuitive threshold accuracies but can hide the magnitude of errors outside the threshold. They remain
secondary to continuous metrics.

### 3.6 Valid coverage

Every depth table must report frames, valid-pixel counts, and failures. A lower error computed after dropping difficult
pixels is not an improvement. Phase 2f fails when any evaluated frame has zero valid pixels or any prediction is non-finite.

## 4. Point and camera-pose metrics

### 4.1 Aligned point error

Depth and intrinsics are back-projected into 3D. The TUM protocol reports:

| Metric | Direction | Meaning |
|---|---:|---|
| Mean aligned point error | Lower | Mean Euclidean error of corresponding back-projected points in metres |
| Median aligned point error | Lower | Robust typical point error |
| 5 cm threshold accuracy | Higher | Fraction of corresponding points within 5 cm |
| 10 cm threshold accuracy | Higher | Fraction within 10 cm |

The current historical JSON keys call the two threshold accuracies `point_fscore_5cm_aligned` and
`point_fscore_10cm_aligned`. The implementation uses known pixel correspondences and computes a within-threshold fraction;
it is not the symmetric accuracy/completeness F-score used by some unordered point-cloud benchmarks. Reports must describe
the implemented quantity, not rely on the key name alone.

### 4.2 Sim(3)-aligned pose

VGGT camera trajectories are aligned to the reference with an Umeyama similarity transform before pose scoring.

| Metric | Unit | Direction | Meaning |
|---|---|---:|---|
| ATE RMSE after Sim(3) | Metres | Lower | Root-mean-square global camera-position error after rotation, translation, and scale alignment |
| ATE mean after Sim(3) | Metres | Lower | Mean aligned camera-position error |
| Rotation geodesic error | Degrees | Lower | Angular distance between aligned predicted and target rotations |
| Relative translation error | Metres | Lower | Error in consecutive aligned camera displacement |
| Alignment scale | Ratio | Diagnostic | Scale required to align predicted trajectory to target |

Sim(3)-aligned pose measures trajectory structure, not recovered metric trajectory scale. The alignment scale must remain
visible so a large scale correction cannot be mistaken for metric localization.

## 5. Uncertainty and calibration metrics

Accuracy asks whether the mean prediction is correct. Uncertainty asks whether the model knows when and by how much it may
be wrong. Ranking and calibration are different estimands.

### 5.1 Validation-only variance multiplier

Phase 2f fits one positive scalar `c` on the registered validation family:

```text
c_raw = mean_i(r_i^2 / exp(v_i))
c = clip(c_raw, 1e-3, 1e3)
v_calibrated_i = v_i + log(c)
```

The multiplier changes uncertainty magnitude, not point predictions or uncertainty ordering. It must be fitted on
validation only and frozen before development-test or external evaluation. Fitting it on test labels is leakage.
The multiplier fit pools validation pixels, whereas Phase 2f's reported NLL first averages within frames and then within
groups. Those are deliberately different weighting estimands and should both remain documented.

| Value | Interpretation |
|---|---|
| `c > 1` | Raw predictive variance was generally too small / overconfident |
| `c < 1` | Raw predictive variance was generally too large / underconfident |
| `c` near a clipping limit | Calibration is extreme and should be treated as a warning |

### 5.2 Gaussian negative log-likelihood

Phase 2f evaluates a Gaussian in log-depth space and omits the constant `0.5 log(2*pi)`:

```text
NLL_i = 0.5 * (r_i^2 / exp(v_calibrated_i) + v_calibrated_i)
NLL = mean_i(NLL_i)
```

| Property | Interpretation |
|---|---|
| Direction | Lower is better |
| Rewards | Small residuals with appropriately small variance |
| Penalizes | Confident errors and unnecessarily broad uncertainty |
| Important caveat | Values can be negative under this convention; compare only identical units and formulas |

NLL is not an accuracy-only metric. A model can improve AbsRel but worsen NLL if its confidence becomes less trustworthy.

### 5.3 Empirical interval coverage

For nominal levels 50%, 80%, 90%, and 95%, Phase 2f checks whether the absolute log residual lies within the corresponding
normal interval `z * sqrt(exp(v_calibrated))`.

| Observation | Meaning |
|---|---|
| Empirical coverage close to nominal | Uncertainty magnitude is approximately calibrated at that level |
| Empirical below nominal | Overconfident intervals |
| Empirical above nominal | Underconfident / excessively wide intervals |

Coverage should be shown as a curve or table, not reduced to one favorable level. Phase 2e also reports a reliability
error that summarizes nominal-versus-empirical gaps; lower is better. The current Phase 2f evaluator emits the four pooled
coverage points but does not emit a scalar reliability error or an empirical regression-calibration curve. A future report
must compute those explicitly under a new schema rather than imply that they already exist.

### 5.4 Risk-coverage and AUSE

Pixels are ordered from least to most uncertain. At each retained coverage, risk is the cumulative mean raw AbsRel of the
retained pixels. An oracle orders pixels by true error. Phase 2f computes:

```text
AUSE = integral_coverage(predicted_risk - oracle_risk)
```

| Property | Interpretation |
|---|---|
| Direction | Lower is better; zero is oracle ordering |
| Measures | Whether uncertainty ranks risky pixels correctly |
| Invariant to | A single positive variance multiplier, because ordering is unchanged |
| Does not establish | Correct probability magnitude or calibrated intervals |

Phase 2f stores a per-frame AUSE used by frame/family macros and a pooled `risk_coverage.pixel_ause` diagnostic. These have
different weighting and must be labeled. Better AUSE with worse NLL means ranking improved while calibration worsened.

### 5.5 Confidence-error correlation

Some earlier geometry reports correlate confidence with true error. When confidence increases as predicted variance
decreases, a negative correlation is desirable: higher confidence should accompany lower error. Correlation alone does not
measure calibration and can be distorted by outliers.

## 6. Factorized training objectives and diagnostics

M1-M3 separate a centered shape branch from global metric scale. M3 adds a bounded spatial scale field. The objectives are
training signals; final model selection still uses held-out metrics.

| Component | Frozen weight | Meaning |
|---|---:|---|
| Centered shape Smooth-L1 | 1.00 | Fit target log-depth after removing its valid-pixel median |
| Shape-gradient Smooth-L1 | 0.25 | Match valid horizontal and vertical depth differences |
| Shape Gaussian NLL | 0.10 | Train dense shape variance |
| Global-scale Smooth-L1 | 1.00 | Fit robust optimal log scale from detached shape |
| Global-scale Gaussian NLL | 0.10 | Train global-scale variance |
| Paired-view scale consistency | 0.10 | Predict consistent scale for two views of one source sample |
| M3 scale-field fit | 0.25 | Fit detached, mean-centered residual scale structure |
| M3 field total variation | 0.01 | Discourage spatially noisy correction fields |

The robust scale target is:

```text
s_star = median_valid(log(d) - stopgrad(centered_shape))
```

It is detached so the scale objective cannot reshape the geometry branch merely to make scale fitting easier.
For M1-M3, composed predictive variance is the sum of dense shape variance and global-scale variance in variance space.
M3 does not predict a separate uncertainty term for its spatial field, so field-specific uncertainty is not identifiable
from the current output.

### Gradient firewall metrics

| Metric | Required behavior | Why it exists |
|---|---|---|
| Allowed gradient norm | Finite and nonzero when the corresponding objective is active | Proves the intended branch can learn |
| Forbidden gradient norm | Bitwise zero | Proves shape and scale/field objectives do not cross-update forbidden parameters |
| Maximum forbidden norm | Exactly zero over the run | A single violation invalidates the claimed factorization |

### Optimization health

| Diagnostic | Healthy interpretation |
|---|---|
| Total and component losses | Finite; broadly decrease without one term silently dominating |
| Model change from initialization | At least one owned trainable tensor changes |
| Gradient clipping | Occasional use is acceptable; persistent clipping suggests instability |
| Validation raw/scale curves | Improve before checkpoint selection; development-test must not select the checkpoint |
| Exact checkpoint reload | State tensors and fixed-batch outputs are bitwise identical |
| Best epoch | Lowest validation raw AbsRel, then lower validation scale error, then earlier epoch |

The joint composed-depth NLL in Phase 2f training is diagnostic only; it does not bypass the gradient firewall.

## 7. Camera-causality metrics

Camera conditioning requires a treatment that actually changes the intrinsics. Phase 2f constructs eight deterministic
crop/resize/pad profiles per source and evaluates four K conditions without retraining:

| Condition | Construction | Question |
|---|---|---|
| `updated` | Analytically transform K with the image | Does the model receive geometrically correct calibration? |
| `stale` | Keep the pre-transform K | Does failure to update K hurt? |
| `wrong` | Scale focal lengths by 1.25 and shift principal point by `(+38.4,-38.4)` pixels | Is the model sensitive to materially incorrect calibration? |
| `permuted` | Apply fixed within-source profile derangement | Does the correct image/K pairing matter? |

Protocol-validity metrics:

| Metric | Required behavior |
|---|---|
| Distinct updated K count | Exactly eight per source |
| Permutation bijection | Every profile appears exactly once |
| Assignment-change fraction | 1.0 in Phase 2f |
| Numerical matrix-change fraction | 1.0 in Phase 2f |
| Mean absolute output delta | Greater than `1e-6` metres for each negative control |

Quality causality requires trained M2/M3 `updated` predictions to beat stale, wrong, and permuted K under a frozen
family-macro metric. A nonzero output delta establishes sensitivity, not usefulness. Conversely, shuffling identical K
matrices is an identity operation and supports no camera claim.

Useful same-checkpoint interventions for the next stage include:

- structural confirmation that M0/M1 declare `consumes_intrinsics=false` and reject any attempt to pass K, as the negative
  control for non-camera arms;
- M2/M3 updated-versus-controlled K comparisons;
- M3 full field versus a zeroed field;
- learned global scale versus a fixed train-median scale.

## 8. Representation metrics

Current Phase 1 smoke metrics validate contracts rather than representation quality:

| Metric | Direction | Meaning / boundary |
|---|---:|---|
| Finite-token fraction | 1.0 required | Detect NaN/Inf in produced features |
| Minimum feature standard deviation | Positive | Reject completely collapsed constant features |
| Modes completed | All required | Single-image, multi-view, and video interfaces execute |
| Adjacent-token cosine | Context-dependent | Temporal feature similarity; high can mean consistency or collapse |
| Cycle consistency / occlusion recovery | Higher | Future correspondence-quality evidence on labeled data |

Model-quality representation evaluation needs task labels: top-k/mean-class accuracy, retrieval Recall@K or mAP, frozen
dense probes, temporal anticipation, and correspondence metrics. Finite noncollapsed tokens alone do not prove semantics.

## 9. Object grounding and identity metrics

### Object grounding

| Metric | Direction | Meaning |
|---|---:|---|
| Box AP / recall | Higher | Detection and phrase-grounding quality at defined IoU thresholds |
| Mask IoU / boundary quality | Higher | Segmentation overlap and edge accuracy |
| Valid-mask fraction | 1.0 expected in smoke tests | Whether observations carry a nonempty mask; contract-only |
| Slot count | Dataset-dependent | Number of persistent object hypotheses; not an accuracy metric alone |
| Pose/centroid error | Lower | Geometric localization of a grounded object |
| Confidence NLL/ECE | Lower | Calibration of object existence, state, or localization confidence |

The current grounding smoke `association_recall` is derived from fixture track length. It is an API regression quantity,
not labeled detector/tracker recall.

### Persistent identity

Identity metrics operate on pairs of observations and track histories:

| Metric | Direction | Meaning |
|---|---:|---|
| Pairwise precision | Higher | Of observation pairs merged into one predicted identity, fraction truly same |
| Pairwise recall | Higher | Of true same-identity pairs, fraction kept together |
| Pairwise F1 | Higher | Harmonic mean of pairwise precision and recall |
| ID switches | Lower | Consecutive observations of one true identity assigned different tracks |
| Fragments | Lower | Extra predicted track IDs used for one true identity |
| False merges | Lower | Predicted tracks containing multiple true identities |
| Track survival | Higher | Dominant predicted-track fraction over each true identity's lifespan |
| Predicted tracks | Diagnostic | Over-fragmentation or over-merging context |

Ground-truth boxes/masks isolate association quality; they do not measure detector or segmenter quality. A correct object
category with the wrong persistent ID remains an identity failure.

## 10. Memory metrics

Current memory smoke metrics are deterministic persistence invariants:

| Metric | Direction | Meaning / boundary |
|---|---:|---|
| History recall | 1.0 | Expected updates remain in object history |
| Observation-reference recall | 1.0 | Evidence references are retained |
| Query recall | 1.0 on fixture | Named object remains retrievable |
| Reload parity | 1.0 | Snapshot reload equals in-memory state |
| Replay parity | 1.0 | Append-only event replay reconstructs the same state |
| Query latency | Lower | Lookup time for the named fixture query |

Future memory-quality evaluation must add localization/relation accuracy, temporal QA, last-seen retrieval, identity
survival, database growth, replay throughput, latency percentiles, and task accuracy versus compression. Perfect fixture
parity does not establish open-world memory quality.

## 11. Planning and closed-loop metrics

| Metric | Direction | Meaning |
|---|---:|---|
| Task success | Higher | All required task conditions were verified |
| Normalized subgoal progress | Higher | Fraction of subgoals completed with fresh evidence |
| Failure attribution | Higher | Failure is assigned to an actionable stage/cause |
| Recovery success | Higher | Task succeeds after at least one bounded replan |
| Replan count | Context-dependent | Too few can mean no recovery; too many can mean instability |
| Verification actions | Lower for equal safety/success | Cost of obtaining fresh evidence |
| Collision/control failures | Lower | Safety and execution reliability |
| Episode latency | Lower subject to success | Closed-loop efficiency |

The current planning smoke injects one deterministic control failure. It validates verification and recovery logic, not
learned dynamics, simulator generalization, real-robot safety, or population success rates.

## 12. Efficiency and resource metrics

Efficiency is always recorded, but its role depends on the research question. During architecture discovery it is a
diagnostic. After a scientific winner is identified, a separately frozen deployment protocol may make it a hard gate.

| Metric | Unit | Interpretation |
|---|---|---|
| Trainable parameters | Count | Capacity/checkpoint-size proxy; not a latency proxy |
| Complete-head wall time | ms/sample | Host-observed head latency on cached features |
| CUDA-event time | ms/sample | Device execution between synchronized events |
| Encoder-plus-head wall time | ms/sample | End-to-end model path including frozen encoder |
| Component time | ms/sample | Pooling, K transform, decoder, scale head, field, composition |
| Throughput | samples/s | Batched training or inference rate |
| Peak allocated/reserved memory | MiB/GiB | Feasibility and headroom |
| Epoch/job elapsed time | seconds/minutes | Experimental resource planning |
| Latency ratio | Candidate/reference | Relative cost under one frozen timing protocol |
| Paired bootstrap CI | Ratio interval | Measurement stability across independent allocations |

Component event ranges may overlap or include synchronization; their means are not necessarily additive. Hardware, batch
size, precision, warmup, software versions, clock state, and resampling unit must be identical before comparing latency.

Phase 2f illustrates why both views matter: M1/M2/M3 head-only ratios were `1.681x/3.606x/4.362x`, while descriptive
encoder-plus-head ratios were `1.012x/1.039x/1.049x`. The head metric exposes architecture overhead; the system metric
describes its current application impact. Neither may replace the other after results are known.

## 13. Aggregation, variation, and intervals

| Quantity | Correct interpretation |
|---|---|
| Per-pixel mean | Pixel-weighted behavior; large images/masks dominate |
| Per-frame macro | Every frame has equal weight |
| Per-sequence macro | Every sequence has equal weight |
| Per-family/domain macro | Every camera family or domain has equal weight |
| Seed mean/SD | Optimizer variability for the fixed data and protocol |
| Paired difference | Candidate minus reference under the same rotation/seed |
| Bootstrap CI | Uncertainty over the explicitly resampled unit |

Rules:

- Report the resampling unit. Resampling frames does not estimate new-camera uncertainty.
- With only four camera families, a family-cluster interval is descriptive and necessarily coarse.
- Do not label optimizer seed SD as a population confidence interval.
- Prefer paired candidate/reference differences when splits and seeds match.
- Do not pool repeated transformed views as independent scenes.
- Persist per-frame/per-family values so macro calculations are auditable.
- Define missing and failed samples before evaluation; do not silently omit them.

## 14. Integrity and provenance metrics

These quantities establish whether a result is trustworthy rather than whether a model is good:

| Check | Required interpretation |
|---|---|
| Expected/completed jobs | Every declared logical cell reaches a valid terminal state |
| Receipt and SUCCESS status | Outputs and uploads finished before success is asserted |
| SHA-256 references | Inputs, code, configs, reports, and parent artifacts are content-bound |
| Dependency validation | A job consumed the exact declared predecessors |
| Online W&B identity | Run/artifact IDs are complete, unique, and successfully uploaded |
| Exact checkpoint reload | Serialized state reproduces fixed predictions |
| Finite-tree validation | No NaN/Inf entered metrics or receipts |
| Fresh-final sentinel | Irreversible external-target opening is explicit |

W&B is the interactive comparison surface. Versioned JSON, CSV, NPZ, checkpoints, HTML, PNG, manifests, and hashes are
the durable source of record. Credentials and raw protected targets must never be serialized.

## 15. Cross-phase comparability traps

The same short label can refer to a different estimand in a different phase. Use these boundaries when reading historical
tables:

- Phase 2 teacher depth is per-frame median-scale aligned; it has no raw single-image metric-scale score. Its point metrics
  inherit that alignment. The Sim(3) pose summary uses all eight sequence frames, not only the four depth-test frames.
- Phase 2 teacher NLL is metric-depth NLL after target-derived alignment. Phase 2b onward primarily use log-depth NLL.
  Absolute NLL values are not comparable across those units.
- Phase 2/2b-2e variance multipliers use a `[1e-4,1e4]` clip, while Phase 2f uses `[1e-3,1e3]`; the fitted values and
  saturation behavior are protocol-specific.
- Phase 2b-2e use ratio-of-medians depth alignment; Phase 2f uses median pixelwise log residual alignment.
- Phase 2b depth accuracy is evaluated after upsampling 24x24 predictions to 518x518, while its NLL remains at 24x24.
  Those metrics intentionally answer different spatial-resolution questions.
- Phase 2c averages frames within a sequence and then sequences equally. Phase 2e averages samples within a seed/control,
  then seeds. Phase 2f's primary aggregation averages frames within family/domain and then groups equally.
- Phase 2, Phase 2e, and Phase 2f use different AUSE coverage grids and aggregation. Compare AUSE between models inside a
  phase, not its absolute magnitude across phases.
- Phase 2d scale oracles fit evaluated targets. They are diagnostic upper bounds, not deployable predictors. Its spatial
  oracle is capacity-bounded by a 4x4 grid but is not numerically amplitude-bounded.
- Phase 2e exposes both target-derived `abs_log_scale_error` and an architecture-dependent
  `global_log_scale_abs_error`. Only the former governed promotion; the latter is not perfectly comparable between
  monolithic and factorized models.
- M0 uses the historical monolithic loss while M1-M3 use the separated shape/scale objective. M0-versus-candidate effects
  therefore measure architecture plus training-system change. M1-versus-M2-versus-M3 comparisons isolate the added
  camera/field mechanisms more cleanly.
- Phase 2f's SD over 12 M0 runs mixes four family effects with seed effects. Per-family three-seed SD is the optimizer
  stability view; neither is a population confidence interval.

Every future report should name the schema and metric implementation version. If two phases differ on alignment, units,
resolution, valid masks, calibration, or aggregation, their numbers belong in separate columns rather than one ranking.

## 16. Why the experiment sequence exists

| Phase | Uncertainty reduced | Why the next phase was necessary |
|---|---|---|
| Phase 1 | Can real V-JEPA produce stable spatial/temporal features? | Finite integration does not establish geometry quality |
| Phase 2 | Is VGGT a measurable geometry teacher and what are its limits? | A large teacher does not show whether compact V-JEPA features retain geometry |
| Phase 2b | Can a lightweight V-JEPA head recover held-out geometry, and which layer works? | One TUM sequence cannot establish cross-camera transfer |
| Phase 2c | Does learned layer fusion generalize across camera families? | A retrained comparison does not identify whether gates or scale caused the behavior |
| Phase 2d | Are gates causally active, is scale dominant, and what is true latency? | Diagnosis showed scale, camera provenance, and calibration needed direct modeling |
| Phase 2e | Can explicit shape/scale/camera factorization repair sensor transfer? | Shape improved but scale/calibration/runtime and the constant-K control failed |
| Phase 2f | Do detached-scale/camera implementations satisfy a latency-first operational screen? | M1-M3 were not trained, so their scientific quality hypotheses remain unanswered |
| Phase 2g proposed | Do M1-M3 improve quality and use camera/field mechanisms causally when speed is descriptive only? | A winner must be established before implementation optimization or external confirmation |

Phases 3-6 then test whether geometry can support grounded objects, durable identity/memory, verified planning, and a
common evaluation contract. Their current smoke results validate interfaces and invariants; official labeled datasets and
independent episodes are still required for model-quality claims.

## 17. Required presentation for future reports

Every important metric in JSON, HTML, Markdown, and W&B should expose:

- exact name, formula/version, unit, and direction;
- split and whether it selected the checkpoint;
- alignment and calibration policy;
- denominator and failure count;
- frame/sequence/family aggregation;
- mean plus distribution or per-unit table;
- baseline, absolute difference, and relative ratio where valid;
- uncertainty/resampling unit;
- evidence level and claim boundary;
- a fixed visualization that helps diagnose rather than select examples.

Minimum geometry visual set:

1. raw versus aligned error by family and seed;
2. signed/absolute scale residual distributions;
3. predicted versus optimal log-scale scatter;
4. reliability/coverage and risk-coverage curves;
5. fixed RGB, target, prediction, error, and uncertainty panels;
6. camera-control deltas for K-conditioned arms;
7. training losses and allowed/forbidden gradient norms;
8. descriptive resource use, clearly separated from quality selection.

## 18. Authoritative implementation references

- [Phase 2f depth, calibration, coverage, and AUSE metrics](../jepa4d/evaluation/phase2f_metrics.py)
- [Phase 2f losses and robust detached-scale target](../jepa4d/training/phase2f_losses.py)
- [Phase 2f gradient firewall and strict reload](../jepa4d/training/phase2f_training.py)
- [Paired camera-control construction](../jepa4d/evaluation/phase2f_camera_controls.py)
- [TUM aligned depth, point, pose, and uncertainty metrics](../jepa4d/benchmarks/geometry/tum_rgbd.py)
- [Persistent identity metrics](../jepa4d/benchmarks/tracking4d/identity.py)
- [Benchmark scope and future dataset requirements](BENCHMARKS.md)
- [Experiment evidence index](experiments/INDEX.md)

If an experiment intentionally changes one of these definitions, it must use a new schema/version, state the change in its
preregistration, and never pool the new number with historical values under an unchanged label.
