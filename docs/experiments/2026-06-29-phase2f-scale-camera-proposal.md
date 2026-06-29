# Phase 2f proposal — identifiable camera controls and detached metric-scale recovery

**Status:** proposed, not preregistered, not authorized for formal execution

**Input evidence:** completed Phase 2d diagnostics and Phase 2e sensor-blocked SUNRGBD benchmark

**Recommendation:** do not continue the current `factorized_full_teacher` candidate unchanged

## Decision at a glance

| Observation | Evidence | Design consequence |
|---|---:|---|
| Learned Phase 2c gates have negligible causal effect. | Original-minus-zero raw AbsRel is `-0.000081`; fixed averaging changes the result much more. | Stop spending formal runs on the current gating mechanism. |
| The Phase 2e candidate improves shape more than metric scale. | Versus monolithic: raw AbsRel improves `2.67%` and aligned AbsRel improves `6.22%`, but absolute log-scale error worsens `13.82%`. | Isolate scale learning from shape gradients and make scale the primary ablation. |
| The registered shuffled-`K` condition is not identifiable. | All 128 kv2 test samples have the same `K`; shuffling is therefore exactly the identity operation. | Replace it with paired, analytically controlled intrinsics perturbations and a multi-`K` set. |
| Parameter count did not predict runtime. | The candidate is `1.0995×` the parameters but `9.3578×` the synchronized head latency. | Require component-level latency qualification before formal training. |
| Uncertainty ranking and calibration disagree. | Candidate AUSE improves `12.96%`, while calibrated NLL is `0.0675` worse. | Model and gate ranking quality separately from probabilistic calibration. |
| Teacher supervision is incremental, not transformative. | Teacher improves the full factorized arm by `1.45%` raw and `0.80%` aligned AbsRel, but worsens calibrated NLL. | Remove VGGT teacher loss from the default arm; retain it only as a late ablation. |

The previous kv2 test has now been opened and must become **development-only**. Phase 2f needs a newly frozen final test.

## Research-grounded direction

- [MoGe-2](https://arxiv.org/abs/2507.02546) separates affine-invariant geometry from a global metric-scale head and supervises scale from an optimal alignment target with stopped shape gradients. This directly addresses the interference visible in Phase 2e.
- [Metric3D v2](https://arxiv.org/abs/2404.15506) uses a canonical camera-space transform to resolve cross-camera metric ambiguity. A canonical transform is a better first camera baseline than the current expensive dense-ray path.
- [UniDepth](https://arxiv.org/abs/2403.18913) conditions depth on a learned dense camera representation and adds geometric invariance. This is a useful second-line design only after a cheap canonical-camera arm passes causal controls.
- [UniK3D](https://openaccess.thecvf.com/content/CVPR2025/html/Piccinelli_UniK3D_Universal_Camera_Monocular_3D_Estimation_CVPR_2025_paper.html) shows why spherical ray representations are preferable when moving beyond pinhole cameras.
- [UniDAC](https://openaccess.thecvf.com/content/CVPR2026/html/Ganesan_UniDAC_Universal_Metric_Depth_Estimation_for_Any_Camera_CVPR_2026_paper.html) decomposes relative depth from spatially varying scale. It motivates one tightly regularized coarse-scale-field arm, not an unrestricted dense correction.
- [Depth Pro](https://machinelearning.apple.com/research/depth-pro) jointly demonstrates metric prediction, focal estimation, and practical dense-prediction speed; it is a useful engineering reference for keeping camera reasoning off the critical dense path.
- [CRUDE](https://arxiv.org/abs/2005.12496) provides a distribution-free empirical regression-calibration baseline. It should be compared with the existing scalar variance multiplier without changing point predictions.

These papers motivate hypotheses and controls; they do not imply that their reported performance transfers to this V-JEPA feature setting.

## Six-step Phase 2f design

### 1. Repair the evaluation protocol before model work

Create two orthogonal development suites:

1. **Sensor transfer:** rotate the four SUNRGBD camera families through train/validation/development-test roles. Report every rotation and the camera-family macro mean.
2. **Paired camera causality:** apply frozen crop/resize, letterbox, focal-scale, and principal-point transforms to the same source frames. For each transformed image, compare:
   - analytically updated `K`;
   - stale pre-transform `K`;
   - deliberately wrong `K`;
   - a seeded permutation over at least eight distinct matrices.

The paired suite tests camera use without confounding scene content. The current constant-`K` shuffle must not appear as a hard gate again.

Reserve a new external final set before training. Practical candidates are an MIT-licensed [DIODE](https://diode-dataset.org/diode-dataset.github.io) mini split for fresh domain transfer and a separately licensed [ETH3D](https://eth3d.ethz.ch/) subset for calibrated multi-intrinsics stress. Perform a license and asset audit before choosing; do not inspect final targets during development.

### 2. Build a small, interpretable model matrix

| Arm | Shape path | Scale path | Camera path | Purpose |
|---|---|---|---|---|
| `M0 monolithic` | Current baseline | Entangled | None | Frozen reference |
| `M1 detached-global` | Centered/affine shape | Pooled V-JEPA global scalar | None | Test scale/shape gradient separation |
| `M2 canonical-K` | Same as M1 | Global scalar | Canonical camera transform or low-rank ray summary | Cheapest identifiable camera model |
| `M3 coarse-scale-field` | Same as M1 | Global scalar plus zero-mean coarse residual field | Canonical `K` | Test spatially varying scale without a free dense correction |
| `M4 camera-prompt` | Same as M1 | Global scalar | Compact learned camera prompt | Optional only if M2 is causal and fast |

Do not include RGB-only or bias-only scale arms in formal training: both failed sensor transfer. Do not include the teacher arm unless a non-teacher survivor already passes scale and latency gates.

### 3. Put a gradient firewall between shape and scale

For a predicted centered shape `z_shape`, derive an optimal training scale from valid target pixels:

`s* = robust_mean(log(depth_gt) - stopgrad(z_shape))`

Train the scale head on `s*` in log space. Train shape with aligned/centered geometry losses. The scale loss must not update the shape branch, and the shape loss must not update the scale head. Log both gradient norms and assert this firewall in tests.

For M3, constrain the coarse residual scale field to zero spatial mean, penalize total variation, and cap its amplitude. This preserves an interpretable global scale rather than allowing the field to absorb arbitrary depth error.

Factor uncertainty as a global scale component plus a dense shape component. Evaluate the existing validation-fitted scalar multiplier against empirical calibration; never fit either on test.

### 4. Qualify causality and latency before 60-epoch training

Run a short Slurm pilot for each arm with:

- exact parameter counts;
- synchronized component latency for pooling, camera transform, ray construction, dense shape decoder, scale head, and composition;
- 12 randomized/interleaved latency jobs using the existing Phase 2d protocol;
- updated/stale/wrong/permuted `K` paired deltas;
- finite gradients and strict checkpoint reload equality.

Hard pre-training gates:

| Gate | Requirement |
|---|---:|
| Parameters | target `≤1.05×`, hard ceiling `≤1.10×` M0 |
| Head latency | upper 95% CI `≤1.10×` M0 |
| Camera identifiability | updated `K` strictly beats stale and wrong `K` on paired raw AbsRel |
| Permutation validity | permutation changes at least 95% of matrices and produces nonzero output deltas |
| Numerical health | no NaN/Inf, zero strict-reload mismatches |

An arm that fails here does not proceed to formal training.

### 5. Train survivors across camera-family rotations

Use three seeds and the same epoch budget for M0 and survivors. Select checkpoints only on the registered validation family. Aggregate camera-family means, not repeated frames, and retain the no-population-significance boundary.

Log online to W&B and persist locally:

- raw/aligned depth, scale error, NLL, reliability, and AUSE per family and seed;
- predicted versus optimal global scale scatter and residual histograms;
- scale/shape/metric loss and separated gradient norms;
- paired intrinsics-control deltas;
- component latency, GPU telemetry, peak memory, and parameter counts;
- fixed target/prediction/error/uncertainty/scale-field panels;
- self-contained HTML, CSV/JSON/NPZ, hashes, test receipt, Slurm graph, git commit, and W&B artifact receipt.

Every cache and shard receipt should embed its own `execution_provenance` block: git commit, test-receipt hash/job, Slurm job, source identities, and parent artifact hashes.

### 6. Open one fresh final test once

After choosing exactly one survivor, evaluate M0 and the survivor on the unopened external set. The promotion gate should require all of:

| Dimension | Hard final condition |
|---|---|
| Metric quality | survivor raw AbsRel strictly lower than M0 |
| Shape | aligned AbsRel no more than `1.02×` M0 |
| Scale | absolute log-scale error strictly lower than M0 |
| Camera causality | updated `K` beats stale, wrong, and permuted `K` in the registered paired suite |
| Calibration | calibrated NLL strictly lower; AUSE no worse |
| Efficiency | latency upper 95% CI `≤1.10×`; parameters `≤1.10×` |
| Integrity | all finite, zero failures, exact receipts and dependency graph |

## Decision tree after Phase 2f

- **M1 passes, camera arms do not:** retain a camera-free detached global-scale model.
- **M2 passes causality and latency:** promote canonical camera conditioning; do not add a learned prompt.
- **Only M3 improves external scale:** retain the coarse field only if its latency and regularization gates pass.
- **AUSE improves but NLL does not:** preserve the ranking signal, replace the variance distribution/calibrator, and do not claim calibrated uncertainty.
- **No arm lowers scale error:** stop architecture ablations and invest in broader metric-scale supervision/data quality.

This proposal becomes a preregistration only after the new final dataset, split hashes, transform matrix, exact loss weights, resource budget, and gate thresholds are frozen.
