# Phase 2e factorized shape/scale geometry on SUN RGB-D — completed

## Experiment metadata

| Field | Value |
|---|---|
| Experiment ID | `phase2e-sunrgbd-factorized-geometry-v1` |
| Stage / status | `formal training and untouched-test evaluation complete` |
| Evidence level | sensor-blocked SUN RGB-D feature-cache experiment |
| Execution commit | `547ecc1ee2d2cc40863c179b625ce7952010a94d` |
| Hardware | NVIDIA A100-SXM4-80GB, Slurm `polar4` |
| Fixed reference | `monolithic_final` |
| Fixed promotion candidate | `factorized_full_teacher` |
| Operational decision | **Gate failed: retain the monolithic reference; do not promote the candidate.** |
| Final W&B run | [89ugevtp](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/89ugevtp) |
| Canonical local result | [phase2e_final_evaluation.json](../../outputs/jepa4d_phase2e/final/phase2e_final_evaluation.json) |
| Visual report | [phase2e_final_report.html](../../outputs/jepa4d_phase2e/final/phase2e_final_report.html) |
| Frozen protocol | [2026-06-29-phase2e-sunrgbd-preregistered.md](2026-06-29-phase2e-sunrgbd-preregistered.md) |

The run is technically complete and its strict postflight passed. That is distinct from the scientific/operational gate:
the fixed candidate passed five of nine conditions but failed scale transfer, calibrated NLL, the shuffled-K control, and
head latency. The held-out Kinect-v2 test split is now consumed and must not be used to tune a replacement candidate.

## Executive result

| Question | Observation | Decision |
|---|---|---|
| Does the fixed factorized candidate improve raw metric depth? | Raw AbsRel is `0.188893` versus `0.194065`, a 2.67% reduction. | Quality signal is positive but insufficient for promotion. |
| Does it preserve relative shape? | Aligned AbsRel is `0.144412` versus `0.153983`, a 6.22% reduction. | Factorized shape is the strongest part of the result. |
| Does it improve metric scale? | Absolute log-scale error is `0.120791` versus `0.106125`, 13.82% worse. | The explicit scale branch did not solve cross-sensor scale transfer. |
| Does it improve calibrated uncertainty? | Calibrated NLL is `-0.801416` versus `-0.868922`; lower is better. | Retain the reference uncertainty path. |
| Is camera metadata causally validated? | Correct and shuffled K are exactly tied because every test row has the same K. | No sample-specific camera-information claim is supported. |
| Is the compact head operationally efficient? | `3.288217 / 0.351388 = 9.357786x` reference head latency. | The latency gate fails decisively. |

The best raw mean among the eight registered variants is `factorized_vjepa_k` at `0.188512`, only 0.20% below the fixed
candidate. The teacher candidate is slightly better in aligned geometry, but adding RGB and VGGT distillation did not win
the primary metric over the simpler V-JEPA-plus-K ablation.

## Frozen data and evaluation scope

| Split | Sensor family | Samples | Views per sample | Permitted role |
|---|---|---:|---:|---|
| Train | Kinect v1 + Asus Xtion | 192 + 192 | 2 | optimization only |
| Validation | Intel RealSense | 128 | 1 | checkpoint selection and variance calibration |
| Test | Kinect v2 | 128 | 1 | one final evaluation |

The train/validation cache SHA-256 is
`f65678d3bf60b4a67013a7029bcc7aa402b654f47aa4108bdd70b807013073d4`; the separately sealed test cache SHA-256 is
`0d7595245bb659705feea24080375a7c05ffbb5b0a20fb214314b3364a22f8d6`. The selected split hash is
`d1815109fa0b34dd2270f1da616d4ff65beaa41fcf437b7d75ead557a1ab75c7` and the manifest SHA-256 is
`174716f4f1bd4a4b709f2a4b1a1cd4dca4fd17ef34cc543c1fc8985b75b44c92`.

The cache job executed frozen V-JEPA on 768 train views, 128 validation images, and 128 test images. VGGT generated only
centered log-depth shape targets for the 768 training views; it never saw validation or test targets and no VGGT metric
scale was fitted. The cache receipt explicitly records `model_metrics_computed=false` and
`large_caches_uploaded_to_wandb=false`: W&B stores the receipt/report, while the large hash-bound caches remain local.

## Exact held-out test comparison

Values are equal-weight test-sample macros, reported as mean +/- sample standard deviation over optimization seeds 0, 1,
and 2. The table is reproduced from the canonical final JSON and rounded only for display to six decimal places. The JSON
retains full floating-point precision. Lower is better except Delta-1.

### Depth and scale

| Variant | Raw AbsRel | Raw RMSE m | Raw Delta-1 | Aligned AbsRel | Aligned RMSE m | Aligned Delta-1 | Abs log-scale error |
|---|---:|---:|---:|---:|---:|---:|---:|
| `monolithic_final` (reference) | 0.194065 +/- 0.018029 | 0.686962 +/- 0.038144 | 0.732942 +/- 0.041745 | 0.153983 +/- 0.003559 | 0.630329 +/- 0.040357 | 0.818481 +/- 0.017201 | **0.106125 +/- 0.014738** |
| `factorized_bias` | 0.269723 +/- 0.013724 | 1.322170 +/- 0.054905 | 0.574183 +/- 0.018291 | 0.310927 +/- 0.021447 | 1.224416 +/- 0.089851 | 0.507496 +/- 0.024694 | 0.248812 +/- 0.029408 |
| `factorized_vjepa` | 0.201658 +/- 0.005400 | 0.676592 +/- 0.017398 | 0.705707 +/- 0.030111 | 0.153280 +/- 0.006392 | 0.590036 +/- 0.020366 | 0.832135 +/- 0.014837 | 0.121480 +/- 0.012253 |
| `factorized_rgb` | 0.272867 +/- 0.018676 | 0.988392 +/- 0.079964 | 0.563779 +/- 0.026052 | 0.207367 +/- 0.029222 | 0.842508 +/- 0.112915 | 0.691357 +/- 0.060252 | 0.174151 +/- 0.005460 |
| `factorized_vjepa_rgb` | 0.192794 +/- 0.006717 | 0.654198 +/- 0.017295 | 0.737689 +/- 0.018005 | 0.145485 +/- 0.006638 | 0.572819 +/- 0.009478 | 0.847227 +/- 0.010481 | 0.120002 +/- 0.012928 |
| `factorized_vjepa_k` | **0.188512 +/- 0.003724** | **0.649903 +/- 0.016280** | **0.755823 +/- 0.010671** | 0.145344 +/- 0.000624 | **0.560903 +/- 0.004572** | **0.853529 +/- 0.002664** | 0.117892 +/- 0.008150 |
| `factorized_full` | 0.191682 +/- 0.005260 | 0.669737 +/- 0.010495 | 0.741690 +/- 0.018887 | 0.145574 +/- 0.003241 | 0.597685 +/- 0.046481 | 0.842584 +/- 0.017309 | 0.121215 +/- 0.005540 |
| `factorized_full_teacher` (candidate) | 0.188893 +/- 0.010236 | 0.661900 +/- 0.034170 | 0.745411 +/- 0.038834 | **0.144412 +/- 0.006591** | 0.567292 +/- 0.010316 | 0.851359 +/- 0.003829 | 0.120791 +/- 0.010963 |

### Uncertainty and resources

| Variant | Raw NLL | Calibrated NLL | Raw reliability error | Calibrated reliability error | AUSE | Head ms | Trainable params |
|---|---:|---:|---:|---:|---:|---:|---:|
| `monolithic_final` (reference) | **-0.966544 +/- 0.174906** | **-0.868922 +/- 0.062233** | 0.120742 +/- 0.040421 | 0.161120 +/- 0.017946 | 0.052547 +/- 0.007487 | **0.351388 +/- 0.002464** | 86,402 |
| `factorized_bias` | -0.666947 +/- 0.085238 | -0.635965 +/- 0.063439 | **0.113051 +/- 0.003394** | **0.129529 +/- 0.006592** | **0.038163 +/- 0.007230** | 0.384690 +/- 0.000225 | 86,403 |
| `factorized_vjepa` | -0.696146 +/- 0.551432 | -0.862087 +/- 0.094782 | 0.172575 +/- 0.082590 | 0.153411 +/- 0.007235 | 0.052024 +/- 0.001998 | 0.563600 +/- 0.004316 | 92,795 |
| `factorized_rgb` | -0.570541 +/- 0.014029 | -0.569258 +/- 0.079961 | 0.159830 +/- 0.010456 | 0.161067 +/- 0.008734 | 0.069525 +/- 0.012138 | 0.728804 +/- 0.004943 | 88,227 |
| `factorized_vjepa_rgb` | -0.289450 +/- 0.355693 | -0.814897 +/- 0.038010 | 0.243095 +/- 0.038363 | 0.175853 +/- 0.011588 | 0.048071 +/- 0.002322 | 0.802974 +/- 0.002773 | 94,571 |
| `factorized_vjepa_k` | 0.049657 +/- 0.628070 | -0.836190 +/- 0.029681 | 0.273047 +/- 0.043512 | 0.172799 +/- 0.006973 | 0.046325 +/- 0.001158 | 3.146919 +/- 0.005629 | 93,227 |
| `factorized_full` | -0.039991 +/- 1.168826 | -0.830945 +/- 0.075622 | 0.239642 +/- 0.115171 | 0.164540 +/- 0.013183 | 0.047382 +/- 0.007401 | 3.342136 +/- 0.037720 | 95,003 |
| `factorized_full_teacher` (candidate) | 0.378884 +/- 0.375626 | -0.801416 +/- 0.029195 | 0.307501 +/- 0.035732 | 0.169485 +/- 0.007777 | 0.045738 +/- 0.003162 | 3.288217 +/- 0.002644 | 95,003 |

Head timing is synchronized batch-one head latency over cached features. It excludes V-JEPA feature extraction and is not
an end-to-end deployment latency number. Parameter counts cover trainable probe parameters, not the frozen encoders.

## Camera-intrinsics controls

Only the three K-conditioned variants have wrong/shuffled controls. Values again are test mean +/- sample SD over seeds.

| Variant | K control | Raw AbsRel | Aligned AbsRel | Abs log-scale error | Calibrated NLL |
|---|---|---:|---:|---:|---:|
| `factorized_vjepa_k` | correct | 0.188512 +/- 0.003724 | 0.145344 +/- 0.000624 | 0.117892 +/- 0.008150 | -0.836190 +/- 0.029681 |
| `factorized_vjepa_k` | shuffled | 0.188512 +/- 0.003724 | 0.145344 +/- 0.000624 | 0.117892 +/- 0.008150 | -0.836190 +/- 0.029681 |
| `factorized_vjepa_k` | wrong | 0.201866 +/- 0.004556 | 0.145302 +/- 0.000649 | 0.117553 +/- 0.003031 | -0.844263 +/- 0.009135 |
| `factorized_full` | correct | 0.191682 +/- 0.005260 | 0.145574 +/- 0.003241 | 0.121215 +/- 0.005540 | -0.830945 +/- 0.075622 |
| `factorized_full` | shuffled | 0.191682 +/- 0.005260 | 0.145574 +/- 0.003241 | 0.121215 +/- 0.005540 | -0.830945 +/- 0.075622 |
| `factorized_full` | wrong | 0.206038 +/- 0.015520 | 0.145586 +/- 0.003300 | 0.125339 +/- 0.013517 | -0.819501 +/- 0.036800 |
| `factorized_full_teacher` | correct | 0.188893 +/- 0.010236 | 0.144412 +/- 0.006591 | 0.120791 +/- 0.010963 | -0.801416 +/- 0.029195 |
| `factorized_full_teacher` | shuffled | 0.188893 +/- 0.010236 | 0.144412 +/- 0.006591 | 0.120791 +/- 0.010963 | -0.801416 +/- 0.029195 |
| `factorized_full_teacher` | wrong | 0.194266 +/- 0.005364 | 0.144348 +/- 0.006553 | 0.114606 +/- 0.007874 | -0.818932 +/- 0.025061 |

The exact correct/shuffled tie has a concrete cause. A post-result cache audit found that the test tensor has 128 K rows
but only one distinct `intrinsics_384` matrix:

```text
[[383.63775635,   0.00000000, 191.86224365],
 [  0.00000000, 383.63775635, 191.86225891],
 [  0.00000000,   0.00000000,   1.00000000]]
```

The registered shuffled control is a one-sample cyclic roll, so it changed 0 of 128 test rows. It is an identity transform
on this split, not a meaningful negative control. Wrong K degraded raw AbsRel for all three variants, which demonstrates
sensitivity to a global K perturbation; it does not demonstrate that the model uses the correct camera calibration for the
corresponding image. No camera-metadata causal claim is made.

## Frozen promotion gate

| Registered condition | Exact observation | Result |
|---|---:|---|
| Candidate raw AbsRel strictly lower | `0.188893 < 0.194065` | PASS |
| Candidate scale error strictly lower | `0.120791 > 0.106125` | **FAIL** |
| Aligned AbsRel no more than 2% worse | ratio `0.937842` | PASS |
| Candidate calibrated NLL strictly lower | `-0.801416 > -0.868922` | **FAIL** |
| Correct K beats wrong K on raw AbsRel | `0.188893 < 0.194266` | PASS |
| Correct K beats shuffled K on raw AbsRel | `0.188893 = 0.188893` | **FAIL** |
| Candidate head latency at most 1.10x | ratio `9.357786x` | **FAIL** |
| Candidate parameters at most 1.10x | ratio `1.099546x` | PASS |
| All runs/checkpoints/metrics/uploads finite and complete, zero failures | 24/24 checkpoints; 0 failures | PASS |
| **Overall fixed gate** | five pass, four fail | **FAIL — retain `monolithic_final`** |

The parameter count sits only 0.045 percentage points below the allowed 10% overhead, while the latency exceeds the
reference by more than ninefold. This is not a borderline resource decision.

## Interpretation and insights

1. **Shape factorization worked better than scale factorization.** The strong factorized variants improve aligned AbsRel
   by about 5.5-6.2% versus the reference, but every one has worse absolute log-scale error. The modest raw gains come from
   improved relative geometry overcoming a scale penalty, not from solving metric scale transfer.
2. **V-JEPA is the useful scale signal; RGB alone is not.** `factorized_bias` and `factorized_rgb` are the two weakest raw
   models. Adding pooled V-JEPA to RGB lowers raw AbsRel from `0.272867` to `0.192794` under the registered architectures.
3. **The extra teacher/RGB machinery is not justified by the primary result.** `factorized_vjepa_k` has the best raw mean
   and the lowest seed SD. The fixed teacher candidate is 0.20% worse on raw AbsRel, although it is 0.64% better on aligned
   AbsRel. Teacher distillation versus otherwise-full factorization improves raw AbsRel by 1.45%, but that gain is too small
   to rescue scale, uncertainty, or latency.
4. **Uncertainty quality did not track depth quality.** The candidate improves AUSE by 12.96% relative to the reference,
   meaning its ranking of risky pixels is useful, but its calibrated NLL is worse by `0.067506` and calibrated reliability
   error is 5.19% higher. Better selective ranking is not equivalent to a better probabilistic depth model.
5. **The K experiment is structurally underidentified on the test split.** A single repeated Kinect-v2 K makes shuffling
   vacuous. Wrong-K sensitivity cannot establish sample-specific use. This is a protocol-design lesson, not evidence that
   camera conditioning is either causally useful or useless.
6. **The camera-conditioned implementation is the dominant head-cost problem.** Latency rises from `0.802974` ms for
   V-JEPA+RGB to `3.146919` ms for V-JEPA+K, a 3.92x increase, while raw AbsRel improves 2.22%. Component profiling is
   required before another resource-constrained candidate is registered.
7. **More same-split optimization seeds would not resolve the main uncertainty.** The three seeds measure optimizer
   variation. They do not provide independent sensors, scenes, or camera calibrations, and no population-significance
   claim is supported.

## Execution graph and integrity

All promoted jobs use the same clean commit and passed through `afterok` dependencies:

```text
29597457 tests
  -> 29597477 feature cache
       -> 29597484 two-epoch pilot
            -> [29597494 shard 0, 29597500 shard 1,
                29597502 shard 2, 29597506 shard 3]
                 -> 29597561 final join/evaluation/postflight
```

| Stage | Job | State / elapsed | Contents | Evidence |
|---|---:|---:|---|---|
| Tests + CUDA receipt | `29597457` | COMPLETED / 00:02:48 | Ruff, mypy, 194 tests, sustained CUDA check | [tests.json](../../outputs/phase2e-gates/tests.json) |
| Feature cache | `29597477` | COMPLETED / 00:08:28 | frozen V-JEPA, train-only VGGT, split caches | [W&B 456flq9b](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/456flq9b) |
| Pilot | `29597484` | COMPLETED / 00:01:38 | reference + fixed candidate, 3 seeds, 2 epochs | [W&B pweok69a](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/pweok69a) |
| Formal shard 0 | `29597494` | COMPLETED / 00:04:25 | reference + factorized bias, 3 seeds | [W&B 89fn7n3h](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/89fn7n3h) |
| Formal shard 1 | `29597500` | COMPLETED / 00:07:40 | factorized V-JEPA + factorized RGB, 3 seeds | [W&B iwey8f0r](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/iwey8f0r) |
| Formal shard 2 | `29597502` | COMPLETED / 00:07:17 | V-JEPA+RGB + V-JEPA+K, 3 seeds | [W&B 1vt1w89f](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/1vt1w89f) |
| Formal shard 3 | `29597506` | COMPLETED / 00:06:49 | full + full teacher, 3 seeds | [W&B 0gz96hdf](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/0gz96hdf) |
| Final evaluation | `29597561` | COMPLETED / 00:02:18 | 24 checkpoints, controls, tables, curves, report, upload | [W&B 89ugevtp](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/89ugevtp) |

The pilot produced six strict-reload passes. Each formal shard contains six exact checkpoints, 360 epoch rows, a 29-file
manifest, GPU telemetry, a self-contained HTML report, and an uploaded online W&B artifact. The final evaluator verified
24 checkpoints and produced 42 seed/control rows and 5,376 per-sample rows over 128 test samples with zero recorded
failures. Final evaluation runtime was 42.6271 seconds under Python 3.12.13, PyTorch 2.7.1+cu118, and CUDA 11.8.

The dependency-graph SHA-256 is `508b3af7235eaf06bd50e35b619e83c29ac27a63cc8e59c94b68d8d555349405`; the
test-receipt SHA-256 is `0dece58f8c2fb3cb0f87dd58aa8d4e65001d54b9cfc863a5eca2a9c11825d9bd`. Strict
[postflight](../../outputs/jepa4d_phase2e/final/postflight.json) status is `pass` while `gate_passed=false`, as intended.

## W&B and durable artifacts

| Stage | Artifact name | W&B artifact ID | Digest |
|---|---|---|---|
| Cache receipt | `phase2e-sunrgbd-cache-29597477-receipt:v0` | `QXJ0aWZhY3Q6MzA3NTE4NTQ2MA==` | `b964c6d755b4cb9b5419b282049f014a` |
| Pilot | `pweok69a-phase2e-factorized-shard:v0` | `QXJ0aWZhY3Q6MzA3NTE5Mzk3Nw==` | `aabd457b35e23661026ced652a0614d3` |
| Formal shard 0 | `89fn7n3h-phase2e-factorized-shard:v0` | `QXJ0aWZhY3Q6MzA3NTIxNDQxNQ==` | `97d07ad56bd97df6a32c18d778f7b4d6` |
| Formal shard 1 | `iwey8f0r-phase2e-factorized-shard:v0` | `QXJ0aWZhY3Q6MzA3NTIyOTc1MA==` | `dca5dbb1fbc1056e3ddea248323513c5` |
| Formal shard 2 | `1vt1w89f-phase2e-factorized-shard:v0` | `QXJ0aWZhY3Q6MzA3NTIyNzY0NA==` | `23305f23c99f1f9be05316c629466175` |
| Formal shard 3 | `0gz96hdf-phase2e-factorized-shard:v0` | `QXJ0aWZhY3Q6MzA3NTIyNTU4Nw==` | `6095d072b2e16bcff00ac4a05672415e` |
| Final | `89ugevtp-phase2e-final-evaluation:v0` | `QXJ0aWZhY3Q6MzA3NTI0MDQzNA==` | `afafaa2b84f2da5a9510ea5f80e62dcd` |

Final durable files:

| Role | Local file | Bytes | SHA-256 |
|---|---|---:|---|
| Canonical evaluation | [phase2e_final_evaluation.json](../../outputs/jepa4d_phase2e/final/phase2e_final_evaluation.json) | 13,465,417 | `2b9336b438a3b0528337f424c2969ee9fedc7a8e4102f48a0e4f55e705c78113` |
| Full predictions | [phase2e_final_predictions.npz](../../outputs/jepa4d_phase2e/final/phase2e_final_predictions.npz) | 21,734,000 | `516b3a032a5366c0f22ffc09ea1e02d60d79591b7f418eb75fe16743c70b2039` |
| Per-sample metrics | [phase2e_final_per_sample.csv](../../outputs/jepa4d_phase2e/final/phase2e_final_per_sample.csv) | 6,072,519 | `11ee5f756417f2d1ef91ccfa5a77535f560746bc44da97758cb65ecd1d12683d` |
| Self-contained visual report | [phase2e_final_report.html](../../outputs/jepa4d_phase2e/final/phase2e_final_report.html) | 5,593,841 | `35f17c0bd092857288173b26bb92b772b8a7c8d4c366113f21c1d7e96c8b9053` |

The HTML report is the preferred visual diagnostic surface for gate cards, variant/seed comparisons, scale plots,
uncertainty curves, K controls, resource plots, and bounded qualitative panels. The JSON/CSV/NPZ plus hashes are the
canonical audit record; W&B is the interactive comparison and training-curve surface.

## Retry and protocol-correction history

All failures below happened before the promoted final evaluation. They are retained because they explain why the complete
DAG was rerun from a single clean commit.

| Commit / job | Outcome | Correction and disposition |
|---|---|---|
| `291c47b`, test `29594930` | Test/CUDA job passed; 189 tests passed and one real-model test skipped. | Initial gate receipt only; not used by the final DAG. |
| `291c47b`, cache `29594951` | Allocation was cancelled after 17 seconds; Slurm reported an expired job. | No result used. A replacement was submitted. |
| `291c47b`, cache `29594960` | Failed before cache computation because the receipt validator treated the approved comma-separated partition fallback list as one unapproved partition. | Commit `e5beb8e` validates every requested fallback partition separately. |
| `e5beb8e`, test `29595898` | Failed the real V-JEPA adapter test because checkpoint and implementation paths referred to unmatched legacy locations. | Commit `1602074` binds the test to the matched Phase-2b model and implementation assets. |
| `1602074`, test `29595964`; cache `29595981` | Both completed successfully. | Cache was not promoted after the downstream pilot exposed a report-writing failure and the commit changed. |
| `1602074`, pilot `29595983` | Training and W&B run [d38sjgzb](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/d38sjgzb) began, but the job failed while writing a report containing a middle-dot character under an ASCII locale. | Commit `547ecc1` makes report writes explicitly UTF-8, exports `PYTHONUTF8=1`, and adds a regression test. |
| `547ecc1`, jobs `29597457` through `29597561` | Full test/cache/pilot/formal/final DAG completed. | This is the only promoted result lineage. |

These were infrastructure and evidence-generation corrections, not result-conditioned changes to the model family,
candidate, split, metrics, or promotion thresholds.

## Claim boundaries

- This establishes the registered result on 128 SUN RGB-D Kinect-v2 samples under the frozen sensor-blocked split. It is
  not evidence of universal monocular metric-depth or camera-family generalization.
- Sensor family, data source, time, and scene composition remain confounded. The split cannot isolate a pure sensor effect.
- Three optimization seeds are not three independent datasets or cameras. The displayed SD is seed variation, not a
  population confidence interval or statistical-superiority test.
- The fixed candidate failed its operational gate. Explanatory ablation rankings cannot replace it after observing test.
- The test split is consumed. It may be used for diagnosis with an explicit non-confirmatory label, never for selecting a
  new promoted architecture or threshold.
- The correct-versus-shuffled-K test was vacuous because all cached test K matrices were identical. Only global wrong-K
  sensitivity was executed; no causal claim about correct image/K pairing is supported.
- Head latency excludes frozen feature extraction, and trainable parameter count excludes frozen encoders. Neither number
  is a complete deployment cost.
- Uncertainty calibration used one scalar fitted on RealSense validation predictions and evaluated on Kinect v2. Better
  AUSE does not override worse calibrated NLL/reliability.

## Recommended next step

Do not run another candidate against this test split. The next confirmatory phase should be preregistered on a fresh,
intrinsics-diverse holdout and should start with the following gates before any model result is opened:

1. **Make the K control non-vacuous by construction.** Require more than one distinct transformed K, verify that shuffling
   changes K for essentially every sample, persist the changed-row fraction and K-distance distribution, and abort before
   inference if the control degenerates. Prefer paired calibrated views or a camera-diverse dataset so correct and
   mismatched K can be compared while the RGB prediction input is held fixed.
2. **Use the simplest evidence-supported candidate.** Begin train/validation-only development from
   `factorized_vjepa_k`, not the larger teacher candidate. Add RGB or VGGT distillation only if each wins a frozen ablation
   on unconsumed validation domains.
3. **Profile and redesign K conditioning before formal training.** Split ray construction, ray embedding, shape branch,
   scale branch, and uncertainty head timing. Test cached rays, compact normalized-intrinsics FiLM, and low-rank camera
   conditioning. A candidate must pass a validation-only latency budget before it can enter another promotion gate.
4. **Preserve the reference's scale path.** The result suggests a hybrid: factorized centered shape plus the monolithic
   global-scale estimator or a bounded residual around it. Select scale losses and calibration only on training/validation
   sensor families, then freeze them before a new test.
5. **Treat uncertainty as a separate objective.** Retain validation-only calibration, but compare richer prespecified
   variance calibration and a reference-uncertainty ablation. Gate calibrated NLL and reliability in addition to AUSE.
6. **Measure the actual deployment path.** Add randomized/interleaved end-to-end encoder-plus-head latency, peak memory,
   and head-component profiles. Keep the current synchronized head-only metric for continuity, but label it as such.
7. **Increase independent domain evidence before increasing same-domain seeds.** More cameras/sensors or a second fresh
   dataset will reduce the dominant generalization uncertainty more than additional optimizer seeds on the consumed split.

The immediate engineering task is therefore a train/validation-only latency and K-control audit, followed by a frozen
Phase-2f protocol on new data. No additional formal SUN RGB-D Kinect-v2 promotion run is justified from this result.
