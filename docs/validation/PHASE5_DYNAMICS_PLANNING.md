# Phase 5 validation plan: action-conditioned dynamics and verified planning

Status: proposed external validation plan, 2026-06-30. This document does not claim a trained world model or safe robot
policy.

## Decision and claim boundary

Phase 5 has two sequential gates:

1. offline action-conditioned dynamics must predict held-out consequences better than action-free and simple transition
   baselines, with useful uncertainty;
2. a frozen model/planner must run repeated closed-loop episodes in one named simulator using
   `propose → execute → observe → verify → update → replan`.

Offline prediction cannot establish control quality, and simulator success cannot establish real-world or hardware safety.
No action from this phase may be sent to LeRobot, ROS 2, or physical hardware.

## Current evidence

| Evidence | Result | Missing evidence |
|---|---|---|
| `ActionConditionedLatentDynamics` | Deterministic action-sensitive contract backend and trainable residual MLP interface return finite rollout, uncertainty, and value tensors. | The learned backend is randomly initialized and has no prediction-quality claim. |
| `CEMPlanner` | Seeded bounded action sequences and uncertainty penalty are tested. | No trained dynamics, constraints, or task-level control evidence. |
| `VerifiedTaskPlanner` + `MockRobot` | One injected pick failure is attributed and recovered; confidence below 0.8 fails closed. | A deterministic mock is not a simulator benchmark. |
| Initial Phase 5 run | Real V-JEPA feature-to-CEM handoff was finite; fixture task success/recovery were 1.0. | One episode and a saved feature are not learned closed-loop evidence. |
| W&B run [`8kctk4mt`](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/8kctk4mt) | Finished run with planning trace, subgoal table, CEM diagnostics, and failure attribution. | No offline dataset training or repeated simulator trials. |

## External datasets and benchmark roles

| Source | Role in Phase 5 | Access and license boundary |
|---|---|---|
| [D4RL](https://arxiv.org/abs/2004.07219) via the [Farama repository](https://github.com/Farama-Foundation/D4RL) or [Minari reproduction](https://minari.farama.org/datasets/D4RL/index.html) | Low-dimensional offline transition sanity lane. Use fixed-observation/action/reward/termination datasets to test action conditioning, multi-step error, uncertainty, and offline-policy baselines before visual robotics. | Unless otherwise noted, D4RL datasets are CC BY 4.0 and code is Apache-2.0. Record the exact environment and dataset version; do not mix legacy Gym and current environment semantics. |
| [RoboNet](https://www.robonet.wiki/) | Primary visual offline-dynamics lane: more than 15M robot-interaction frames from 113 viewpoints and seven robot platforms, with end-effector/gripper actions. Evaluate cross-view/platform action-conditioned latent rollouts. See the [primary paper](https://arxiv.org/abs/1910.11215). | Use the official data path and record every shard/hash. The landing page describes the release as open source but does not provide a sufficient data-license statement for redistribution; legal/license verification is a blocking preflight, and raw frames must not be mirrored to W&B. |
| [ManiSkill3](https://www.maniskill.ai/ManiSkill3) | Named closed-loop simulator and secondary offline-demonstration lane. Freeze a version and evaluate repeated visual manipulation episodes such as PickCube, StackCube, and PegInsertionSide with the same action/observation contract. | Use the [official repository](https://github.com/mani-skill/ManiSkill), [documentation](https://maniskill.readthedocs.io/en/latest/), and [primary paper](https://arxiv.org/abs/2410.00425). Record the code license plus every downloaded task asset/demo license separately; do not assume the code license covers third-party assets. |
| [LIBERO](https://libero-project.github.io/) | Independent closed-loop transfer lane after the ManiSkill checkpoint, thresholds, and planner are frozen. Use fixed Spatial, Object, Goal, and Long suites to test language/task transfer rather than retuning. | Pin the [official repository](https://github.com/Lifelong-Robot-Learning/LIBERO), benchmark release, task files, demonstrations, assets, and dependency stack. Audit code, dataset, and third-party asset terms separately before download or redistribution. |

Dataset ingestion is blocked until source URL, terms, version/commit, environment ID, observation/action schema, split,
bytes, and SHA-256 are in an immutable manifest. Training never contacts the simulator in the offline gate.

## Frozen evaluation lanes

| Lane | Protocol | Required output |
|---|---|---|
| P0 contract | Existing tensor, CEM, verification, failure-injection, and determinism tests. | Finite bounded outputs, exact seed replay, explicit evidence, and fail-closed uncertainty. |
| P1 D4RL offline | Train/validation/test trajectory splits grouped by original episode. No simulator interaction during selection. | One- and multi-step state/reward/termination predictions, calibration, offline-policy score, and receipts. |
| P2 RoboNet offline | Split by trajectory and hold out at least one camera/platform stratum; actions and timing are normalized once and hashed. | H-step latent predictions, action sensitivity, uncertainty, cross-view/platform breakdown, and rollout visuals. |
| P3 ManiSkill demos | Train only from a frozen demonstration manifest; simulator is used for final evaluation, not model selection. | Offline imitation/dynamics metrics and a frozen checkpoint/config. |
| P4 ManiSkill closed loop | Repeated fixed-seed episodes for at least three tasks and predefined perturbation/failure strata. | Success, verified progress, collisions, false verification, replans, recovery, latency, videos, and trace receipts. |
| P5 LIBERO transfer | Freeze the complete ManiSkill-selected system, then evaluate fixed LIBERO suites without model, prompt, threshold, controller, or calibrator tuning. | Per-suite/task success, progress, safety, verification/recovery, language-condition controls, videos, and trace receipts. |

The closed-loop evaluator loads a frozen checkpoint exactly once. Formal seeds, task versions, initial states, camera
settings, action bounds, horizon, retry budget, confidence threshold, and termination rules are committed before launch.

Before any P4/P5 formal episode runs, jointly freeze a Phase-5 component manifest and a disjoint Phase-6 system manifest
at the task/scene/suite level, not merely by changing random seeds. Phase-5 jobs must be unable to enumerate or load the
Phase-6-reserved IDs. If a benchmark lacks enough disjoint tasks/scenes, its reused portion is tagged
`consumed_regression` in Phase 6 and cannot support an independent composed-system claim.

## Baselines and ablations

- copy-last-state and action-free dynamics;
- linear action-conditioned delta and the current residual MLP;
- action-shuffled and time-shuffled negative controls;
- behavior cloning, recurrent/Transformer behavior cloning, and one maintained offline RL baseline (IQL or CQL) on D4RL;
- open-loop plan, receding-horizon CEM, CEM without uncertainty penalty, and oracle simulator transition upper bound;
- symbolic planner without memory, without fresh verification, without replanning, and with fixed versus calibrated
  confidence thresholds;
- identical model with real actions versus zero/shuffled actions to prove that reported prediction uses control input.

All methods share representations, dataset manifests, action normalization, evaluation seeds, and compute budgets.

## Metrics

### Offline dynamics

- H=1/5/10 latent cosine error and normalized MSE, reported per step and as rollout area under the error curve;
- reward and value error, termination AUROC/AUPRC, and optional decoded PSNR/SSIM/LPIPS only when a frozen decoder exists;
- action sensitivity: paired error increase under shuffled/zero actions and counterfactual separation for distinct actions;
- uncertainty NLL/calibration error, coverage-risk, and AUROC for top-decile rollout failures;
- D4RL normalized return for policy baselines, always paired with offline model metrics;
- cross-camera/platform/task macro means, bootstrap intervals, training throughput, peak memory, and inference p50/p95/p99.

### Repeated closed loop

- task success and verified subgoal progress with Wilson 95% intervals per task;
- false-verification rate, collision/contact-limit violations, invalid actions, timeouts, and catastrophic failures;
- attributed failure counts, recovery success, replans, attempts, steps, path/action cost, and completion time;
- planner latency p50/p95/p99, simulator steps/second, GPU memory, uncertainty before failures, and calibration;
- paired success delta versus behavior cloning/open-loop/CEM ablations on identical episode seeds.

The episode—not frame or simulator environment—is the statistical unit. Report every seed, including crashes and timeouts.

## Staged TODO

- [ ] **P0 — contract and safety hardening:** add cancellation/timeouts, action-schema validation, checkpoint double reload,
   collision/constraint hooks, simulator-reset determinism, and terminal receipt tests.
- [ ] **P1 — sealed data preparation:** approve licenses; build D4RL/RoboNet/ManiSkill/LIBERO manifests; normalize actions and timing;
   verify episode-disjoint splits, reserve task/scene/suite IDs for Phase 6, and prohibit simulator calls from offline jobs.
- [ ] **P2 — baseline reproduction:** reproduce copy/action-free/linear dynamics and BC plus one D4RL offline RL reference.
- [ ] **P3 — offline pilots:** establish horizon, capacity, calibration, latency, and memory bounds. Freeze thresholds, model
   variants, tasks, seeds, and formal promotion rules in a hashed preregistration.
- [ ] **P4 — formal offline arrays:** train/evaluate model × dataset × seed jobs; aggregate paired horizon and calibration
   results; select at most one checkpoint without looking at closed-loop formal seeds.
- [ ] **P5 — simulator adapter qualification:** pin ManiSkill/SAPIEN/Vulkan versions, reproduce official task success with a
   reference controller, and verify resets, observations, actions, and safety events.
- [ ] **P6 — repeated closed loop:** execute the frozen planner across tasks, seeds, and perturbations; no tuning or retries
   outside the preregistered policy.
- [ ] **P7 — LIBERO transfer:** evaluate the frozen ManiSkill-selected system on fixed LIBERO suites without retuning; keep
   suite/task results separate and run language-shuffle/goal controls.
- [ ] **P8 — aggregation and postflight:** validate all scheduler/W&B/local receipts, compute confidence intervals and failure
   strata, render reports, and fail closed on missing episodes.

## Logging and visual evidence

Use online W&B under `jepa4d-worldmodel`, grouped by immutable execution ID. Every run logs Git/config/dataset/checkpoint
hashes, scheduler and GPU identity, simulator version, task, seed, parents, restart count, and output hashes.

Required W&B panels and matching local artifacts:

- one-/multi-step train and held-out error by horizon, dataset, task, camera, and platform;
- real-action versus zero/shuffled-action curves and counterfactual rollout grids;
- uncertainty reliability, coverage-risk, and error-versus-uncertainty plots;
- D4RL normalized-return tables beside dynamics metrics;
- simulator success with Wilson intervals, progress by subgoal, collisions, false verification, replans, and recovery;
- episode action/state/uncertainty timelines and failure-attribution Sankey/confusion views;
- side-by-side ground-truth/predicted offline rollouts and closed-loop videos with task/seed overlays;
- latency/throughput/GPU-memory panels and seed-level result tables.

Local outputs include `manifest.json`, `config.yaml`, checkpoint plus normalization identity, `metrics.json`, seed-level
CSV/JSONL, curves `.npz/.png`, calibration plots, approved MP4s, `report.html`, scheduler receipt, and postflight receipt.
Raw RoboNet media and licensed simulator assets are not W&B artifacts.

## Slurm execution policy

- Login nodes perform environment builds, asset manifests, and submission only; experiments run through `sbatch`.
- Account: `edgeai_tao-ptm_image-foundation-model-clip`.
- Partitions: `polar4,polar3,polar,batch_block1,grizzly,batch_block2,batch_block3`.
- Maximum job time is four hours; long training is checkpointed into independently receipted jobs rather than extending
  the wall time.
- Submit tests, asset/cache audit, baselines, offline arrays, qualification, simulator arrays, aggregation, selection, and
  postflight held. Atomically write the logical dependency graph before release; use `afterok` dependencies.
- Arrays use `%8` or less and the entire execution must remain at no more than eight expanded `RUNNING` tasks. Record
  `COMPLETING` separately, continuously audit concurrency, and store the maximum in postflight.
- No silent relaunch: an operator requeue retains the logical ID and records reason, old/new allocation, restart count,
  empty-output audit, and receipt. Missing episodes, nonzero exits, offline W&B, or hash mismatches fail closed.

## Promotion gates

Phase 5 advances to simulator-validated only when all gates pass:

1. contract, action-bound, determinism, reload, cancellation, timeout, and fail-closed verification tests pass;
2. official/reference baseline reproduction is within a preregistered tolerance;
3. action-conditioned dynamics beats copy-last and action-free baselines at H=1/5/10 with paired 95% bootstrap confidence
   intervals above zero, and shuffled actions significantly degrade prediction as expected;
4. uncertainty has positive preregistered ranking/calibration value and no non-finite formal rollout;
5. on every named ManiSkill task, the frozen closed-loop method improves over its selected BC/open-loop control, its task
   success lower confidence bound clears the threshold frozen before formal evaluation, and no task is hidden by averaging;
6. false verification, collisions, invalid actions, timeouts, and catastrophic failures remain within frozen safety
   bounds; any severe uncontained event fails the execution;
7. latency, memory, retry, and replan budgets pass, all expected seeds have scheduler/W&B/local receipt triples, and
   expanded `RUNNING` never exceeds eight;
8. independent postflight validates licenses, manifests, split isolation, hashes, exact checkpoint identity, and claims.

LIBERO is a separate transfer gate after ManiSkill selection. Passing ManiSkill but failing the frozen LIBERO criteria
supports only within-simulator development evidence and blocks a cross-benchmark planning claim.

Passing these gates supports only the statement that a frozen model/planner has offline predictive value and repeated
closed-loop success in the named simulator/version. It does not authorize transfer to another simulator, a real robot, a
human environment, or safety-critical operation.
