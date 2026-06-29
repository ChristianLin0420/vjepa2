# Phase 5 initial verified-planning experiment

## Experiment metadata

| Field | Value |
|---|---|
| Experiment ID | `2026-06-29-phase5-verified-recovery-v1` |
| Stage / status | `dynamics + planning / complete` |
| Evidence level | `contract-only` |
| Hardware | `NVIDIA A100 80GB PCIe` for CUDA CEM smoke; deterministic robot fixture on CPU |
| Decision | Keep the closed-loop contracts; next integrate real learned dynamics and a named simulator. |
| W&B | [8kctk4mt](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/8kctk4mt) |

## Objective

Establish that Phase 5 advances beyond open-loop action generation: every subgoal must carry fresh evidence, uncertainty
must fail closed, execution failures must be attributed, and a bounded replan must recover when recovery is possible.

## Configuration

- task: find a mug, pick it, and place it on a table;
- injected failure: first `pick:mug` command fails at the control stage;
- observation confidence: 0.95;
- verification threshold: 0.8;
- per-subgoal attempt limit: 2;
- total replan limit: 4;
- real V-JEPA handoff: local ViT-B checkpoint, one RGB image, `[1,1,1,576,768]` dense tokens on CUDA;
- CEM CUDA smoke: pooled real token, horizon 3, population 64, three iterations, seven-dimensional bounded actions.

## Results

| Metric | Result |
|---|---:|
| Task success | 1.0 |
| Verified subgoal progress | 1.0 |
| Failure attribution present | 1.0 |
| Recovery success | 1.0 |
| Replans | 1 |
| Verification actions | 3 |
| Planning trace events | 9 |

The trace first verifies `visible:mug`. The injected pick failure is labeled `control / injected_control_failure`, after
which the planner retries exactly once. Fresh observations then verify `holding:mug` and `at:mug:table`, each at confidence
0.95. A separate test lowers observation confidence to 0.6 and confirms that the 0.8 threshold rejects the condition and
exhausts bounded recovery rather than marking a false success.

The CUDA health check reports PyTorch 2.5.1+cu121, one A100, compute capability 8.0, and 79.15 GiB usable memory. The local
V-JEPA 2.1 ViT-B checkpoint produced finite `[1,1,1,576,768]` features on `cuda:0` in 3.79 seconds end to end (0.50-second
model forward). Seeded CEM consumed the pooled real feature, returned a bounded `[3,7]` action proposal, and reported
finite score `-0.9005` and predicted uncertainty `0.0235`. This validates the representation-to-planning handoff only.

A subsequent W&B logging run reused that real feature artifact and executed CEM plus verified recovery on CPU because the
A100 dropped back to PCI revision `ff` immediately before logging. Run `8kctk4mt` contains planning event/subgoal
tables, confidence and uncertainty, cumulative replans, failure attribution, recovery metrics, CEM diagnostics, system
telemetry, and the JSON trace artifact. It was synced to the `jepa4d-worldmodel` project and verified in the finished
state with three logged artifacts.

## Verification

- `scripts/check_cuda.py`: pass;
- CUDA latent CEM smoke: pass;
- real V-JEPA 2.1 CUDA feature-to-CEM handoff: pass;
- Phase-5 W&B run [8kctk4mt](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/8kctk4mt): synced and verified;
- Ruff over `jepa4d`: pass;
- mypy over `jepa4d`: pass;
- combined upstream and JEPA-4D pytest trees: 73 passed, with 20 third-party/deprecation warnings;
- planner CLI with injected failure: pass;
- planner HTTP replan endpoint: pass.

## Claim boundary and limitations

- The deterministic dynamics backend validates conditioning, rollout, uncertainty, value, and MPC contracts only.
- The learned residual model is randomly initialized; no learned prediction-quality claim is made.
- Confidence values are controlled fixture inputs, not calibrated probabilities.
- One deterministic episode is not simulator, benchmark, or hardware evidence.
- LeRobot and ROS 2 remain unimplemented until learned-dynamics and simulator gates pass.

## Next experiments

1. Load and evaluate a real V-JEPA 2-AC or JEPA-WM checkpoint against action-free and open-loop baselines.
2. Calibrate dynamics uncertainty on held-out action-conditioned episodes.
3. Connect MPC actions to constrained symbolic skills in a named simulator.
4. Run repeated randomized failures and report success, false verification, recovery, collision, and latency intervals.

## Downloaded W&B record snapshot

The promoted run record was downloaded through the W&B API on 2026-06-29 and verified in `finished` state. Its persisted
summary reports task success 1.0, subgoal progress 1.0, one attributed failure, one replan, three verification actions,
nine events, MPC score -0.901793, and predicted uncertainty 0.024726. W&B lists three logged artifacts: event/subgoal
tables and the planning trace. The remote run used the saved real V-JEPA feature but executed CEM/recovery on CPU, as
already noted above.
