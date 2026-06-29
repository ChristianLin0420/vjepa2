# Phase 5 design: latent dynamics and verified planning

## Scope and claim boundary

Phase 5 adds an initial closed-loop planning substrate. It does not claim that a learned world model is trained, that
fixture confidence is calibrated, or that the mock robot establishes hardware safety. The implementation makes those
future integrations explicit while exercising the complete control loop offline.

The planner follows `propose → execute → observe → verify → update → replan`. Symbolic code receives task-level evidence,
never raw V-JEPA tokens. Continuous MPC remains behind the latent-dynamics boundary.

## Latent dynamics and MPC

`ActionConditionedLatentDynamics` has two backends:

- `deterministic` is an action-sensitive, uncertainty-producing contract backend for CI;
- `learned` is a residual MLP with action/proprioception conditioning and uncertainty/value heads. It is a training and
  checkpoint integration boundary, not a pretrained V-JEPA 2-AC or JEPA-WM claim.

Both consume `[B,N,C]` tokens and `[B,A]` actions. Multi-step rollout consumes `[B,H,A]` and preserves the complete token,
uncertainty, and value trajectory. `CEMPlanner` samples bounded action sequences, penalizes action magnitude and predicted
uncertainty, and returns the first receding-horizon action plus the complete proposal. A local seeded generator makes the
contract benchmark reproducible on CPU and CUDA.

## Symbolic execution

`TaskGraph` contains dependency-checked typed `Subgoal` records. Each subgoal owns its action, target, parameters,
verification condition, attempt count, evidence, failure reason, and state. A subgoal becomes verified only from a fresh
`RobotObservation` whose condition confidence clears the configured threshold.

`VerifiedTaskPlanner` executes one ready subgoal at a time. Control and verification failures receive distinct attribution.
`ReplanningPolicy` bounds attempts and total replans; exhausted recovery fails closed. Successful evidence can also be
written to `WorldModelQueryAPI.task_state`.

The behavior-tree module supplies typed status, callback, and sequence nodes plus the domain node vocabulary. The current
task executor uses the task graph as its scheduling authority; richer asynchronous trees can be layered over the same
robot and verification contracts without changing memory APIs.

## Mock robot and gate

`MockRobot` owns observable object/holding state, supports search/pick/place/observe, and can inject a named one-shot control
failure. The planning smoke episode finds a mug, fails its first pick, attributes the control failure, replans, retries,
verifies holding state, places the mug, and verifies its destination.

The initial Phase-5 gate requires:

- all subgoals have explicit positive evidence;
- low-confidence observations are rejected at the safety threshold;
- an injected failure is attributed and recovered within bounded retries;
- the trace records actions, observations, verification, uncertainty, and replans;
- CEM executes reproducibly and within action bounds on the available A100.

## Deferred work

- integrate and evaluate a real V-JEPA 2-AC or JEPA-WM checkpoint;
- train/calibrate uncertainty and value heads on action-conditioned episodes;
- connect CEM proposals to symbolic skills and collision constraints;
- add parallel/fallback/decorator behavior-tree nodes and asynchronous execution;
- benchmark in a named simulator before enabling LeRobot or ROS 2;
- add cancellation, timeouts, operator intervention, and hardware safety envelopes.
