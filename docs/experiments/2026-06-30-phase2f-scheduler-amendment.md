# Phase 2f scheduler-throttling amendment

## Administrative status

| Field | Value |
|---|---|
| Amendment timestamp | `2026-06-30T04:17:19Z` |
| Authority | User-mandated operational safety change |
| Reason | Limit Phase-2f scheduler pressure and concurrent GPU allocations to at most eight while preserving the frozen experiment |
| Applies to | The next new Phase-2f execution only, after the amended scheduler implementation and focused tests pass |
| Superseded execution | `05dd7e7e-20260630T040538Z`, canceled and archived as `cancelled-for-8-task-throttle` before the cache job completed or produced its receipt/SUCCESS marker; unvalidated partial files are not reused |
| Scientific effect | None |

The original preregistration,
[2026-06-29-phase2f-scale-camera-preregistered.md](2026-06-29-phase2f-scale-camera-preregistered.md), remains
immutable and authoritative for every scientific choice. This amendment changes only how its logical jobs are represented
to Slurm. It does not revise or reopen any result from an earlier execution.

## Frozen scientific contract

This amendment makes **no change** to datasets, sample selection, rotations, transforms, target opacity, model arms,
parameter limits, losses, gradient firewall, seeds, epochs, checkpoint selection, calibration, metrics, latency protocol,
bootstrap, pilot/formal qualification, survivor selection, external-final gate, claim boundary, or W&B artifact contract.
In particular:

- the canonical DAG still contains exactly 73 logical tasks with the same labels, scientific parents, outputs, and skip
  semantics;
- latency still uses 12 independent allocations and the frozen `1.10x` upper-CI gate;
- formal training still declares all `4 arms x 4 rotations x 3 seeds`, with disqualified tasks writing zero-step skip
  receipts;
- DIODE remains sealed unless the unchanged selector authorizes exactly one survivor; and
- no artifact from the canceled execution is reused in the replacement execution.

## Amended scheduler representation

The 73 logical tasks are represented by exactly 12 held Slurm submissions:

| Submission | Logical tasks | Scheduler form | Maximum active elements |
|---|---|---|---:|
| `T` | tests | scalar | 1 |
| `A` | sealed-archive byte audit | scalar | 1 |
| `C` | SUN development cache | scalar | 1 |
| `Q` | static/cache audit | scalar | 1 |
| `L` | `L00`-`L11` latency replicas | array `0-11%8` | 8 |
| `LA` | latency aggregate | scalar | 1 |
| `P` | `P0`-`P3` pilots | array `0-3%4` | 4 |
| `PG` | pilot gate | scalar | 1 |
| `F` | 48 arm/rotation/seed tasks | array `0-47%8` | 8 |
| `S` | development selector | scalar | 1 |
| `E` | external-final guard/evaluator | scalar | 1 |
| `Z` | strict postflight | scalar | 1 |

Every submission is created with `--hold`. The canonical graph is written atomically only after all 12 scheduler IDs are
known, and all submissions are then released. Latency and formal arrays carry the literal `%8` concurrency cap. Pilot
has only four elements. The latency array waits for both the static audit and independent asset-seal submission, and all
later arrays are sequential. Dependency ordering and the array caps therefore make eight a global upper bound on
simultaneously active Phase-2f tasks, not merely a per-array convention.

All 12 scheduler job names use distinct stage-specific `p2f8` names so the throttled execution cannot be confused with a
73-submission Phase-2f run in `squeue`, `sacct`, logs, provenance, or W&B. The submitter must reject duplicate stage names
and must attest that it issued exactly 12 `sbatch` calls.

## Logical-task and array identity

The dependency graph remains logical-task-addressed. Its scheduler-ID grammar is `N` for a scalar submission and
`N_TASK` for an array-backed logical task, where both `N` and `TASK` are decimal integers. Thus array rows use identities
such as `12345_7`, while scalar rows use identities such as `12345`. Scheduler identity is the pair encoded by
`N_TASK`, not the shared base ID alone.

Runtime provenance and parent validation must preserve both identities. An array element may satisfy only its registered
logical label, output path, and parent set. Duplicate `N_TASK` identities, an out-of-range task index, an invalid
non-numeric suffix, an incomplete Slurm array identity, or a mismatch with `SLURM_ARRAY_JOB_ID` and
`SLURM_ARRAY_TASK_ID` fails closed. Shared array base IDs are expected and do not weaken per-logical-task receipt,
SUCCESS, W&B, hash, or postflight validation.

## Cancellation and relaunch boundary

The prior corrected run `05dd7e7e-20260630T040538Z` was canceled for this administrative throttle before the development
cache produced its receipt or SUCCESS marker. Its partial cache files are not validated parents and cannot be reused. The
replacement must use a new execution ID, clean committed source, a new test receipt, a new 12-submission graph, new W&B
runs/artifacts, and new output roots.

This cancellation is not a scientific threshold miss and conveys no model result. The original preregistration remains
frozen; this dated amendment is the sole record of the scheduler-only change.

## Required verification before release

Focused tests must prove all of the following:

1. the dedicated array-dispatch sbatch wrapper exists, requests the frozen account/partition/GPU resources, and maps only
   registered latency, pilot, and formal indices;
2. the submitter makes exactly 12 held submissions, uses distinct `p2f8` stage names, declares `0-11%8` latency and
   `0-47%8` formal arrays, and writes the graph before release;
3. the graph still contains 73 logical tasks while accepting shared scheduler IDs only for distinct registered array
   task IDs;
4. the parser accepts exactly scalar `N` and array `N_TASK` numeric identities and rejects malformed task suffixes;
5. graph, runtime provenance, and parent validation compare the full scheduler identity, including
   `SLURM_ARRAY_TASK_ID` when present; and
6. static analysis demonstrates a global Phase-2f active-task ceiling of eight.

Only a new run created after these checks pass may be used for the eventual Phase-2f result record.
