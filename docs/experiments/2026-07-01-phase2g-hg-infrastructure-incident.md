# Phase 2g-A HG infrastructure incident

Execution `p2gq-0a81684b-20260701T050817Z-2a8b389c` was the first formal Phase 2g-A attempt from immutable commit
`0a81684b32815de2b312c8ebd6d1c245825ba6e8`. Gates T, O, C, and Q completed successfully, and all 48 H tuning tasks
produced SUCCESS receipts. HG then failed before reading a tuning receipt or producing scientific output.

The failure was scheduler-contract-only. The Slurm `afterok` dependency released HG while the final array task was still
visible as `COMPLETING`, and the parent verifier queried `sacct JobIDRaw`. On this cluster, `JobIDRaw` is a distinct
allocation identifier for array elements while the frozen graph records the logical `JobID` form such as
`29699312_0`. Forty-seven elements had distinct raw allocation IDs and the last had the base array ID; none of the 48
`JobIDRaw` values matched the graph's logical element IDs, even after accounting settled.

No tuning metric values were inspected while diagnosing this failure. Only scheduler states, SUCCESS/receipt counts,
file presence, and the failing stack trace were observed. One same-ID HG requeue was requested after the initial
COMPLETING-race diagnosis, then held and cancelled before execution when the deterministic JobID mismatch was found.
The original external logs and scheduler history are retained; downstream F, V, S, G, and Z tasks never ran.

Recovery is a new immutable execution lineage with no scientific changes. The scheduler verifier uses logical `JobID`,
waits boundedly for accounting to settle, and carries cluster-realistic regression tests. Data membership, cached model
features, architectures, seeds, learning rates, epochs, losses, metrics, gates, and selection rules remain unchanged.
