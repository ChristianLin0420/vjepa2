# Phase 2b Slurm jobs

These entrypoints keep the Phase 2b protocol on allocated GPU nodes and fail
closed when tests, CUDA, assets, the pinned dataset, or final artifacts are
incomplete. They do not submit jobs themselves.

All jobs request the project account and the allowed partition fallback list.
The formal job requests one node, one task, one GPU, 16 CPUs, 220 GiB, and the
four-hour maximum. Test and preflight jobs use the same hardware shape with
shorter limits.

## Reproducible login-node environment and asset setup

Run the preparation script from the repository root on the login node. It
creates a Python 3.12 Conda prefix at `.conda-gpu`, installs the constrained
CUDA-enabled stack plus test/reporting tools, and downloads and checksums the
public model and TUM assets. It performs no GPU work:

```bash
export JEPA4D_REPO_ROOT="$PWD"
bash slurm/prepare_phase2b_login.sh
```

Downloads finalize serially on node-local `/tmp` and copy verified bytes into
Lustre because Hugging Face's atomic rename can block there.
`phase2b_setup.sbatch`
remains an all-in-one recovery option, but it is not needed after login
preparation succeeds. CUDA allocation and sustained compute are always checked
by the submitted test, preflight, and formal jobs.

The V-JEPA model's `main` ref is resolved to an immutable commit before
download; the resolved commit and weight hashes are persisted. The V-JEPA
compatibility source and VGGT weights have explicit default commits. Override
them with `JEPA4D_VJEPA_REVISION`, `JEPA4D_VJEPA_IMPLEMENTATION_REVISION`, or
`JEPA4D_VGGT_REVISION`. The VGGT weights carry their upstream CC-BY-NC-4.0
license.

The defaults produce these paths for subsequent jobs:

```bash
export JEPA4D_PYTHON="$PWD/.conda-gpu/bin/python"
export JEPA4D_ASSET_ROOT="$PWD"
export JEPA4D_DATASET_ROOT="$PWD/checkpoints/datasets/rgbd_dataset_freiburg1_xyz"
export JEPA4D_TUM_ARCHIVE="$PWD/checkpoints/datasets/rgbd_dataset_freiburg1_xyz.tgz"
```

Paths can instead live elsewhere on Lustre:

```bash
export JEPA4D_PYTHON=/absolute/path/to/python3.12
export JEPA4D_DATASET_ROOT=/absolute/path/to/rgbd_dataset_freiburg1_xyz
export JEPA4D_TUM_ARCHIVE=/absolute/path/to/rgbd_dataset_freiburg1_xyz.tgz

# Either set one root containing checkpoints/<names below> ...
export JEPA4D_ASSET_ROOT=/absolute/path/to/assets-root

# ... or set all three exact paths.
export JEPA4D_VJEPA_CHECKPOINT=/absolute/path/to/phase2b_assets/vjepa2.1-vitb-fpc64-384
export JEPA4D_VJEPA_IMPLEMENTATION=/absolute/path/to/phase2b_assets/vjepa21_hf_impl
export JEPA4D_VGGT_CHECKPOINT=/absolute/path/to/phase2b_assets/VGGT-1B
```

`JEPA4D_ENV_ACTIVATE` may point to an activation script instead of setting the
interpreter directly. Model loading is offline after setup, so all assets must
be local. Configure W&B authentication in the shared home directory (for
example, the existing netrc credential) or via the scheduler environment;
never put a key in a tracked script. Preflight creates a real online run and
waits for its artifact receipt. Formal training is hard-locked to online W&B.

## Safe launch order

Use stable test and preflight receipt paths. Preflight proves that the exact
repository and Python environment passed the submitted unit/static/CUDA tests;
formal training then recomputes and compares the code, environment, manifest,
archive, selected extracted frames, and every model/source asset hash:

```bash
mkdir -p outputs/phase2b-gates
export JEPA4D_TEST_REPORT="$PWD/outputs/phase2b-gates/tests.json"
export JEPA4D_PREFLIGHT_REPORT="$PWD/outputs/phase2b-gates/preflight.json"
export JEPA4D_WANDB_ENTITY=crlc112358

test_job=$(sbatch --parsable slurm/phase2b_tests.sbatch)
preflight_job=$(sbatch --parsable --dependency="afterok:${test_job}" slurm/phase2b_preflight.sbatch)
```

Inspect the test logs and the JSON report. Only after both jobs pass, submit the
formal run (the dependency is an additional scheduler-side guard):

```bash
export JEPA4D_OUTPUT_DIR="$PWD/outputs/jepa4d_phase2b/tum_rgbd_v1"
export JEPA4D_RUN_NAME=phase2b-jepa-geometry-distillation-v1
export JEPA4D_WANDB_PROJECT=jepa4d-worldmodel
train_job=$(sbatch --parsable --dependency="afterok:${preflight_job}" slurm/phase2b_train.sbatch)
printf 'tests=%s preflight=%s training=%s\n' "$test_job" "$preflight_job" "$train_job"
```

Every job writes scheduler stdout/stderr in the submission directory and a
structured directory under `outputs/slurm_logs/`. The latter contains the git
revision/status, Python package inventory, Slurm allocation, full NVIDIA report,
CUDA health JSON, continuously sampled GPU telemetry CSV, and a `SUCCESS`
marker only when every gate passes. Preflight adds extraction/model checksums,
B=N/V=1 chunk-invariance comparisons, real V-JEPA/VGGT tensor summaries, a
one-step optimizer/checkpoint-reload test, a local HTML report, and the online
W&B artifact receipt. Formal training is locked to exactly 60 epochs and adds
strict validation, continuous telemetry, SHA-256 hashes for every output and
all nine learned checkpoints, a self-contained diagnostic HTML report, online
tables/curves/media, and a backend-confirmed W&B artifact version/digest.

Useful overrides include `JEPA4D_LOG_ROOT`, `JEPA4D_MANIFEST`,
`JEPA4D_GPU_MONITOR_INTERVAL`, and `JEPA4D_WANDB_ENTITY`. Existing nonempty
result directories are rejected; always choose a fresh output path.
`JEPA4D_EPOCHS` cannot override the formal 60-epoch protocol, preflight cannot
be bypassed, asset hashes cannot be weakened to metadata, and formal
`WANDB_MODE` cannot be changed from `online`.

## Phase 2c cross-sequence learned-fusion gate

Phase 2c has a separate receipt namespace and Slurm chain so that the completed
Phase 2b evidence remains immutable. Its bundle contract contains exactly five
TUM sequences split by camera family into two training, one validation, and two
held-out test sequences, with 128/64/128 selected frames. Formal output requires
one teacher plus four three-seed probe variants, twelve checkpoints, four
training-only normalization artifacts, sixty epochs per probe, and online W&B.

Build or refresh the same Python 3.12 environment and download/check the bundle
from the login node. This performs package installation, downloads, hashing,
and safe extraction, but no CUDA work:

```bash
export JEPA4D_REPO_ROOT="$PWD"
export JEPA4D_DATASET_PARENT="$PWD/checkpoints/datasets"
export JEPA4D_MANIFEST="$PWD/jepa4d/config/benchmarks/manifests/tum_rgbd_phase2c_cross_sequence_v1.yaml"
bash slurm/prepare_phase2c_login.sh
```

Use a unique gate directory for each exact repository revision to prevent two
submission chains from replacing one another's receipts:

```bash
gate_id="$(git rev-parse --short=12 HEAD)-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$PWD/outputs/phase2c-gates/$gate_id"
export JEPA4D_TEST_REPORT="$PWD/outputs/phase2c-gates/$gate_id/tests.json"
export JEPA4D_PREFLIGHT_REPORT="$PWD/outputs/phase2c-gates/$gate_id/preflight.json"
export JEPA4D_WANDB_ENTITY=crlc112358

test_job=$(sbatch --parsable slurm/phase2c_tests.sbatch)
preflight_job=$(sbatch --parsable --dependency="afterok:${test_job}" slurm/phase2c_preflight.sbatch)
```

Only submit formal training after inspecting the passing test and preflight
receipts. The scheduler dependency remains as an additional guard:

```bash
export JEPA4D_OUTPUT_DIR="$PWD/outputs/jepa4d_phase2c/tum_rgbd_cross_sequence_${gate_id}"
export JEPA4D_RUN_NAME="phase2c-cross-sequence-learned-fusion-${gate_id}"
export JEPA4D_WANDB_PROJECT=jepa4d-worldmodel
train_job=$(sbatch --parsable --dependency="afterok:${preflight_job}" slurm/phase2c_train.sbatch)
printf 'tests=%s preflight=%s training=%s\n' "$test_job" "$preflight_job" "$train_job"
```

The test receipt binds the full repository and Python distribution set to a
sustained CUDA check. Preflight recomputes those identities, fully hashes every
model/source asset and every selected extracted RGB/depth file against its
pinned archive, checks mixed-sequence B=N/V=1/T=1 inference at chunk sizes one
and eight, exercises exact-final initialization and a finite-gradient optimizer
step for residual fusion, reloads its checkpoint, renders a diagnostic report,
and waits for an online W&B artifact receipt. Formal authorization recomputes
all identities before the runner starts. Postflight requires exactly thirteen
result rows, twelve checkpoints, 720 history rows, complete per-frame and
per-sequence metrics, a self-contained interactive report, a bijective artifact
manifest, and a backend-confirmed online W&B receipt.

## Governed consumed-TUM official-mini smoke

The post-Wave-A smoke is intentionally narrower than the historical Phase 2b
training chain. It replays only the eight already-consumed
`freiburg1_xyz` test frames, extracts them into ephemeral temporary storage
selected by Python from the hash-verified archive, and runs a locally supplied
VGGT checkpoint whose file-tree identity is checked for stability and bound
into the receipts. It publishes aggregate-only diagnostics to online W&B. This
is an integrity and regression check, not fresh held-out evidence, an
architecture comparison, or authorization for formal training.

The submission wrapper is the only supported entrypoint. It rejects a dirty or
uncommitted repository, any pre-existing output path, unsafe names, and eight or
more active jobs/tasks owned by the authenticated user. Its `squeue -r` query
expands array elements. A lock in the user's shared home serializes invocations
of this supported wrapper across login nodes that mount the same home; it cannot
serialize unrelated submission tools. The job verifies the actual
one-node/one-task/one-GPU allocation before opening the archive or model. The
runner publishes a preliminary online W&B run marked pending-postflight. Strict
postflight resumes that exact run and publishes terminal pass evidence only
after the complete preliminary artifact set validates; it then verifies the
expanded terminal artifact set again.

```bash
export JEPA4D_TUM_ARCHIVE="$PWD/checkpoints/datasets/rgbd_dataset_freiburg1_xyz.tgz"
export JEPA4D_VGGT_CHECKPOINT="$PWD/checkpoints/phase2b_assets/VGGT-1B"
export JEPA4D_WANDB_ENTITY="your-approved-entity"
export JEPA4D_WANDB_PROJECT=jepa4d-worldmodel
bash slurm/submit_geometry_official_mini.sh
```

Authentication must already be available through the submitter's home (for
example, a mode-0600 netrc); never export or embed a W&B key in a script. The
wrapper deliberately accepts no extracted-dataset root. Success requires the
terminal content-addressed receipt, not merely a zero exit from inference or a
preliminary W&B upload. Keep the output directory immutable and use its
execution, postflight, and terminal receipts when updating the experiment
record.

## Synthetic Phase 2g training-instrumentation smoke

This is an implementation-only, synthetic CUDA smoke for the M0-M3 training
and W&B instrumentation path. It consumes no dataset, model, checkpoint, or
archive input and is not roadmap first-round training, held-out evidence, or a
promotion result. The current roadmap does not authorize submitting it as a
Phase 2g experiment; the wrapper is provided for a separately authorized
engineering check after its complete implementation is committed.

The wrapper is the only supported Slurm entrypoint. It requires a clean
committed tree, a fresh output, one node/task/GPU for no more than 30 minutes,
and fewer than eight expanded active tasks before submission. It runs all four
instrumentation modes in one W&B run with one to ten optimizer steps per mode
(three by default). Scheduler stdout/stderr and the structured job directory
are printed as machine-readable `key=value` lines.

W&B authentication must already exist at `$HOME/.netrc`, owned by the user and
set to mode 0600. The credential is read by W&B from `HOME`; the wrapper never
accepts, logs, or exports an API key or token. Once separately authorized, the
non-secret invocation is:

```bash
chmod 600 "$HOME/.netrc"
export JEPA4D_WANDB_ENTITY="your-approved-entity"
export JEPA4D_WANDB_PROJECT=jepa4d-worldmodel
export JEPA4D_MAX_STEPS=3
bash slurm/submit_phase2g_training_smoke.sh
```

Success requires `training_receipt.json`, `steps.jsonl`, all four M0-M3
checkpoints, `wandb_receipt.json`, and `SUCCESS` in the fresh output, plus a
separate `SUCCESS` marker in the structured Slurm log directory. GPU telemetry
is sampled every second. These artifacts prove only that the synthetic
instrumentation path executed; they do not authorize or stand in for any
real-data training.
