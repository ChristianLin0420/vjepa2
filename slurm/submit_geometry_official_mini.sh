#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'
umask 027

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

reject_sensitive_identifier() {
  local name="$1"
  local value="$2"
  local normalized="${value,,}"
  local credential_pattern='wandb_v1_|(^|[._-])hf_[a-z0-9]{16,}|api[._-]?key|access[._-]?token|secret'
  [[ ! "$normalized" =~ $credential_pattern ]] || {
    die "$name resembles credential material and may not be published as an identifier"
  }
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

git rev-parse --is-inside-work-tree >/dev/null 2>&1 || die "submission requires a Git worktree"
COMMIT="$(git rev-parse --verify 'HEAD^{commit}')" || die "submission requires a committed HEAD"
[[ -z "$(git status --porcelain=v1 --untracked-files=all)" ]] || {
  die "governed TUM official-mini submission requires a clean committed worktree"
}
SHORT="$(git rev-parse --short=8 "$COMMIT")"

PYTHON="$ROOT/.conda-gpu/bin/python"
[[ -x "$PYTHON" ]] || die "missing executable governed-evaluation Python: $PYTHON"
ARCHIVE="${JEPA4D_TUM_ARCHIVE:?export JEPA4D_TUM_ARCHIVE before submitting}"
MODEL_ID="${JEPA4D_VGGT_CHECKPOINT:?export JEPA4D_VGGT_CHECKPOINT before submitting}"
[[ -d "$MODEL_ID" ]] || die "VGGT checkpoint is not a directory: $MODEL_ID"
[[ -f "$ROOT/configs/validation/dataset_registry.yaml" ]] || die "validation dataset registry is missing"
[[ -f "$ROOT/configs/validation/consumed_test_ledger.yaml" ]] || die "consumed-test ledger is missing"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
NONCE="$(printf '%04x%04x' "$RANDOM" "$RANDOM")"
DEFAULT_SUFFIX="${SHORT}-${STAMP}-${NONCE}"
EXECUTION_ID="${JEPA4D_EXECUTION_ID:-tum-mini-${DEFAULT_SUFFIX}}"
RUN_NAME="${JEPA4D_RUN_NAME:-tum-official-mini-${DEFAULT_SUFFIX}}"
JOB_NAME="${JEPA4D_JOB_NAME:-j4d-gmini-${DEFAULT_SUFFIX}}"
WANDB_PROJECT="${JEPA4D_WANDB_PROJECT:-jepa4d-worldmodel}"
WANDB_ENTITY="${JEPA4D_WANDB_ENTITY:-}"
name_pattern='^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$'
[[ "$EXECUTION_ID" =~ $name_pattern ]] || die "JEPA4D_EXECUTION_ID must be a path-safe unique name"
[[ "$RUN_NAME" =~ $name_pattern ]] || die "JEPA4D_RUN_NAME must be a path-safe unique name"
[[ "$JOB_NAME" =~ $name_pattern ]] || die "JEPA4D_JOB_NAME must be a path-safe unique name"
[[ "$WANDB_PROJECT" =~ $name_pattern ]] || die "JEPA4D_WANDB_PROJECT must be a path-safe name"
[[ -z "$WANDB_ENTITY" || "$WANDB_ENTITY" =~ $name_pattern ]] || die "JEPA4D_WANDB_ENTITY must be a path-safe name"
[[ "$JOB_NAME" == j4d-gmini-* ]] || die "JEPA4D_JOB_NAME must begin with j4d-gmini-"
[[ "$EXECUTION_ID" != "$RUN_NAME" && "$EXECUTION_ID" != "$JOB_NAME" && "$RUN_NAME" != "$JOB_NAME" ]] || {
  die "execution, W&B run, and Slurm job names must be distinct"
}
reject_sensitive_identifier JEPA4D_EXECUTION_ID "$EXECUTION_ID"
reject_sensitive_identifier JEPA4D_RUN_NAME "$RUN_NAME"
reject_sensitive_identifier JEPA4D_JOB_NAME "$JOB_NAME"
reject_sensitive_identifier JEPA4D_WANDB_PROJECT "$WANDB_PROJECT"
[[ -z "$WANDB_ENTITY" ]] || reject_sensitive_identifier JEPA4D_WANDB_ENTITY "$WANDB_ENTITY"

STAGE_OUTPUT="${JEPA4D_STAGE_OUTPUT:-$ROOT/outputs/geometry-official-mini/$EXECUTION_ID}"
VALIDATION_STATE_ROOT="${JEPA4D_VALIDATION_STATE_ROOT:-$ROOT/outputs/validation-state}"
[[ "$STAGE_OUTPUT" == /* ]] || STAGE_OUTPUT="$ROOT/$STAGE_OUTPUT"
[[ "$VALIDATION_STATE_ROOT" == /* ]] || VALIDATION_STATE_ROOT="$ROOT/$VALIDATION_STATE_ROOT"
[[ ! -e "$STAGE_OUTPUT" ]] || die "fresh output path already exists: $STAGE_OUTPUT"

command -v squeue >/dev/null 2>&1 || die "squeue is required for the allocation guard"
command -v sbatch >/dev/null 2>&1 || die "sbatch is required for submission"
command -v flock >/dev/null 2>&1 || die "flock is required for the atomic submission guard"
SCHEDULER_USER="$(id -un)" || die "unable to determine the authenticated scheduler user"
[[ -n "$SCHEDULER_USER" ]] || die "authenticated scheduler user is empty"
JOB_HOME="${HOME:?HOME is required for the shared submission lock and W&B authentication}"
LOCK_ROOT="$JOB_HOME/.cache/jepa4d"
mkdir -p "$LOCK_ROOT"
exec 9>"$LOCK_ROOT/slurm-submit.lock"
flock -x 9 || die "unable to acquire the atomic Slurm submission guard"
if ! active_raw="$(squeue -r -h -u "$SCHEDULER_USER" -o "%i")"; then
  die "unable to query active jobs; refusing submission"
fi
mapfile -t active_job_tasks < <(printf '%s\n' "$active_raw" | awk 'NF {print $1}' | sort -u)
if (( ${#active_job_tasks[@]} >= 8 )); then
  die "found ${#active_job_tasks[@]} distinct active job tasks (limit is 8); refusing submission"
fi

if ! queued_names="$(squeue -h -u "$SCHEDULER_USER" -o "%j")"; then
  die "unable to query existing Slurm job names; refusing submission"
fi
if printf '%s\n' "$queued_names" | awk -v candidate="$JOB_NAME" '$0 == candidate { found=1 } END { exit !found }'; then
  die "Slurm job name is already present in the queue: $JOB_NAME"
fi

JOB_PATH="${PATH:?PATH is required for the allocation environment}"
for exported_value in \
  "$ROOT" "$ARCHIVE" "$MODEL_ID" "$STAGE_OUTPUT" "$EXECUTION_ID" \
  "$RUN_NAME" "$COMMIT" "$VALIDATION_STATE_ROOT" "$WANDB_PROJECT" "$WANDB_ENTITY" "$JOB_HOME" "$JOB_PATH"; do
  [[ "$exported_value" != *','* && "$exported_value" != *$'\n'* ]] || {
    die "Slurm export values may not contain commas or newlines"
  }
done

exports=(
  "HOME=$JOB_HOME"
  "PATH=$JOB_PATH"
  "JEPA4D_REPO_ROOT=$ROOT"
  "JEPA4D_TUM_ARCHIVE=$ARCHIVE"
  "JEPA4D_VGGT_CHECKPOINT=$MODEL_ID"
  "JEPA4D_STAGE_OUTPUT=$STAGE_OUTPUT"
  "JEPA4D_EXECUTION_ID=$EXECUTION_ID"
  "JEPA4D_RUN_NAME=$RUN_NAME"
  "JEPA4D_GIT_COMMIT=$COMMIT"
  "JEPA4D_VALIDATION_STATE_ROOT=$VALIDATION_STATE_ROOT"
  "JEPA4D_WANDB_PROJECT=$WANDB_PROJECT"
  "WANDB_MODE=online"
  "PYTHONUTF8=1"
)
if [[ -n "$WANDB_ENTITY" ]]; then
  exports+=("JEPA4D_WANDB_ENTITY=$WANDB_ENTITY")
fi
export_list="$(IFS=,; printf '%s' "${exports[*]}")"
SUBMISSION_LOG_ROOT="$ROOT/outputs/slurm-submit-logs"
mkdir -p "$SUBMISSION_LOG_ROOT"

result="$(
  sbatch --parsable \
    --job-name "$JOB_NAME" \
    --output "$SUBMISSION_LOG_ROOT/%x-%j.out" \
    --error "$SUBMISSION_LOG_ROOT/%x-%j.err" \
    --export "$export_list" \
    "$ROOT/slurm/geometry_official_mini.sbatch"
)"
job_id="${result%%;*}"
[[ "$job_id" =~ ^[0-9]+$ ]] || die "sbatch returned an invalid job ID: $result"

printf 'Governed TUM official-mini job: %s\n' "$job_id"
printf 'Execution: %s\n' "$EXECUTION_ID"
printf 'Run name: %s\n' "$RUN_NAME"
printf 'Output: %s\n' "$STAGE_OUTPUT"
printf 'Commit: %s\n' "$COMMIT"
printf 'Active job tasks before submission: %s\n' "${#active_job_tasks[@]}"
