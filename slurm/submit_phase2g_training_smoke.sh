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

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"

git rev-parse --is-inside-work-tree >/dev/null 2>&1 || die "submission requires a Git worktree"
COMMIT="$(git rev-parse --verify 'HEAD^{commit}')" || die "submission requires a committed HEAD"
[[ -z "$(git status --porcelain=v1 --untracked-files=all)" ]] || {
  die "synthetic training-smoke submission requires a clean committed worktree"
}
SHORT="$(git rev-parse --short=8 "$COMMIT")"

PYTHON="$ROOT/.conda-gpu/bin/python"
[[ -x "$PYTHON" ]] || die "missing executable training-smoke Python: $PYTHON"
[[ -f "$ROOT/scripts/run_phase2g_training_smoke.py" ]] || die "synthetic training-smoke runner is missing"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
NONCE="$(printf '%04x%04x' "$RANDOM" "$RANDOM")"
DEFAULT_SUFFIX="${SHORT}-${STAMP}-${NONCE}"
EXECUTION_ID="${JEPA4D_EXECUTION_ID:-p2g-smoke-exec-${DEFAULT_SUFFIX}}"
RUN_NAME="${JEPA4D_RUN_NAME:-p2g-smoke-run-${DEFAULT_SUFFIX}}"
JOB_NAME="${JEPA4D_JOB_NAME:-j4d-p2g-smoke-${DEFAULT_SUFFIX}}"
MAX_STEPS="${JEPA4D_MAX_STEPS:-3}"
WANDB_PROJECT="${JEPA4D_WANDB_PROJECT:-jepa4d-worldmodel}"
WANDB_ENTITY="${JEPA4D_WANDB_ENTITY:-}"
name_pattern='^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$'
[[ "$EXECUTION_ID" =~ $name_pattern ]] || die "JEPA4D_EXECUTION_ID must be a path-safe unique name"
[[ "$RUN_NAME" =~ $name_pattern ]] || die "JEPA4D_RUN_NAME must be a path-safe unique name"
[[ "$JOB_NAME" =~ $name_pattern ]] || die "JEPA4D_JOB_NAME must be a path-safe unique name"
[[ "$WANDB_PROJECT" =~ $name_pattern ]] || die "JEPA4D_WANDB_PROJECT must be a path-safe name"
[[ -z "$WANDB_ENTITY" || "$WANDB_ENTITY" =~ $name_pattern ]] || die "JEPA4D_WANDB_ENTITY must be a path-safe name"
[[ "$JOB_NAME" == j4d-p2g-smoke-* ]] || die "JEPA4D_JOB_NAME must begin with j4d-p2g-smoke-"
[[ "$MAX_STEPS" =~ ^([1-9]|10)$ ]] || die "JEPA4D_MAX_STEPS must be an integer from 1 through 10"
[[ "$EXECUTION_ID" != "$RUN_NAME" && "$EXECUTION_ID" != "$JOB_NAME" && "$RUN_NAME" != "$JOB_NAME" ]] || {
  die "execution, W&B run, and Slurm job names must be distinct"
}
reject_sensitive_identifier JEPA4D_EXECUTION_ID "$EXECUTION_ID"
reject_sensitive_identifier JEPA4D_RUN_NAME "$RUN_NAME"
reject_sensitive_identifier JEPA4D_JOB_NAME "$JOB_NAME"
reject_sensitive_identifier JEPA4D_WANDB_PROJECT "$WANDB_PROJECT"
[[ -z "$WANDB_ENTITY" ]] || reject_sensitive_identifier JEPA4D_WANDB_ENTITY "$WANDB_ENTITY"

STAGE_OUTPUT="${JEPA4D_STAGE_OUTPUT:-$ROOT/outputs/phase2g-training-smoke/$EXECUTION_ID}"
[[ "$STAGE_OUTPUT" == /* ]] || STAGE_OUTPUT="$ROOT/$STAGE_OUTPUT"
STAGE_OUTPUT="$(realpath -m "$STAGE_OUTPUT")"
[[ "$STAGE_OUTPUT" != / && "$STAGE_OUTPUT" != "$ROOT" ]] || die "unsafe output path: $STAGE_OUTPUT"
[[ ! -e "$STAGE_OUTPUT" && ! -L "$STAGE_OUTPUT" ]] || die "fresh output path already exists: $STAGE_OUTPUT"

JOB_HOME="${HOME:?HOME is required for shared W&B authentication}"
NETRC="$JOB_HOME/.netrc"
[[ -f "$NETRC" && ! -L "$NETRC" ]] || die "W&B authentication requires a regular HOME/.netrc"
[[ "$(stat -Lc '%a' "$NETRC")" == "600" ]] || die "HOME/.netrc must have mode 0600"
[[ "$(stat -Lc '%u' "$NETRC")" == "$(id -u)" ]] || die "HOME/.netrc must be owned by the submitting user"

command -v squeue >/dev/null 2>&1 || die "squeue is required for the allocation guard"
command -v sbatch >/dev/null 2>&1 || die "sbatch is required for submission"
command -v flock >/dev/null 2>&1 || die "flock is required for the atomic submission guard"
SCHEDULER_USER="$(id -un)" || die "unable to determine the authenticated scheduler user"
[[ -n "$SCHEDULER_USER" ]] || die "authenticated scheduler user is empty"
LOCK_ROOT="$JOB_HOME/.cache/jepa4d"
mkdir -p "$LOCK_ROOT"
exec 9>"$LOCK_ROOT/slurm-submit.lock"
flock -x 9 || die "unable to acquire the atomic Slurm submission guard"
if ! active_raw="$(squeue -r -h -u "$SCHEDULER_USER" -o "%i")"; then
  die "unable to query active jobs; refusing submission"
fi
mapfile -t active_job_tasks < <(printf '%s\n' "$active_raw" | awk 'NF {print $1}' | sort -u)
if (( ${#active_job_tasks[@]} >= 8 )); then
  die "found ${#active_job_tasks[@]} distinct active job tasks (limit is less than 8 before submission); refusing submission"
fi

if ! queued_names="$(squeue -h -u "$SCHEDULER_USER" -o "%j")"; then
  die "unable to query existing Slurm job names; refusing submission"
fi
if printf '%s\n' "$queued_names" | awk -v candidate="$JOB_NAME" '$0 == candidate { found=1 } END { exit !found }'; then
  die "Slurm job name is already present in the queue: $JOB_NAME"
fi

JOB_PATH="${PATH:?PATH is required for the allocation environment}"
for exported_value in \
  "$ROOT" "$STAGE_OUTPUT" "$EXECUTION_ID" "$RUN_NAME" "$JOB_NAME" "$COMMIT" \
  "$MAX_STEPS" "$WANDB_PROJECT" "$WANDB_ENTITY" "$JOB_HOME" "$JOB_PATH"; do
  [[ "$exported_value" != *','* && "$exported_value" != *$'\n'* ]] || {
    die "Slurm export values may not contain commas or newlines"
  }
done

exports=(
  "HOME=$JOB_HOME"
  "PATH=$JOB_PATH"
  "JEPA4D_REPO_ROOT=$ROOT"
  "JEPA4D_STAGE_OUTPUT=$STAGE_OUTPUT"
  "JEPA4D_EXECUTION_ID=$EXECUTION_ID"
  "JEPA4D_RUN_NAME=$RUN_NAME"
  "JEPA4D_GIT_COMMIT=$COMMIT"
  "JEPA4D_MAX_STEPS=$MAX_STEPS"
  "JEPA4D_WANDB_PROJECT=$WANDB_PROJECT"
  "JEPA4D_GPU_MONITOR_INTERVAL=1"
  "WANDB_MODE=online"
  "PYTHONUTF8=1"
)
if [[ -n "$WANDB_ENTITY" ]]; then
  exports+=("JEPA4D_WANDB_ENTITY=$WANDB_ENTITY")
fi
export_list="$(IFS=,; printf '%s' "${exports[*]}")"

SUBMISSION_LOG_ROOT="$ROOT/outputs/slurm-submit-logs"
STRUCTURED_LOG_ROOT="$ROOT/outputs/slurm_logs/phase2g-training-smoke"
mkdir -p "$SUBMISSION_LOG_ROOT" "$STRUCTURED_LOG_ROOT"
result="$(
  sbatch --parsable \
    --job-name "$JOB_NAME" \
    --output "$SUBMISSION_LOG_ROOT/%x-%j.out" \
    --error "$SUBMISSION_LOG_ROOT/%x-%j.err" \
    --export "$export_list" \
    "$ROOT/slurm/phase2g_training_smoke.sbatch"
)"
job_id="${result%%;*}"
[[ "$job_id" =~ ^[0-9]+$ ]] || die "sbatch returned an invalid job ID: $result"

printf 'job_id=%s\n' "$job_id"
printf 'output_path=%s\n' "$STAGE_OUTPUT"
printf 'stdout_log=%s/%s-%s.out\n' "$SUBMISSION_LOG_ROOT" "$JOB_NAME" "$job_id"
printf 'stderr_log=%s/%s-%s.err\n' "$SUBMISSION_LOG_ROOT" "$JOB_NAME" "$job_id"
printf 'structured_log_dir=%s/%s-%s\n' "$STRUCTURED_LOG_ROOT" "$JOB_NAME" "$job_id"
printf 'active_job_tasks_before_submission=%s\n' "${#active_job_tasks[@]}"
