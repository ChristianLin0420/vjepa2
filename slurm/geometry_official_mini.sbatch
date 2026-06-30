#!/usr/bin/env bash
#SBATCH --job-name=j4d-gmini-submit-required
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=polar4,polar3,polar,batch_block1,grizzly,batch_block2,batch_block3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@120

set -Eeuo pipefail
IFS=$'\n\t'
umask 027

bootstrap="${JEPA4D_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ ! -f "$bootstrap/slurm/lib.sh" && -f "$bootstrap/../slurm/lib.sh" ]]; then
  bootstrap="$bootstrap/.."
fi
[[ -f "$bootstrap/slurm/lib.sh" ]] || {
  printf 'ERROR: submit from the repository root or export JEPA4D_REPO_ROOT\n' >&2
  exit 2
}
# shellcheck disable=SC1091
source "$bootstrap/slurm/lib.sh"

REPO_ROOT="$(jepa4d_find_repo_root "$bootstrap")"
[[ -n "${SLURM_JOB_ID:-}" ]] || jepa4d_die "the governed TUM official-mini evaluation must run through sbatch"
[[ "${SLURM_JOB_NAME:-}" == j4d-gmini-* ]] || jepa4d_die "governed mini job name lacks its required prefix"
jepa4d_activate_python "$REPO_ROOT"
jepa4d_init_log_dir "$REPO_ROOT" "geometry-official-mini"
jepa4d_install_traps

PYTHON="$REPO_ROOT/.conda-gpu/bin/python"
[[ -x "$PYTHON" ]] || jepa4d_die "the governed evaluation requires $PYTHON"
ARCHIVE="${JEPA4D_TUM_ARCHIVE:?export JEPA4D_TUM_ARCHIVE before submitting}"
MODEL_ID="${JEPA4D_VGGT_CHECKPOINT:?export JEPA4D_VGGT_CHECKPOINT before submitting}"
STAGE_OUTPUT="${JEPA4D_STAGE_OUTPUT:?submit through slurm/submit_geometry_official_mini.sh}"
EXECUTION_ID="${JEPA4D_EXECUTION_ID:?submit through slurm/submit_geometry_official_mini.sh}"
RUN_NAME="${JEPA4D_RUN_NAME:?submit through slurm/submit_geometry_official_mini.sh}"
SUBMITTED_COMMIT="${JEPA4D_GIT_COMMIT:?submit through slurm/submit_geometry_official_mini.sh}"
VALIDATION_STATE_ROOT="${JEPA4D_VALIDATION_STATE_ROOT:?submit through slurm/submit_geometry_official_mini.sh}"
REGISTRY="$REPO_ROOT/configs/validation/dataset_registry.yaml"
LEDGER="$REPO_ROOT/configs/validation/consumed_test_ledger.yaml"
WANDB_PROJECT="${JEPA4D_WANDB_PROJECT:-jepa4d-worldmodel}"
WANDB_ENTITY="${JEPA4D_WANDB_ENTITY:-}"

printf '\n[%s] allocated Slurm policy preflight\n' "$(date --iso-8601=seconds)"
jepa4d_run_step "$PYTHON" "$REPO_ROOT/slurm/validate_geometry_official_mini.py" \
  --job-id "$SLURM_JOB_ID" \
  --allocation-only

jepa4d_require_dir "$MODEL_ID" "VGGT checkpoint"
jepa4d_require_file "$REGISTRY" "validation dataset registry"
jepa4d_require_file "$LEDGER" "consumed-test ledger"
[[ "$(git -C "$REPO_ROOT" rev-parse --verify 'HEAD^{commit}')" == "$SUBMITTED_COMMIT" ]] || {
  jepa4d_die "repository HEAD changed after submission"
}
[[ -z "$(git -C "$REPO_ROOT" status --porcelain=v1 --untracked-files=all)" ]] || {
  jepa4d_die "repository became dirty after submission"
}
[[ ! -e "$STAGE_OUTPUT" ]] || jepa4d_die "fresh output path already exists: $STAGE_OUTPUT"
mkdir -p "$(dirname "$STAGE_OUTPUT")" "$VALIDATION_STATE_ROOT"
mkdir "$STAGE_OUTPUT"

[[ "${WANDB_MODE:-online}" == "online" ]] || jepa4d_die "governed validation requires WANDB_MODE=online"
export WANDB_MODE=online
export PYTHONUTF8=1
export JEPA4D_VALIDATION_STATE_ROOT

jepa4d_capture_environment "$REPO_ROOT" "$JEPA4D_JOB_LOG_DIR"
jepa4d_start_gpu_monitor "$JEPA4D_JOB_LOG_DIR/gpu-telemetry.csv"

command=(
  "$PYTHON" -m jepa4d.validation.geometry_official_mini
  --registry "$REGISTRY"
  --ledger "$LEDGER"
  --archive "$ARCHIVE"
  --output "$STAGE_OUTPUT"
  --model-id "$MODEL_ID"
  --device cuda:0
  --execution-id "$EXECUTION_ID"
  --run-name "$RUN_NAME"
  --wandb-project "$WANDB_PROJECT"
)
if [[ -n "$WANDB_ENTITY" ]]; then
  command+=(--wandb-entity "$WANDB_ENTITY")
fi

cd "$REPO_ROOT"
printf '\n[%s] governed TUM official-mini regression\n' "$(date --iso-8601=seconds)"
jepa4d_run_step "${command[@]}"

printf '\n[%s] content-addressed dashboard and postflight validation\n' "$(date --iso-8601=seconds)"
jepa4d_run_step "$PYTHON" "$REPO_ROOT/slurm/validate_geometry_official_mini.py" \
  --output "$STAGE_OUTPUT" \
  --repo-root "$REPO_ROOT" \
  --job-id "$SLURM_JOB_ID"
printf 'pass\n' >"$JEPA4D_JOB_LOG_DIR/SUCCESS"
