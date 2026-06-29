#!/usr/bin/env bash
#SBATCH --job-name=jepa4d-p2b-train
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=polar4,polar3,polar,batch_block1,grizzly,batch_block2,batch_block3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=220G
#SBATCH --time=04:00:00
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
[[ -n "${SLURM_JOB_ID:-}" ]] || jepa4d_die "formal Phase 2b training must run through sbatch"
jepa4d_activate_python "$REPO_ROOT"
jepa4d_init_log_dir "$REPO_ROOT" "phase2b-train"
jepa4d_install_traps
jepa4d_asset_paths "$REPO_ROOT"
jepa4d_validate_assets

DATASET_ROOT="${JEPA4D_DATASET_ROOT:?export JEPA4D_DATASET_ROOT before submitting}"
TUM_ARCHIVE="${JEPA4D_TUM_ARCHIVE:?export JEPA4D_TUM_ARCHIVE before submitting}"
MANIFEST="${JEPA4D_MANIFEST:-$REPO_ROOT/jepa4d/config/benchmarks/manifests/tum_rgbd_phase2b_v1.yaml}"
jepa4d_require_dir "$DATASET_ROOT" "TUM dataset root"
jepa4d_require_file "$TUM_ARCHIVE" "TUM archive"
jepa4d_require_file "$MANIFEST" "Phase 2b manifest"
DATASET_ROOT="$(jepa4d_realpath "$DATASET_ROOT")"
TUM_ARCHIVE="$(jepa4d_realpath "$TUM_ARCHIVE")"
MANIFEST="$(jepa4d_realpath "$MANIFEST")"

OUTPUT_DIR="${JEPA4D_OUTPUT_DIR:-$REPO_ROOT/outputs/jepa4d_phase2b/tum_rgbd_v1}"
if [[ -d "$OUTPUT_DIR" && -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  jepa4d_die "output directory is not empty: $OUTPUT_DIR (choose a new path; overwrite is unsafe)"
elif [[ -e "$OUTPUT_DIR" && ! -d "$OUTPUT_DIR" ]]; then
  jepa4d_die "output path exists and is not a directory: $OUTPUT_DIR"
fi
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(jepa4d_realpath "$OUTPUT_DIR")"

PREFLIGHT_REPORT="${JEPA4D_PREFLIGHT_REPORT:-$REPO_ROOT/outputs/phase2b-gates/preflight.json}"
TEST_REPORT="${JEPA4D_TEST_REPORT:-$REPO_ROOT/outputs/phase2b-gates/tests.json}"
jepa4d_require_file "$PREFLIGHT_REPORT" "successful Phase 2b preflight report"
jepa4d_require_file "$TEST_REPORT" "successful Phase 2b Slurm test receipt"

EPOCHS="${JEPA4D_EPOCHS:-60}"
[[ "$EPOCHS" == "60" ]] || jepa4d_die "formal Phase 2b training is locked to exactly 60 epochs"
RUN_NAME="${JEPA4D_RUN_NAME:-phase2b-jepa-geometry-distillation-v1}"
WANDB_PROJECT="${JEPA4D_WANDB_PROJECT:-jepa4d-worldmodel}"
WANDB_ENTITY="${JEPA4D_WANDB_ENTITY:-}"
[[ "${WANDB_MODE:-online}" == "online" ]] || jepa4d_die "formal Phase 2b requires WANDB_MODE=online"
export WANDB_MODE=online
export WANDB_DIR="${WANDB_DIR:-$JEPA4D_JOB_LOG_DIR/wandb}"
mkdir -p "$WANDB_DIR"

jepa4d_capture_environment "$REPO_ROOT" "$JEPA4D_JOB_LOG_DIR"
jepa4d_start_gpu_monitor "$JEPA4D_JOB_LOG_DIR/gpu-telemetry.csv"

printf '\n[%s] final allocated-node CUDA gate\n' "$(date --iso-8601=seconds)"
jepa4d_run_step "$JEPA4D_PYTHON" "$REPO_ROOT/scripts/check_cuda.py" \
  --device "${JEPA4D_CUDA_DEVICE_INDEX:-0}" \
  --stress-seconds "${JEPA4D_TRAIN_CUDA_STRESS_SECONDS:-15}" \
  --matrix-size "${JEPA4D_CUDA_MATRIX_SIZE:-4096}" \
  --allocation-mib "${JEPA4D_TRAIN_CUDA_ALLOCATION_MIB:-1024}" \
  --json-output "$JEPA4D_JOB_LOG_DIR/cuda-health.json"

printf '\n[%s] bind passing tests/preflight to exact code, environment, data, and models\n' \
  "$(date --iso-8601=seconds)"
jepa4d_run_step "$JEPA4D_PYTHON" "$REPO_ROOT/slurm/validate_phase2b_preflight.py" \
  --preflight-report "$PREFLIGHT_REPORT" \
  --test-report "$TEST_REPORT" \
  --repo-root "$REPO_ROOT" \
  --dataset-root "$DATASET_ROOT" \
  --archive "$TUM_ARCHIVE" \
  --manifest "$MANIFEST" \
  --vjepa-checkpoint "$JEPA4D_VJEPA_CHECKPOINT" \
  --vjepa-implementation "$JEPA4D_VJEPA_IMPLEMENTATION" \
  --vggt-checkpoint "$JEPA4D_VGGT_CHECKPOINT" \
  --output "$JEPA4D_JOB_LOG_DIR/formal-authorization.json"

command=(
  "$JEPA4D_PYTHON" "$REPO_ROOT/scripts/run_phase2b_geometry_distillation.py"
  --dataset-root "$DATASET_ROOT"
  --archive "$TUM_ARCHIVE"
  --manifest "$MANIFEST"
  --output "$OUTPUT_DIR"
  --vjepa-checkpoint "$JEPA4D_VJEPA_CHECKPOINT"
  --vjepa-implementation "$JEPA4D_VJEPA_IMPLEMENTATION"
  --vggt-checkpoint "$JEPA4D_VGGT_CHECKPOINT"
  --device "cuda:${JEPA4D_CUDA_DEVICE_INDEX:-0}"
  --epochs "$EPOCHS"
  --wandb
  --wandb-project "$WANDB_PROJECT"
  --run-name "$RUN_NAME"
)
if [[ -n "$WANDB_ENTITY" ]]; then
  command+=(--wandb-entity "$WANDB_ENTITY")
fi

printf '\n[%s] formal Phase 2b training\n' "$(date --iso-8601=seconds)"
printf 'output=%s epochs=%s run_name=%s wandb_project=%s wandb_mode=%s\n' \
  "$OUTPUT_DIR" "$EPOCHS" "$RUN_NAME" "$WANDB_PROJECT" "$WANDB_MODE"
printf 'command:'
printf ' %q' "${command[@]}"
printf '\n'
cd "$REPO_ROOT"
jepa4d_run_step "${command[@]}"

printf '\n[%s] strict output and checkpoint validation\n' "$(date --iso-8601=seconds)"
validation_command=("$JEPA4D_PYTHON" "$REPO_ROOT/slurm/validate_phase2b_output.py" --output "$OUTPUT_DIR")
validation_command+=(--require-wandb --report "$JEPA4D_JOB_LOG_DIR/postflight-validation.json")
jepa4d_run_step "${validation_command[@]}"
printf 'pass\n' >"$JEPA4D_JOB_LOG_DIR/SUCCESS"
printf '[%s] formal training and postflight validation passed; output=%s\n' \
  "$(date --iso-8601=seconds)" "$OUTPUT_DIR"
