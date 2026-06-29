#!/usr/bin/env bash
#SBATCH --job-name=jepa4d-p2d-latency
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=polar4,polar3,polar,batch_block1,grizzly,batch_block2,batch_block3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:00:00
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
[[ -n "${SLURM_JOB_ID:-}" ]] || jepa4d_die "Phase 2d profiling must run through sbatch"
jepa4d_activate_python "$REPO_ROOT"
jepa4d_init_log_dir "$REPO_ROOT" "phase2d-latency"
jepa4d_install_traps
jepa4d_asset_paths "$REPO_ROOT"
jepa4d_validate_assets

REPLICATE="${JEPA4D_REPLICATE:?export JEPA4D_REPLICATE before submitting}"
[[ "$REPLICATE" =~ ^[0-9]+$ ]] || jepa4d_die "JEPA4D_REPLICATE must be a non-negative integer"
PHASE2C_OUTPUT="${JEPA4D_PHASE2C_OUTPUT:?export JEPA4D_PHASE2C_OUTPUT before submitting}"
DATA_PARENT="${JEPA4D_DATASET_PARENT:?export JEPA4D_DATASET_PARENT before submitting}"
MANIFEST="${JEPA4D_MANIFEST:-$REPO_ROOT/jepa4d/config/benchmarks/manifests/tum_rgbd_phase2c_cross_sequence_v1.yaml}"
OUTPUT_ROOT="${JEPA4D_OUTPUT_ROOT:-$REPO_ROOT/outputs/jepa4d_phase2d/latency}"
OUTPUT="$OUTPUT_ROOT/replicate-$(printf '%02d' "$REPLICATE")"
TEST_REPORT="${JEPA4D_TEST_REPORT:-$REPO_ROOT/outputs/phase2d-gates/tests.json}"

jepa4d_require_dir "$PHASE2C_OUTPUT" "Phase 2c output"
jepa4d_require_dir "$DATA_PARENT" "dataset parent"
jepa4d_require_file "$MANIFEST" "Phase 2c manifest"
jepa4d_require_file "$TEST_REPORT" "passing Phase 2d test receipt"
if [[ -d "$OUTPUT" && -n "$(find "$OUTPUT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  jepa4d_die "latency output is not empty: $OUTPUT"
fi
mkdir -p "$OUTPUT"

export WANDB_MODE=online
export WANDB_DIR="${WANDB_DIR:-$JEPA4D_JOB_LOG_DIR/wandb}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-180}"
mkdir -p "$WANDB_DIR"
jepa4d_capture_environment "$REPO_ROOT" "$JEPA4D_JOB_LOG_DIR"
jepa4d_start_gpu_monitor "$JEPA4D_JOB_LOG_DIR/gpu-telemetry.csv"

jepa4d_run_step "$JEPA4D_PYTHON" "$REPO_ROOT/slurm/validate_phase2d_test_receipt.py" \
  --repo-root "$REPO_ROOT" --receipt "$TEST_REPORT"

printf '\n[%s] allocated GPU health check\n' "$(date --iso-8601=seconds)"
jepa4d_run_step "$JEPA4D_PYTHON" "$REPO_ROOT/scripts/check_cuda.py" \
  --device "${JEPA4D_CUDA_DEVICE_INDEX:-0}" \
  --stress-seconds "${JEPA4D_CUDA_STRESS_SECONDS:-15}" \
  --matrix-size "${JEPA4D_CUDA_MATRIX_SIZE:-4096}" \
  --allocation-mib "${JEPA4D_CUDA_ALLOCATION_MIB:-1024}" \
  --json-output "$JEPA4D_JOB_LOG_DIR/cuda-health.json"

command=(
  "$JEPA4D_PYTHON" "$REPO_ROOT/scripts/run_phase2d_latency_confirmation.py"
  --phase2c-output "$PHASE2C_OUTPUT"
  --dataset-parent "$DATA_PARENT"
  --manifest "$MANIFEST"
  --vjepa-checkpoint "$JEPA4D_VJEPA_CHECKPOINT"
  --vjepa-implementation "$JEPA4D_VJEPA_IMPLEMENTATION"
  --output "$OUTPUT"
  --test-receipt "$TEST_REPORT"
  --replicate "$REPLICATE"
  --warmups "${JEPA4D_LATENCY_WARMUPS:-30}"
  --blocks "${JEPA4D_LATENCY_BLOCKS:-30}"
  --iterations "${JEPA4D_LATENCY_ITERATIONS:-100}"
  --device "cuda:${JEPA4D_CUDA_DEVICE_INDEX:-0}"
  --wandb-project "${JEPA4D_WANDB_PROJECT:-jepa4d-worldmodel}"
  --wandb
)
if [[ -n "${JEPA4D_WANDB_ENTITY:-}" ]]; then
  command+=(--wandb-entity "$JEPA4D_WANDB_ENTITY")
fi

printf '\n[%s] Phase 2d randomized/interleaved latency replicate %s\n' \
  "$(date --iso-8601=seconds)" "$REPLICATE"
cd "$REPO_ROOT"
jepa4d_run_step "${command[@]}"

jepa4d_require_file "$OUTPUT/latency.json" "latency JSON"
jepa4d_require_file "$OUTPUT/latency_rows.csv" "latency raw rows"
jepa4d_require_file "$OUTPUT/latency_report.html" "latency HTML report"
jepa4d_require_file "$OUTPUT/source_identity.json" "Phase 2c source identity"
jepa4d_require_file "$OUTPUT/gpu_telemetry.csv" "latency GPU telemetry snapshot"
jepa4d_require_file "$OUTPUT/wandb_receipt.json" "latency W&B receipt"
printf 'pass\n' >"$JEPA4D_JOB_LOG_DIR/SUCCESS"
printf '[%s] Phase 2d latency replicate completed: %s\n' "$(date --iso-8601=seconds)" "$OUTPUT"
