#!/usr/bin/env bash
#SBATCH --job-name=jepa4d-p2d-attribution
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
if [[ ! -f "$bootstrap/slurm/lib.sh" && -f "$bootstrap/../slurm/lib.sh" ]]; then bootstrap="$bootstrap/.."; fi
[[ -f "$bootstrap/slurm/lib.sh" ]] || exit 2
# shellcheck disable=SC1091
source "$bootstrap/slurm/lib.sh"
REPO_ROOT="$(jepa4d_find_repo_root "$bootstrap")"
[[ -n "${SLURM_JOB_ID:-}" ]] || jepa4d_die "Phase 2d attribution must run through sbatch"
jepa4d_activate_python "$REPO_ROOT"
jepa4d_init_log_dir "$REPO_ROOT" "phase2d-attribution"
jepa4d_install_traps
jepa4d_asset_paths "$REPO_ROOT"
jepa4d_validate_assets

SOURCE="${JEPA4D_PHASE2C_OUTPUT:?export JEPA4D_PHASE2C_OUTPUT before submitting}"
DATA_PARENT="${JEPA4D_DATASET_PARENT:?export JEPA4D_DATASET_PARENT before submitting}"
MANIFEST="${JEPA4D_MANIFEST:-$REPO_ROOT/jepa4d/config/benchmarks/manifests/tum_rgbd_phase2c_cross_sequence_v1.yaml}"
OUTPUT="${JEPA4D_OUTPUT:-$REPO_ROOT/outputs/jepa4d_phase2d/fusion-attribution}"
TEST_REPORT="${JEPA4D_TEST_REPORT:-$REPO_ROOT/outputs/phase2d-gates/tests.json}"
jepa4d_require_dir "$SOURCE" "formal Phase 2c output"
jepa4d_require_dir "$DATA_PARENT" "TUM dataset parent"
jepa4d_require_file "$MANIFEST" "Phase 2c manifest"
jepa4d_require_file "$TEST_REPORT" "passing Phase 2d test receipt"
if [[ -d "$OUTPUT" && -n "$(find "$OUTPUT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  jepa4d_die "attribution output is not empty: $OUTPUT"
fi
mkdir -p "$OUTPUT"
export WANDB_MODE=online
export WANDB_DIR="${WANDB_DIR:-$JEPA4D_JOB_LOG_DIR/wandb}"
mkdir -p "$WANDB_DIR"
jepa4d_capture_environment "$REPO_ROOT" "$JEPA4D_JOB_LOG_DIR"
jepa4d_start_gpu_monitor "$JEPA4D_JOB_LOG_DIR/gpu-telemetry.csv"

jepa4d_run_step "$JEPA4D_PYTHON" "$REPO_ROOT/slurm/validate_phase2d_test_receipt.py" \
  --repo-root "$REPO_ROOT" --receipt "$TEST_REPORT"

jepa4d_run_step "$JEPA4D_PYTHON" "$REPO_ROOT/scripts/check_cuda.py" \
  --device "${JEPA4D_CUDA_DEVICE_INDEX:-0}" --stress-seconds 15 --matrix-size 4096 --allocation-mib 1024 \
  --json-output "$JEPA4D_JOB_LOG_DIR/cuda-health.json"

cd "$REPO_ROOT"
jepa4d_run_step "$JEPA4D_PYTHON" "$REPO_ROOT/scripts/run_phase2d_fusion_attribution.py" \
  --phase2c-output "$SOURCE" --dataset-parent "$DATA_PARENT" --manifest "$MANIFEST" \
  --vjepa-checkpoint "$JEPA4D_VJEPA_CHECKPOINT" --vjepa-implementation "$JEPA4D_VJEPA_IMPLEMENTATION" \
  --device "cuda:${JEPA4D_CUDA_DEVICE_INDEX:-0}" --chunk-size 8 --probe-batch-size 8 --output "$OUTPUT" \
  --test-receipt "$TEST_REPORT"

log_command=(
  "$JEPA4D_PYTHON" "$REPO_ROOT/scripts/log_phase2d_diagnostics.py"
  --kind attribution --input "$OUTPUT" --run-name "phase2d-fusion-attribution-${SLURM_JOB_ID}"
  --project "${JEPA4D_WANDB_PROJECT:-jepa4d-worldmodel}"
)
if [[ -n "${JEPA4D_WANDB_ENTITY:-}" ]]; then log_command+=(--entity "$JEPA4D_WANDB_ENTITY"); fi
jepa4d_run_step "${log_command[@]}"

for required in fusion_attribution.json fusion_attribution_report.html full_predictions.npz qualitative_examples.npz receipt.json \
  source_identity.json wandb_receipt.json; do
  jepa4d_require_file "$OUTPUT/$required" "$required"
done
if grep -Eiq '<script[^>]+src=' "$OUTPUT/fusion_attribution_report.html"; then
  jepa4d_die "attribution HTML contains an external script dependency"
fi
printf 'pass\n' >"$JEPA4D_JOB_LOG_DIR/SUCCESS"
