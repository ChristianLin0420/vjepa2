#!/usr/bin/env bash
#SBATCH --job-name=jepa4d-p2d-lat-aggregate
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=polar4,polar3,polar,batch_block1,grizzly,batch_block2,batch_block3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --open-mode=append

set -Eeuo pipefail
IFS=$'\n\t'
umask 027

bootstrap="${JEPA4D_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ ! -f "$bootstrap/slurm/lib.sh" && -f "$bootstrap/../slurm/lib.sh" ]]; then
  bootstrap="$bootstrap/.."
fi
[[ -f "$bootstrap/slurm/lib.sh" ]] || exit 2
# shellcheck disable=SC1091
source "$bootstrap/slurm/lib.sh"
REPO_ROOT="$(jepa4d_find_repo_root "$bootstrap")"
[[ -n "${SLURM_JOB_ID:-}" ]] || jepa4d_die "latency aggregation must run through sbatch"
jepa4d_activate_python "$REPO_ROOT"
jepa4d_init_log_dir "$REPO_ROOT" "phase2d-latency-aggregate"
jepa4d_install_traps

INPUT_ROOT="${JEPA4D_INPUT_ROOT:-$REPO_ROOT/outputs/jepa4d_phase2d/latency}"
OUTPUT="${JEPA4D_OUTPUT:-$REPO_ROOT/outputs/jepa4d_phase2d/latency-aggregate}"
EXPECTED=12
TEST_REPORT="${JEPA4D_TEST_REPORT:-$REPO_ROOT/outputs/phase2d-gates/tests.json}"
jepa4d_require_file "$TEST_REPORT" "passing Phase 2d test receipt"
if [[ -d "$OUTPUT" && -n "$(find "$OUTPUT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  jepa4d_die "aggregate output is not empty: $OUTPUT"
fi
mkdir -p "$OUTPUT"
export WANDB_MODE=online
export WANDB_DIR="${WANDB_DIR:-$JEPA4D_JOB_LOG_DIR/wandb}"
mkdir -p "$WANDB_DIR"
jepa4d_capture_environment "$REPO_ROOT" "$JEPA4D_JOB_LOG_DIR"
jepa4d_run_step "$JEPA4D_PYTHON" "$REPO_ROOT/slurm/validate_phase2d_test_receipt.py" \
  --repo-root "$REPO_ROOT" --receipt "$TEST_REPORT"

command=(
  "$JEPA4D_PYTHON" "$REPO_ROOT/scripts/aggregate_phase2d_latency.py"
  --input-root "$INPUT_ROOT"
  --output "$OUTPUT"
  --expected-replicates "$EXPECTED"
  --wandb-project "${JEPA4D_WANDB_PROJECT:-jepa4d-worldmodel}"
  --wandb
)
if [[ -n "${JEPA4D_WANDB_ENTITY:-}" ]]; then
  command+=(--wandb-entity "$JEPA4D_WANDB_ENTITY")
fi
cd "$REPO_ROOT"
jepa4d_run_step "${command[@]}"
jepa4d_require_file "$OUTPUT/latency_aggregate.json" "aggregate latency JSON"
jepa4d_require_file "$OUTPUT/latency_aggregate_report.html" "aggregate latency report"
jepa4d_require_file "$OUTPUT/wandb_receipt.json" "aggregate latency W&B receipt"
jepa4d_run_step "$JEPA4D_PYTHON" "$REPO_ROOT/slurm/validate_phase2d_diagnostics.py" \
  --latency-root "$INPUT_ROOT" --latency-output "$OUTPUT" \
  --receipt-output "$OUTPUT/postflight.json"
jepa4d_require_file "$OUTPUT/postflight.json" "strict aggregate latency postflight receipt"
printf 'pass\n' >"$JEPA4D_JOB_LOG_DIR/SUCCESS"
