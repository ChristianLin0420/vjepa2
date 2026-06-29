#!/usr/bin/env bash
#SBATCH --job-name=jepa4d-p2d-aggregate
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
if [[ ! -f "$bootstrap/slurm/lib.sh" && -f "$bootstrap/../slurm/lib.sh" ]]; then bootstrap="$bootstrap/.."; fi
[[ -f "$bootstrap/slurm/lib.sh" ]] || exit 2
# shellcheck disable=SC1091
source "$bootstrap/slurm/lib.sh"
REPO_ROOT="$(jepa4d_find_repo_root "$bootstrap")"
[[ -n "${SLURM_JOB_ID:-}" ]] || jepa4d_die "Phase 2d aggregation must run through sbatch"
jepa4d_activate_python "$REPO_ROOT"
jepa4d_init_log_dir "$REPO_ROOT" "phase2d-aggregate"
jepa4d_install_traps
TEST_REPORT="${JEPA4D_TEST_REPORT:-$REPO_ROOT/outputs/phase2d-gates/tests.json}"
DEPENDENCY_GRAPH="${JEPA4D_DEPENDENCY_GRAPH:?export JEPA4D_DEPENDENCY_GRAPH before submitting the final aggregate}"
ATTRIBUTION="${JEPA4D_ATTRIBUTION_OUTPUT:-$REPO_ROOT/outputs/jepa4d_phase2d/fusion-attribution}"
CALIBRATION="${JEPA4D_CALIBRATION_OUTPUT:-$REPO_ROOT/outputs/jepa4d_phase2d/calibration-scale-audit}"
LATENCY="${JEPA4D_LATENCY_OUTPUT:-$REPO_ROOT/outputs/jepa4d_phase2d/latency-aggregate}"
LATENCY_ROOT="${JEPA4D_LATENCY_ROOT:-$REPO_ROOT/outputs/jepa4d_phase2d/latency}"
OUTPUT="${JEPA4D_OUTPUT:-$REPO_ROOT/outputs/jepa4d_phase2d/aggregate}"
jepa4d_require_file "$TEST_REPORT" "passing Phase 2d test receipt"
jepa4d_require_file "$DEPENDENCY_GRAPH" "Phase 2d Slurm dependency graph"
jepa4d_require_file "$ATTRIBUTION/fusion_attribution.json" "fusion attribution result"
jepa4d_require_file "$CALIBRATION/phase2d_calibration_scale_audit.json" "calibration audit result"
jepa4d_require_file "$LATENCY/latency_aggregate.json" "latency aggregate result"
jepa4d_require_dir "$LATENCY_ROOT" "raw latency replicate root"
if [[ -d "$OUTPUT" && -n "$(find "$OUTPUT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  jepa4d_die "diagnostic aggregate output is not empty: $OUTPUT"
fi
mkdir -p "$OUTPUT"
export WANDB_MODE=online
export WANDB_DIR="${WANDB_DIR:-$JEPA4D_JOB_LOG_DIR/wandb}"
mkdir -p "$WANDB_DIR"
jepa4d_capture_environment "$REPO_ROOT" "$JEPA4D_JOB_LOG_DIR"
jepa4d_run_step "$JEPA4D_PYTHON" "$REPO_ROOT/slurm/validate_phase2d_test_receipt.py" \
  --repo-root "$REPO_ROOT" --receipt "$TEST_REPORT"
command=(
  "$JEPA4D_PYTHON" "$REPO_ROOT/scripts/aggregate_phase2d_diagnostics.py"
  --attribution-json "$ATTRIBUTION/fusion_attribution.json"
  --calibration-json "$CALIBRATION/phase2d_calibration_scale_audit.json"
  --latency-json "$LATENCY/latency_aggregate.json"
  --output "$OUTPUT"
  --repo-root "$REPO_ROOT"
  --test-receipt "$TEST_REPORT"
  --dependency-graph "$DEPENDENCY_GRAPH"
  --project "${JEPA4D_WANDB_PROJECT:-jepa4d-worldmodel}"
)
if [[ -n "${JEPA4D_WANDB_ENTITY:-}" ]]; then command+=(--entity "$JEPA4D_WANDB_ENTITY"); fi
cd "$REPO_ROOT"
jepa4d_run_step "${command[@]}"
jepa4d_require_file "$OUTPUT/phase2d_diagnostics.json" "Phase 2d diagnostic aggregate"
jepa4d_require_file "$OUTPUT/phase2d_diagnostics_report.html" "Phase 2d visual report"
jepa4d_require_file "$OUTPUT/wandb_receipt.json" "Phase 2d aggregate W&B receipt"
jepa4d_run_step "$JEPA4D_PYTHON" "$REPO_ROOT/slurm/validate_phase2d_diagnostics.py" \
  --latency-root "$LATENCY_ROOT" --latency-output "$LATENCY" \
  --attribution "$ATTRIBUTION" --calibration "$CALIBRATION" --aggregate-output "$OUTPUT" \
  --repo-root "$REPO_ROOT" --test-receipt "$TEST_REPORT" --dependency-graph "$DEPENDENCY_GRAPH" \
  --receipt-output "$OUTPUT/postflight.json"
jepa4d_require_file "$OUTPUT/postflight.json" "strict Phase 2d postflight receipt"
if grep -Eiq '<script[^>]+src=' "$OUTPUT/phase2d_diagnostics_report.html"; then
  jepa4d_die "aggregate HTML contains an external script dependency"
fi
printf 'pass\n' >"$JEPA4D_JOB_LOG_DIR/SUCCESS"
