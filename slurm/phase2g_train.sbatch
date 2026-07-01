#!/usr/bin/env bash
#SBATCH --job-name=p2gq-F-dispatch-only
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=polar4,polar3,polar,batch_block1,grizzly,batch_block2,batch_block3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --signal=B:USR1@120

set -Eeuo pipefail
IFS=$'\n\t'
umask 027
ROOT="${JEPA4D_REPO_ROOT:?}"
source "$ROOT/slurm/lib.sh"
REPO_ROOT="$(jepa4d_find_repo_root "$ROOT")"
jepa4d_activate_python "$REPO_ROOT"
jepa4d_init_log_dir "$REPO_ROOT" "phase2g-formal"
jepa4d_install_traps
jepa4d_capture_environment "$REPO_ROOT" "$JEPA4D_JOB_LOG_DIR"
jepa4d_start_gpu_monitor "$JEPA4D_JOB_LOG_DIR/gpu-telemetry.csv"
export JEPA4D_GPU_TELEMETRY_PATH="$JEPA4D_JOB_LOG_DIR/gpu-telemetry.csv"
export WANDB_MODE=online PYTHONUTF8=1
OUT="${JEPA4D_STAGE_OUTPUT:?}"
VIEW="${JEPA4D_CACHE_VIEW:?}"
PROVENANCE="$JEPA4D_JOB_LOG_DIR/execution_provenance.json"
[[ -d "$VIEW" ]] || jepa4d_die "rotation-scoped cache view is missing: $VIEW"
jepa4d_run_step "$JEPA4D_PYTHON" "$REPO_ROOT/slurm/phase2g_contract.py" emit-provenance \
  --graph "${JEPA4D_GRAPH:?}" --label "${JEPA4D_JOB_LABEL:?}" --test-receipt "${JEPA4D_TEST_RECEIPT:?}" \
  --parent "HG=${JEPA4D_LR_SELECTION:?}" --output "$PROVENANCE"
jepa4d_run_step "$JEPA4D_PYTHON" "$REPO_ROOT/scripts/run_phase2g_formal_training.py" \
  --arm "${JEPA4D_ARM:?}" --rotation "${JEPA4D_ROTATION:?}" --seed "${JEPA4D_SEED:?}" \
  --cache-root "$VIEW" --lr-selection "$JEPA4D_LR_SELECTION" --provenance "$PROVENANCE" --output "$OUT"
jepa4d_run_step "$JEPA4D_PYTHON" "$REPO_ROOT/slurm/phase2g_contract.py" verify-receipt \
  --receipt "$OUT/training_receipt.json" --provenance "$PROVENANCE"
