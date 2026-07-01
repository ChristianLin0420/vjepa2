#!/usr/bin/env bash
#SBATCH --job-name=p2gq-S-submit-required
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=polar4,polar3,polar,batch_block1,grizzly,batch_block2,batch_block3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
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
jepa4d_init_log_dir "$REPO_ROOT" "phase2g-selection"
jepa4d_install_traps
jepa4d_capture_environment "$REPO_ROOT" "$JEPA4D_JOB_LOG_DIR"
export WANDB_MODE=online PYTHONUTF8=1
OUT="${JEPA4D_STAGE_OUTPUT:?}"
EVALUATION_ROOT="${JEPA4D_EVALUATION_ROOT:?}"
PROVENANCE="$JEPA4D_JOB_LOG_DIR/execution_provenance.json"
prov_args=()
run_args=()
for arm in M0 M1 M2 M3; do
  for rotation in R0 R1 R2 R3; do
    for seed in 0 1 2; do
      label="V-$arm-$rotation-S$seed"
      receipt="$EVALUATION_ROOT/$arm/$rotation/seed-$seed/evaluation_receipt.json"
      prov_args+=(--parent "$label=$receipt")
      run_args+=(--evaluation-receipt "$receipt")
    done
  done
done
jepa4d_run_step "$JEPA4D_PYTHON" "$REPO_ROOT/slurm/phase2g_contract.py" emit-provenance \
  --graph "${JEPA4D_GRAPH:?}" --label S --test-receipt "${JEPA4D_TEST_RECEIPT:?}" \
  "${prov_args[@]}" --output "$PROVENANCE"
jepa4d_run_step "$JEPA4D_PYTHON" "$REPO_ROOT/scripts/select_phase2g_survivor.py" \
  "${run_args[@]}" --provenance "$PROVENANCE" --output "$OUT"
jepa4d_run_step "$JEPA4D_PYTHON" "$REPO_ROOT/slurm/phase2g_contract.py" verify-receipt \
  --receipt "$OUT/selector.json" --provenance "$PROVENANCE"
