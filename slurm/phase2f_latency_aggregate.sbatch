#!/usr/bin/env bash
#SBATCH --job-name=j4d-p2f-latagg
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=polar4,polar3,polar,batch_block1,grizzly,batch_block2,batch_block3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
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
jepa4d_init_log_dir "$REPO_ROOT" "phase2f-latency-aggregate"
jepa4d_install_traps
jepa4d_capture_environment "$REPO_ROOT" "$JEPA4D_JOB_LOG_DIR"
export WANDB_MODE=online PYTHONUTF8=1
OUT="${JEPA4D_STAGE_OUTPUT:?}"
LATENCY_ROOT="${JEPA4D_LATENCY_ROOT:?}"
PROVENANCE="$JEPA4D_JOB_LOG_DIR/execution_provenance.json"
prov_args=()
run_args=()
for replicate in $(seq 0 11); do
  label="$(printf 'L%02d' "$replicate")"
  receipt="$LATENCY_ROOT/replicate-$(printf '%02d' "$replicate")/latency_receipt.json"
  prov_args+=(--parent "$label=$receipt")
  run_args+=(--receipt "$receipt")
done
jepa4d_run_step "$JEPA4D_PYTHON" "$REPO_ROOT/slurm/phase2f_contract.py" emit-provenance \
  --graph "${JEPA4D_GRAPH:?}" --label LA --test-receipt "${JEPA4D_TEST_RECEIPT:?}" \
  "${prov_args[@]}" --output "$PROVENANCE"
jepa4d_run_step "$JEPA4D_PYTHON" "$REPO_ROOT/slurm/phase2f_pipeline_adapter.py" \
  --mode latency --provenance "$PROVENANCE" --output "$OUT" "${run_args[@]}"
