#!/usr/bin/env bash
#SBATCH --job-name=j4d-p2g-smoke-submit-required
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
#SBATCH --signal=B:USR1@60

set -Eeuo pipefail
IFS=$'\n\t'
umask 027

bootstrap="${JEPA4D_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ ! -f "$bootstrap/slurm/lib.sh" && -f "$bootstrap/../slurm/lib.sh" ]]; then
  bootstrap="$bootstrap/.."
fi
[[ -f "$bootstrap/slurm/lib.sh" ]] || {
  printf 'ERROR: submit from the repository root through slurm/submit_phase2g_training_smoke.sh\n' >&2
  exit 2
}
# shellcheck disable=SC1091
source "$bootstrap/slurm/lib.sh"

REPO_ROOT="$(jepa4d_find_repo_root "$bootstrap")"
JOB_ID="${SLURM_JOB_ID:-}"
JOB_NAME="${SLURM_JOB_NAME:-}"
[[ "$JOB_ID" =~ ^[0-9]+$ ]] || jepa4d_die "the synthetic training smoke must run through sbatch"
[[ "$JOB_NAME" == j4d-p2g-smoke-* ]] || jepa4d_die "synthetic training-smoke job name lacks its required prefix"

jepa4d_activate_python "$REPO_ROOT"
jepa4d_init_log_dir "$REPO_ROOT" "phase2g-training-smoke"
jepa4d_install_traps

PYTHON="$REPO_ROOT/.conda-gpu/bin/python"
[[ -x "$PYTHON" ]] || jepa4d_die "the synthetic training smoke requires $PYTHON"
STAGE_OUTPUT="${JEPA4D_STAGE_OUTPUT:?submit through slurm/submit_phase2g_training_smoke.sh}"
EXECUTION_ID="${JEPA4D_EXECUTION_ID:?submit through slurm/submit_phase2g_training_smoke.sh}"
RUN_NAME="${JEPA4D_RUN_NAME:?submit through slurm/submit_phase2g_training_smoke.sh}"
SUBMITTED_COMMIT="${JEPA4D_GIT_COMMIT:?submit through slurm/submit_phase2g_training_smoke.sh}"
MAX_STEPS="${JEPA4D_MAX_STEPS:?submit through slurm/submit_phase2g_training_smoke.sh}"
WANDB_PROJECT="${JEPA4D_WANDB_PROJECT:?submit through slurm/submit_phase2g_training_smoke.sh}"
WANDB_ENTITY="${JEPA4D_WANDB_ENTITY:-}"

[[ "$(git -C "$REPO_ROOT" rev-parse --verify 'HEAD^{commit}')" == "$SUBMITTED_COMMIT" ]] || {
  jepa4d_die "repository HEAD changed after submission"
}
[[ -z "$(git -C "$REPO_ROOT" status --porcelain=v1 --untracked-files=all)" ]] || {
  jepa4d_die "repository became dirty after submission"
}
[[ ! -e "$STAGE_OUTPUT" && ! -L "$STAGE_OUTPUT" ]] || {
  jepa4d_die "fresh output path already exists: $STAGE_OUTPUT"
}
[[ "$MAX_STEPS" == "3" ]] || jepa4d_die "governed Phase 2g smoke requires exactly three steps per arm"
[[ "${WANDB_MODE:-}" == "online" ]] || jepa4d_die "synthetic training instrumentation requires WANDB_MODE=online"
[[ "${JEPA4D_GPU_MONITOR_INTERVAL:-}" == "1" ]] || jepa4d_die "GPU telemetry interval must be exactly one second"

NETRC="${HOME:?HOME is required for W&B authentication}/.netrc"
[[ -f "$NETRC" && ! -L "$NETRC" ]] || jepa4d_die "W&B authentication requires a regular HOME/.netrc"
[[ "$(stat -Lc '%a' "$NETRC")" == "600" ]] || jepa4d_die "HOME/.netrc must have mode 0600"
[[ "$(stat -Lc '%u' "$NETRC")" == "$(id -u)" ]] || jepa4d_die "HOME/.netrc must be owned by the submitting user"

command -v scontrol >/dev/null 2>&1 || jepa4d_die "scontrol is required to verify the allocation"
if ! allocation="$(scontrol show job -o "$JOB_ID")" || [[ -z "$allocation" ]]; then
  jepa4d_die "unable to resolve the active Slurm allocation"
fi
printf '%s\n' "$allocation" >"$JEPA4D_JOB_LOG_DIR/slurm-allocation.txt"
jepa4d_run_step \
  "$PYTHON" "$REPO_ROOT/slurm/validate_phase2g_training_smoke.py" \
  --job-id "$JOB_ID" \
  --allocation-only

printf '\n[%s] allocated synthetic training-smoke preflight\n' "$(date --iso-8601=seconds)"
jepa4d_run_step "$PYTHON" - <<'PY'
import json
import os

import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable inside the allocated task")
if torch.cuda.device_count() != 1:
    raise SystemExit(f"expected exactly one visible GPU, found {torch.cuda.device_count()}")
probe = torch.ones(1, device="cuda:0")
torch.cuda.synchronize()
print(json.dumps({
    "cuda_available": True,
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "device_count": torch.cuda.device_count(),
    "device_name": torch.cuda.get_device_name(0),
    "probe": float(probe.item()),
}, sort_keys=True))
PY

mkdir -p "$(dirname "$STAGE_OUTPUT")"
export WANDB_MODE=online
export PYTHONUTF8=1
export JEPA4D_GPU_MONITOR_INTERVAL=1
jepa4d_capture_environment "$REPO_ROOT" "$JEPA4D_JOB_LOG_DIR"
jepa4d_start_gpu_monitor "$JEPA4D_JOB_LOG_DIR/gpu-telemetry.csv"

command=(
  "$PYTHON" "$REPO_ROOT/scripts/run_phase2g_training_smoke.py"
  --output "$STAGE_OUTPUT"
  --max-steps "$MAX_STEPS"
  --device cuda:0
  --execution-id "$EXECUTION_ID"
  --run-name "$RUN_NAME"
  --wandb-project "$WANDB_PROJECT"
)
if [[ -n "$WANDB_ENTITY" ]]; then
  command+=(--wandb-entity "$WANDB_ENTITY")
fi

cd "$REPO_ROOT"
printf '\n[%s] synthetic M0-M3 training instrumentation smoke\n' "$(date --iso-8601=seconds)"
jepa4d_run_step "${command[@]}"

printf '\n[%s] governed Phase 2g postflight and terminal W&B publication\n' "$(date --iso-8601=seconds)"
postflight_command=(
  "$PYTHON" "$REPO_ROOT/slurm/validate_phase2g_training_smoke.py" \
  --output "$STAGE_OUTPUT" \
  --repo-root "$REPO_ROOT" \
  --job-id "$JOB_ID" \
  --run-name "$RUN_NAME" \
  --wandb-project "$WANDB_PROJECT"
)
if [[ -n "$WANDB_ENTITY" ]]; then
  postflight_command+=(--wandb-entity "$WANDB_ENTITY")
fi
jepa4d_run_step "${postflight_command[@]}"

[[ -f "$STAGE_OUTPUT/SUCCESS" && ! -L "$STAGE_OUTPUT/SUCCESS" && -s "$STAGE_OUTPUT/SUCCESS" ]] || {
  jepa4d_die "governed postflight did not produce a terminal-bound SUCCESS marker"
}
shopt -s nullglob
postflight_receipts=("$STAGE_OUTPUT"/postflight/postflight-*.json)
final_wandb_receipts=("$STAGE_OUTPUT"/wandb-final/wandb-final-*.json)
terminal_receipts=("$STAGE_OUTPUT"/terminal/terminal-*.json)
(( ${#postflight_receipts[@]} == 1 && ${#final_wandb_receipts[@]} == 1 && ${#terminal_receipts[@]} == 1 )) || {
  jepa4d_die "governed postflight did not produce exactly one immutable receipt per terminal stage"
}
printf 'pass\n' >"$JEPA4D_JOB_LOG_DIR/SUCCESS"
