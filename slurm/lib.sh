#!/usr/bin/env bash

# Shared helpers for JEPA-4D Slurm jobs.  This file is sourced by the sbatch
# entrypoints after they locate the canonical repository checkout.

set -Eeuo pipefail
IFS=$'\n\t'

jepa4d_die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

jepa4d_realpath() {
  if command -v realpath >/dev/null 2>&1; then
    realpath "$1"
  else
    readlink -f "$1"
  fi
}

jepa4d_find_repo_root() {
  local requested="${1:-}"
  local candidate
  local -a candidates=()
  [[ -n "$requested" ]] && candidates+=("$requested")
  [[ -n "${JEPA4D_REPO_ROOT:-}" ]] && candidates+=("$JEPA4D_REPO_ROOT")
  [[ -n "${SLURM_SUBMIT_DIR:-}" ]] && candidates+=("$SLURM_SUBMIT_DIR" "$SLURM_SUBMIT_DIR/..")
  candidates+=("$PWD" "$PWD/..")

  for candidate in "${candidates[@]}"; do
    [[ -d "$candidate" ]] || continue
    candidate="$(jepa4d_realpath "$candidate")"
    if [[ -f "$candidate/pyproject.toml" && -f "$candidate/scripts/run_phase2b_geometry_distillation.py" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  jepa4d_die "cannot locate the JEPA-4D repository; export JEPA4D_REPO_ROOT before sbatch"
}

jepa4d_require_file() {
  local path="$1"
  local label="$2"
  [[ -f "$path" ]] || jepa4d_die "$label is not a readable file: $path"
}

jepa4d_require_dir() {
  local path="$1"
  local label="$2"
  [[ -d "$path" ]] || jepa4d_die "$label is not a readable directory: $path"
}

jepa4d_activate_python() {
  local repo_root="$1"
  if [[ -n "${JEPA4D_ENV_ACTIVATE:-}" ]]; then
    jepa4d_require_file "$JEPA4D_ENV_ACTIVATE" "JEPA4D_ENV_ACTIVATE"
    # shellcheck disable=SC1090
    source "$JEPA4D_ENV_ACTIVATE"
  fi

  if [[ -n "${JEPA4D_PYTHON:-}" ]]; then
    [[ -x "$JEPA4D_PYTHON" ]] || jepa4d_die "Python is not executable: $JEPA4D_PYTHON"
    JEPA4D_PYTHON="$(jepa4d_realpath "$JEPA4D_PYTHON")"
  elif [[ -x "$repo_root/.conda-gpu/bin/python" ]]; then
    JEPA4D_PYTHON="$(jepa4d_realpath "$repo_root/.conda-gpu/bin/python")"
  elif [[ -x "$repo_root/.venv/bin/python" ]]; then
    JEPA4D_PYTHON="$(jepa4d_realpath "$repo_root/.venv/bin/python")"
  elif command -v python3 >/dev/null 2>&1; then
    JEPA4D_PYTHON="$(command -v python3)"
  else
    jepa4d_die "no Python interpreter found; export JEPA4D_PYTHON or JEPA4D_ENV_ACTIVATE"
  fi
  [[ -x "$JEPA4D_PYTHON" ]] || jepa4d_die "Python is not executable: $JEPA4D_PYTHON"
  export JEPA4D_PYTHON
  export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
  export PYTHONUNBUFFERED=1
  export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-16}}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-16}}"
}

jepa4d_init_log_dir() {
  local repo_root="$1"
  local phase="$2"
  local job_id="${SLURM_JOB_ID:-manual-$$}"
  local job_name="${SLURM_JOB_NAME:-$phase}"
  local root="${JEPA4D_LOG_ROOT:-$repo_root/outputs/slurm_logs}"
  JEPA4D_JOB_LOG_DIR="$root/$phase/${job_name}-${job_id}"
  mkdir -p "$JEPA4D_JOB_LOG_DIR"
  JEPA4D_JOB_LOG_DIR="$(jepa4d_realpath "$JEPA4D_JOB_LOG_DIR")"
  export JEPA4D_JOB_LOG_DIR
  exec > >(tee -a "$JEPA4D_JOB_LOG_DIR/job.log") 2>&1
}

jepa4d_capture_environment() {
  local repo_root="$1"
  local log_dir="$2"
  printf '\n[%s] JEPA-4D job environment\n' "$(date --iso-8601=seconds)"
  printf 'repository=%s\n' "$repo_root"
  printf 'working_directory=%s\n' "$PWD"
  printf 'python=%s\n' "$JEPA4D_PYTHON"
  printf 'host=%s\n' "$(hostname -f 2>/dev/null || hostname)"
  printf 'job_id=%s job_name=%s partition=%s nodes=%s cpus=%s\n' \
    "${SLURM_JOB_ID:-none}" "${SLURM_JOB_NAME:-none}" "${SLURM_JOB_PARTITION:-none}" \
    "${SLURM_JOB_NODELIST:-none}" "${SLURM_CPUS_PER_TASK:-none}"
  printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
  printf 'wandb_mode=%s wandb_key_present=%s hf_token_present=%s\n' \
    "${WANDB_MODE:-unset}" "$([[ -n "${WANDB_API_KEY:-}" ]] && printf yes || printf no)" \
    "$([[ -n "${HF_TOKEN:-}" ]] && printf yes || printf no)"
  printf 'git_revision=%s\n' "$(git -C "$repo_root" rev-parse HEAD)"
  git -C "$repo_root" status --short >"$log_dir/git-status.txt"
  "$JEPA4D_PYTHON" - <<'PY' | tee "$log_dir/python-environment.txt"
import importlib.metadata
import json
import os
import platform
import sys

packages = {}
for name in ("torch", "torchvision", "numpy", "transformers", "safetensors", "vggt", "wandb", "typer"):
    try:
        packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        packages[name] = None
print(json.dumps({
    "python": sys.version,
    "executable": sys.executable,
    "platform": platform.platform(),
    "packages": packages,
    "slurm_job_id": os.getenv("SLURM_JOB_ID"),
}, indent=2, sort_keys=True))
PY
  "$JEPA4D_PYTHON" -m pip freeze >"$log_dir/pip-freeze.txt" 2>&1 || true
  if command -v scontrol >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
    scontrol show job "$SLURM_JOB_ID" >"$log_dir/slurm-job.txt" 2>&1 || true
  fi
  {
    uname -a
    ulimit -a
    free -h 2>/dev/null || true
    df -h "$repo_root" 2>/dev/null || true
  } >"$log_dir/host.txt" 2>&1
  nvidia-smi -L >"$log_dir/nvidia-smi-list.txt" 2>&1 || true
  nvidia-smi topo -m >"$log_dir/nvidia-topology.txt" 2>&1 || true
  nvidia-smi -q >"$log_dir/nvidia-smi-q.txt" 2>&1 || true
}

jepa4d_start_gpu_monitor() {
  local destination="$1"
  local interval="${JEPA4D_GPU_MONITOR_INTERVAL:-15}"
  JEPA4D_GPU_MONITOR_PID=""
  if command -v nvidia-smi >/dev/null 2>&1; then
    local query_fields selector discovered
    selector="${CUDA_VISIBLE_DEVICES:-${SLURM_JOB_GPUS:-}}"
    if [[ "$selector" == *,* ]]; then
      jepa4d_die "GPU monitor requires exactly one allocated GPU, got: $selector"
    fi
    if [[ -z "$selector" ]]; then
      discovered="$(nvidia-smi --query-gpu=uuid --format=csv,noheader,nounits 2>/dev/null || true)"
      if [[ -n "$discovered" && "$(printf '%s\n' "$discovered" | awk 'NF {count++} END {print count+0}')" == 1 ]]; then
        selector="$(printf '%s\n' "$discovered" | awk 'NF {print $1; exit}')"
      else
        jepa4d_die "GPU monitor cannot identify exactly one allocated GPU"
      fi
    fi
    query_fields="timestamp,index,uuid,name,pstate,temperature.gpu,utilization.gpu,utilization.memory"
    query_fields+=",memory.used,memory.total,power.draw,clocks.sm"
    nvidia-smi \
      --id="$selector" \
      --query-gpu="$query_fields" \
      --format=csv --loop="$interval" >"$destination" 2>&1 &
    JEPA4D_GPU_MONITOR_PID=$!
    JEPA4D_GPU_MONITOR_SELECTOR="$selector"
  fi
  export JEPA4D_GPU_MONITOR_PID JEPA4D_GPU_MONITOR_SELECTOR
}

jepa4d_stop_gpu_monitor() {
  if [[ -n "${JEPA4D_GPU_MONITOR_PID:-}" ]]; then
    kill "$JEPA4D_GPU_MONITOR_PID" >/dev/null 2>&1 || true
    wait "$JEPA4D_GPU_MONITOR_PID" >/dev/null 2>&1 || true
    JEPA4D_GPU_MONITOR_PID=""
  fi
}

jepa4d_install_traps() {
  trap 'printf "[%s] Slurm time-limit warning received (USR1)\n" "$(date --iso-8601=seconds)" >&2' USR1
  trap 'status=$?; jepa4d_stop_gpu_monitor; printf "[%s] job_exit_status=%s\n" "$(date --iso-8601=seconds)" "$status"; exit "$status"' EXIT
}

jepa4d_asset_paths() {
  local repo_root="$1"
  local asset_root="${JEPA4D_ASSET_ROOT:-$repo_root}"
  JEPA4D_VJEPA_CHECKPOINT="${JEPA4D_VJEPA_CHECKPOINT:-$asset_root/checkpoints/phase2b_assets/vjepa2.1-vitb-fpc64-384}"
  JEPA4D_VJEPA_IMPLEMENTATION="${JEPA4D_VJEPA_IMPLEMENTATION:-$asset_root/checkpoints/phase2b_assets/vjepa21_hf_impl}"
  JEPA4D_VGGT_CHECKPOINT="${JEPA4D_VGGT_CHECKPOINT:-$asset_root/checkpoints/phase2b_assets/VGGT-1B}"
  export JEPA4D_VJEPA_CHECKPOINT JEPA4D_VJEPA_IMPLEMENTATION JEPA4D_VGGT_CHECKPOINT
}

jepa4d_validate_assets() {
  jepa4d_require_dir "$JEPA4D_VJEPA_CHECKPOINT" "V-JEPA checkpoint"
  jepa4d_require_file "$JEPA4D_VJEPA_CHECKPOINT/config.json" "V-JEPA config"
  jepa4d_require_file "$JEPA4D_VJEPA_CHECKPOINT/model.safetensors" "V-JEPA weights"
  jepa4d_require_dir "$JEPA4D_VJEPA_IMPLEMENTATION" "V-JEPA compatibility implementation"
  jepa4d_require_file "$JEPA4D_VJEPA_IMPLEMENTATION/configuration_vjepa21.py" "V-JEPA compatibility config"
  jepa4d_require_file "$JEPA4D_VJEPA_IMPLEMENTATION/modeling_vjepa21.py" "V-JEPA compatibility model"
  jepa4d_require_dir "$JEPA4D_VGGT_CHECKPOINT" "VGGT checkpoint"
  jepa4d_require_file "$JEPA4D_VGGT_CHECKPOINT/config.json" "VGGT config"
  JEPA4D_VJEPA_CHECKPOINT="$(jepa4d_realpath "$JEPA4D_VJEPA_CHECKPOINT")"
  JEPA4D_VJEPA_IMPLEMENTATION="$(jepa4d_realpath "$JEPA4D_VJEPA_IMPLEMENTATION")"
  JEPA4D_VGGT_CHECKPOINT="$(jepa4d_realpath "$JEPA4D_VGGT_CHECKPOINT")"
  export JEPA4D_VJEPA_CHECKPOINT JEPA4D_VJEPA_IMPLEMENTATION JEPA4D_VGGT_CHECKPOINT
}

jepa4d_run_step() {
  if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v srun >/dev/null 2>&1; then
    srun --kill-on-bad-exit=1 --unbuffered "$@"
  else
    "$@"
  fi
}
