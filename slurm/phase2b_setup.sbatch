#!/usr/bin/env bash
#SBATCH --job-name=jepa4d-p2b-setup
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
jepa4d_init_log_dir "$REPO_ROOT" "phase2b-setup"
jepa4d_install_traps

CONDA_EXECUTABLE="${JEPA4D_CONDA_EXE:-${CONDA_EXE:-$HOME/miniconda3/bin/conda}}"
[[ -x "$CONDA_EXECUTABLE" ]] || jepa4d_die "cannot find conda; export JEPA4D_CONDA_EXE"
ENV_PREFIX="${JEPA4D_ENV_PREFIX:-$REPO_ROOT/.conda-gpu}"
if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  "$CONDA_EXECUTABLE" create --yes --prefix "$ENV_PREFIX" python=3.12 pip
fi
ENV_PREFIX="$(jepa4d_realpath "$ENV_PREFIX")"
JEPA4D_PYTHON="$ENV_PREFIX/bin/python"
export JEPA4D_PYTHON
"$JEPA4D_PYTHON" - <<'PY'
import sys
if sys.version_info[:2] != (3, 12):
    raise SystemExit(f"Phase 2b requires Python 3.12, found {sys.version}")
PY

printf '\n[%s] install pinned CUDA, geometry, project, and test dependencies\n' "$(date --iso-8601=seconds)"
"$JEPA4D_PYTHON" -m pip install --upgrade "pip==25.1.1" "setuptools==80.9.0" "wheel==0.45.1"
"$JEPA4D_PYTHON" -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu118 \
  --constraint "$REPO_ROOT/constraints-cuda.txt" \
  --requirement "$REPO_ROOT/requirements-cpu.txt"
"$JEPA4D_PYTHON" -m pip install timm einops "opencv-python<4.12"
"$JEPA4D_PYTHON" -m pip install -r "$REPO_ROOT/requirements-geometry.txt"
"$JEPA4D_PYTHON" -m pip install -r "$REPO_ROOT/requirements-test.txt"
"$JEPA4D_PYTHON" -m pip install "pytest>=8,<9" "pytest-cov>=5,<7" "ruff>=0.9,<1" "mypy>=1.14,<2" \
  "kaleido>=1,<2" "types-PyYAML>=6.0"
"$JEPA4D_PYTHON" -m pip install --no-deps --editable "$REPO_ROOT"
# `decord` is required only by the upstream video-training stack and its
# legacy wheel declares itself unsupported on Python 3.12. Phase 2b loads PNG
# frames through Pillow and must not inherit this unrelated broken package
# from a partially completed setup attempt.
"$JEPA4D_PYTHON" -m pip uninstall --yes decord >/dev/null 2>&1 || true
"$JEPA4D_PYTHON" -m pip check

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
jepa4d_capture_environment "$REPO_ROOT" "$JEPA4D_JOB_LOG_DIR"
"$CONDA_EXECUTABLE" list --prefix "$ENV_PREFIX" --explicit >"$JEPA4D_JOB_LOG_DIR/conda-explicit.txt"
jepa4d_start_gpu_monitor "$JEPA4D_JOB_LOG_DIR/gpu-telemetry.csv"

printf '\n[%s] validate installed CUDA runtime\n' "$(date --iso-8601=seconds)"
jepa4d_run_step "$JEPA4D_PYTHON" "$REPO_ROOT/scripts/check_cuda.py" \
  --device "${JEPA4D_CUDA_DEVICE_INDEX:-0}" \
  --stress-seconds "${JEPA4D_SETUP_CUDA_STRESS_SECONDS:-15}" \
  --matrix-size "${JEPA4D_CUDA_MATRIX_SIZE:-4096}" \
  --allocation-mib "${JEPA4D_SETUP_CUDA_ALLOCATION_MIB:-1024}" \
  --json-output "$JEPA4D_JOB_LOG_DIR/cuda-health.json"

ASSET_ROOT="${JEPA4D_ASSET_ROOT:-$REPO_ROOT}"
DATA_PARENT="${JEPA4D_DATA_PARENT:-$ASSET_ROOT/checkpoints/datasets}"
MANIFEST="${JEPA4D_MANIFEST:-$REPO_ROOT/jepa4d/config/benchmarks/manifests/tum_rgbd_phase2b_v1.yaml}"
mkdir -p "$ASSET_ROOT" "$DATA_PARENT"

printf '\n[%s] download and checksum public Phase 2b assets\n' "$(date --iso-8601=seconds)"
jepa4d_run_step "$JEPA4D_PYTHON" "$REPO_ROOT/slurm/download_phase2b_assets.py" \
  --manifest "$MANIFEST" \
  --asset-root "$ASSET_ROOT" \
  --data-parent "$DATA_PARENT" \
  --vjepa-revision "${JEPA4D_VJEPA_REVISION:-main}" \
  --implementation-revision \
    "${JEPA4D_VJEPA_IMPLEMENTATION_REVISION:-b22f310ee1ed02126842983d9a3adc4e296d9284}" \
  --vggt-revision "${JEPA4D_VGGT_REVISION:-d88e637d32a505f4a64de03f8588547b7f7d3ba6}" \
  --output "$JEPA4D_JOB_LOG_DIR/asset-setup.json"

printf 'pass\n' >"$JEPA4D_JOB_LOG_DIR/SUCCESS"
printf '[%s] environment and assets are ready; manifest=%s\n' \
  "$(date --iso-8601=seconds)" "$JEPA4D_JOB_LOG_DIR/asset-setup.json"
