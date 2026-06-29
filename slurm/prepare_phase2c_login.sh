#!/usr/bin/env bash
# Build the reusable Phase 2c environment and verified bundle on a login node.

set -Eeuo pipefail
IFS=$'\n\t'
umask 027

bootstrap="${JEPA4D_REPO_ROOT:-$PWD}"
[[ -f "$bootstrap/slurm/lib.sh" ]] || {
  printf 'ERROR: run from the repository root or export JEPA4D_REPO_ROOT\n' >&2
  exit 2
}
# shellcheck disable=SC1091
source "$bootstrap/slurm/lib.sh"

REPO_ROOT="$(jepa4d_find_repo_root "$bootstrap")"
jepa4d_init_log_dir "$REPO_ROOT" "phase2c-login-setup"
jepa4d_install_traps

CONDA_EXECUTABLE="${JEPA4D_CONDA_EXE:-${CONDA_EXE:-$HOME/miniconda3/bin/conda}}"
[[ -x "$CONDA_EXECUTABLE" ]] || jepa4d_die "cannot find conda; export JEPA4D_CONDA_EXE"
ENV_PREFIX="${JEPA4D_ENV_PREFIX:-$REPO_ROOT/.conda-gpu}"
if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  "$CONDA_EXECUTABLE" create --yes --prefix "$ENV_PREFIX" python=3.12 pip
fi
JEPA4D_PYTHON="$(jepa4d_realpath "$ENV_PREFIX/bin/python")"
export JEPA4D_PYTHON
"$JEPA4D_PYTHON" - <<'PY'
import sys
if sys.version_info[:2] != (3, 12):
    raise SystemExit(f"Phase 2c requires Python 3.12, found {sys.version}")
PY

printf '\n[%s] install the reusable Phase 2c Python environment\n' "$(date --iso-8601=seconds)"
"$JEPA4D_PYTHON" -m pip install --upgrade "pip==25.1.1" "setuptools==80.9.0" "wheel==0.45.1"
"$JEPA4D_PYTHON" -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu118 \
  --constraint "$REPO_ROOT/constraints-cuda.txt" \
  --requirement "$REPO_ROOT/requirements-cpu.txt"
"$JEPA4D_PYTHON" -m pip install timm einops "opencv-python<4.12"
"$JEPA4D_PYTHON" -m pip install -r "$REPO_ROOT/requirements-geometry.txt"
"$JEPA4D_PYTHON" -m pip install -r "$REPO_ROOT/requirements-test.txt"
"$JEPA4D_PYTHON" -m pip install \
  "pytest>=8,<9" "pytest-cov>=5,<7" "ruff>=0.9,<1" "mypy>=1.14,<2" \
  "kaleido>=1,<2" "types-PyYAML>=6.0"
"$JEPA4D_PYTHON" -m pip install --no-deps --editable "$REPO_ROOT"
"$JEPA4D_PYTHON" -m pip uninstall --yes decord >/dev/null 2>&1 || true
"$JEPA4D_PYTHON" -m pip check

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
jepa4d_capture_environment "$REPO_ROOT" "$JEPA4D_JOB_LOG_DIR"
"$CONDA_EXECUTABLE" list --prefix "$ENV_PREFIX" --explicit >"$JEPA4D_JOB_LOG_DIR/conda-explicit.txt"

jepa4d_asset_paths "$REPO_ROOT"
jepa4d_validate_assets
DATA_PARENT="${JEPA4D_DATASET_PARENT:-$REPO_ROOT/checkpoints/datasets}"
MANIFEST="${JEPA4D_MANIFEST:-$REPO_ROOT/jepa4d/config/benchmarks/manifests/tum_rgbd_phase2c_cross_sequence_v1.yaml}"
ASSET_REPORT="${JEPA4D_ASSET_REPORT:-$REPO_ROOT/outputs/phase2c-gates/asset-setup-login.json}"
jepa4d_require_file "$MANIFEST" "Phase 2c bundle manifest"
mkdir -p "$DATA_PARENT" "$(dirname "$ASSET_REPORT")"

printf '\n[%s] download and verify immutable Phase 2c sequence bundle\n' "$(date --iso-8601=seconds)"
"$JEPA4D_PYTHON" "$REPO_ROOT/slurm/download_phase2c_assets.py" \
  --manifest "$MANIFEST" \
  --data-parent "$DATA_PARENT" \
  --vjepa-checkpoint "$JEPA4D_VJEPA_CHECKPOINT" \
  --vjepa-implementation "$JEPA4D_VJEPA_IMPLEMENTATION" \
  --vggt-checkpoint "$JEPA4D_VGGT_CHECKPOINT" \
  --output "$ASSET_REPORT"

printf 'pass\n' >"$JEPA4D_JOB_LOG_DIR/SUCCESS"
printf '[%s] login-node environment/bundle ready; report=%s\n' \
  "$(date --iso-8601=seconds)" "$ASSET_REPORT"
