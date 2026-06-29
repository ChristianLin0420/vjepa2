#!/usr/bin/env bash
# Build the reusable Phase 2b environment and verified assets on a login node.

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
jepa4d_init_log_dir "$REPO_ROOT" "phase2b-login-setup"
jepa4d_install_traps

CONDA_EXECUTABLE="${JEPA4D_CONDA_EXE:-${CONDA_EXE:-$HOME/miniconda3/bin/conda}}"
[[ -x "$CONDA_EXECUTABLE" ]] || jepa4d_die "cannot find conda; export JEPA4D_CONDA_EXE"
ENV_PREFIX="${JEPA4D_ENV_PREFIX:-$REPO_ROOT/.conda-gpu}"
if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  "$CONDA_EXECUTABLE" create --yes --prefix "$ENV_PREFIX" python=3.12 pip
fi
JEPA4D_PYTHON="$(jepa4d_realpath "$ENV_PREFIX/bin/python")"
export JEPA4D_PYTHON

printf '\n[%s] install the reusable Phase 2b Python environment\n' "$(date --iso-8601=seconds)"
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
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
jepa4d_capture_environment "$REPO_ROOT" "$JEPA4D_JOB_LOG_DIR"
"$CONDA_EXECUTABLE" list --prefix "$ENV_PREFIX" --explicit >"$JEPA4D_JOB_LOG_DIR/conda-explicit.txt"

ASSET_ROOT="${JEPA4D_ASSET_ROOT:-$REPO_ROOT}"
DATA_PARENT="${JEPA4D_DATA_PARENT:-$ASSET_ROOT/checkpoints/datasets}"
MANIFEST="${JEPA4D_MANIFEST:-$REPO_ROOT/jepa4d/config/benchmarks/manifests/tum_rgbd_phase2b_v1.yaml}"
ASSET_REPORT="${JEPA4D_ASSET_REPORT:-$REPO_ROOT/outputs/phase2b-gates/asset-setup-login.json}"
printf '\n[%s] download and verify immutable public assets\n' "$(date --iso-8601=seconds)"
"$JEPA4D_PYTHON" "$REPO_ROOT/slurm/download_phase2b_assets.py" \
  --manifest "$MANIFEST" \
  --asset-root "$ASSET_ROOT" \
  --data-parent "$DATA_PARENT" \
  --vjepa-revision "${JEPA4D_VJEPA_REVISION:-main}" \
  --implementation-revision \
    "${JEPA4D_VJEPA_IMPLEMENTATION_REVISION:-b22f310ee1ed02126842983d9a3adc4e296d9284}" \
  --vggt-revision "${JEPA4D_VGGT_REVISION:-d88e637d32a505f4a64de03f8588547b7f7d3ba6}" \
  --output "$ASSET_REPORT"

printf 'pass\n' >"$JEPA4D_JOB_LOG_DIR/SUCCESS"
printf '[%s] login-node environment/assets ready; report=%s\n' \
  "$(date --iso-8601=seconds)" "$ASSET_REPORT"
