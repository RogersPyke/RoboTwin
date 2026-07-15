#!/usr/bin/env bash
# Install the RoboTwin runtime in the currently active conda environment.
# The script is intentionally idempotent and must be run from any directory.
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python}"
PIP=("$PYTHON_BIN" -m pip)

echo "[RoboTwin] Python: $($PYTHON_BIN -c 'import sys; print(sys.executable)')"
"${PIP[@]}" install -r script/requirements.txt
"${PIP[@]}" install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation

# RoboTwin's Curobo extension is compiled by nvcc. CUDA 12.1 rejects GCC > 12;
# prefer the system GCC 11 toolchain when available, without modifying global state.
if [[ -x /usr/bin/gcc-11 && -x /usr/bin/g++-11 ]]; then
    export CC=/usr/bin/gcc-11
    export CXX=/usr/bin/g++-11
    export CUDAHOSTCXX=/usr/bin/g++-11
    echo "[RoboTwin] Using GCC 11 for CUDA host compilation."
elif command -v x86_64-conda-linux-gnu-gcc >/dev/null 2>&1; then
    echo "[RoboTwin] Using conda compiler: $(command -v x86_64-conda-linux-gnu-gcc)"
else
    echo "[RoboTwin] WARNING: GCC 11 not found; nvcc may reject the default compiler." >&2
fi

# Keep the RoboTwin-pinned Warp version before installing Curobo.
"${PIP[@]}" install warp-lang==1.12.0

CUROBO_DIR="$ROOT_DIR/envs/curobo"
if [[ ! -d "$CUROBO_DIR/.git" ]]; then
    if [[ -e "$CUROBO_DIR" ]]; then
        BACKUP_DIR="${CUROBO_DIR}.bak_$(date -u +%Y%m%d%H%M%S)"
        mv "$CUROBO_DIR" "$BACKUP_DIR"
        echo "[RoboTwin] Backed up non-git Curobo directory to $BACKUP_DIR"
    fi
    git clone --branch v0.7.8 --depth 1 https://github.com/NVlabs/curobo.git "$CUROBO_DIR"
else
    echo "[RoboTwin] Reusing existing Curobo checkout: $CUROBO_DIR"
fi
MAX_JOBS="${MAX_JOBS:-$(nproc 2>/dev/null || echo 2)}"
MAX_JOBS=$(( MAX_JOBS > 4 ? 4 : MAX_JOBS ))
MAX_JOBS="$MAX_JOBS" "${PIP[@]}" install -e "$CUROBO_DIR" --no-build-isolation
"${PIP[@]}" install setuptools==69.5.1

# Apply RoboTwin's known compatibility edits, preserving a backup first.
SAPIEN_LOCATION="$($PYTHON_BIN -c 'import sapien, os; print(os.path.dirname(sapien.__file__))')"
URDF_LOADER="$SAPIEN_LOCATION/wrapper/urdf_loader.py"
if [[ -f "$URDF_LOADER" ]]; then
    cp -n "$URDF_LOADER" "$URDF_LOADER.bak_robotwin" 2>/dev/null || true
    sed -i -E 's/("r")([)])( as)/\1, encoding="utf-8"\2 as/g' "$URDF_LOADER"
fi
MPLIB_LOCATION="$($PYTHON_BIN -c 'import mplib, os; print(os.path.dirname(mplib.__file__))')"
PLANNER="$MPLIB_LOCATION/planner.py"
if [[ -f "$PLANNER" ]]; then
    cp -n "$PLANNER" "$PLANNER.bak_robotwin" 2>/dev/null || true
    sed -i -E 's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' "$PLANNER"
fi

echo "[RoboTwin] Installation complete. Download assets separately with:"
echo "  cd $ROOT_DIR && bash script/_download_assets.sh"
