#!/usr/bin/env bash
set -euo pipefail

export PIP_USER=0
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PIP_NO_BUILD_ISOLATION=1

ENV_SOURCE="twin"
ENV_TARGET="twin-act"

eval "$(conda shell.bash hook)"

echo "[INFO] Rebuilding ${ENV_TARGET} from ${ENV_SOURCE}"
conda env remove -n "${ENV_TARGET}" -y >/dev/null 2>&1 || true
conda create -n "${ENV_TARGET}" --clone "${ENV_SOURCE}" -y

echo "[INFO] Installing ACT dependencies"
conda run -n "${ENV_TARGET}" python -m pip install --no-cache-dir --no-user --no-build-isolation \
    pyquaternion \
    pyyaml \
    rospkg \
    pexpect \
    mujoco==2.3.7 \
    dm_control==1.0.14 \
    opencv-python \
    matplotlib \
    einops \
    packaging \
    h5py \
    ipython \
    absl-py \
    pyopengl \
    glfw

echo "[INFO] Installing local DETR package (non-editable)"
conda run -n "${ENV_TARGET}" python -m pip install --no-cache-dir --no-user --no-build-isolation ./detr

echo "[OK] ${ENV_TARGET} ready"
echo "[INFO] To use: conda activate ${ENV_TARGET}"