#!/usr/bin/env bash
set -euo pipefail

export PIP_USER=0
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PIP_NO_BUILD_ISOLATION=1

ENV_SOURCE="twin"
ENV_TARGET="twin-dp"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLICY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# PyAV: conda-forge prebuilt (CUDA-agnostic). Unpinned; avoids pip + PIP_NO_BUILD_ISOLATION build breakage.
AV_CONDA_CHANNEL="conda-forge"

eval "$(conda shell.bash hook)"

echo "[INFO] Rebuilding ${ENV_TARGET} from ${ENV_SOURCE}"
conda env remove -n "${ENV_TARGET}" -y >/dev/null 2>&1 || true
conda create -n "${ENV_TARGET}" --clone "${ENV_SOURCE}" -y

echo "[INFO] Installing av via conda (-c ${AV_CONDA_CHANNEL}, unpinned)"
conda install -n "${ENV_TARGET}" -y -c "${AV_CONDA_CHANNEL}" av

echo "[INFO] Installing DP dependencies (pip PEP 517 isolation; unset PIP_NO_BUILD_ISOLATION for this step)"
conda run -n "${ENV_TARGET}" env -u PIP_NO_BUILD_ISOLATION python -m pip install --no-cache-dir --no-user \
    zarr==2.12.0 \
    wandb \
    ipdb \
    gpustat \
    dm_control \
    omegaconf \
    hydra-core==1.2.0 \
    dill==0.3.5.1 \
    einops==0.4.1 \
    diffusers==0.11.1 \
    numba==0.56.4 \
    moviepy \
    imageio \
    matplotlib \
    termcolor \
    sympy

echo "[INFO] Installing local policy package (non-editable)"
conda run -n "${ENV_TARGET}" python -m pip install --no-cache-dir --no-user --no-build-isolation "${POLICY_DIR}"

echo "[OK] ${ENV_TARGET} ready"
echo "[INFO] To use: conda activate ${ENV_TARGET}"