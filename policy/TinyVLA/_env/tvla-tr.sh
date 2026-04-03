#!/usr/bin/env bash
set -euo pipefail

export PIP_USER=0
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PIP_NO_BUILD_ISOLATION=1

ENV_TARGET="tvla-tr"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLICY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROBOTWIN_DIR="$(cd "${POLICY_DIR}/../.." && pwd)"
ENV_FILE="${POLICY_DIR}/Train_Tiny_DexVLA_train.yml"
FLASH_ATTN_DIR="${ROBOTWIN_DIR}/envs/flash-attention"
# PyAV: conda-forge prebuilt (CUDA-agnostic). Unpinned like former `pip install av`.
AV_CONDA_CHANNEL="conda-forge"

eval "$(conda shell.bash hook)"

if [[ ! -f "${ENV_FILE}" ]]; then
    echo "[ERROR] Missing env file: ${ENV_FILE}"
    exit 1
fi

if [[ ! -d "${FLASH_ATTN_DIR}" ]]; then
    echo "[ERROR] Missing flash-attention directory: ${FLASH_ATTN_DIR}"
    exit 1
fi

echo "[INFO] Rebuilding ${ENV_TARGET} from ${ENV_FILE}"
conda env remove -n "${ENV_TARGET}" -y >/dev/null 2>&1 || true
conda env create -n "${ENV_TARGET}" -f "${ENV_FILE}"

echo "[INFO] Installing TinyVLA extra dependency av via conda (-c ${AV_CONDA_CHANNEL}, unpinned)"
conda install -n "${ENV_TARGET}" -y -c "${AV_CONDA_CHANNEL}" av

shopt -s nullglob
FLASH_ATTN_WHEELS=("${FLASH_ATTN_DIR}"/flash_attn-*.whl)
shopt -u nullglob
if [[ ${#FLASH_ATTN_WHEELS[@]} -eq 0 ]]; then
    echo "[ERROR] No flash-attention wheel found in: ${FLASH_ATTN_DIR}"
    exit 1
fi
FLASH_ATTN_WHEEL="${FLASH_ATTN_WHEELS[0]}"
echo "[INFO] Installing flash-attention wheel: ${FLASH_ATTN_WHEEL}"
conda run -n "${ENV_TARGET}" python -m pip install --no-cache-dir --no-user --no-build-isolation "${FLASH_ATTN_WHEEL}"

echo "[INFO] Installing local policy_heads package (non-editable)"
conda run -n "${ENV_TARGET}" python -m pip install --no-user --no-build-isolation "${POLICY_DIR}/policy_heads"

# Restore conda-managed metadata if pip downgraded wheel/packaging.
# This keeps conda-pack consistent.
echo "[INFO] Repairing conda-managed wheel/packaging for conda-pack"
conda install -n "${ENV_TARGET}" -y --force-reinstall wheel packaging

echo "[OK] ${ENV_TARGET} ready"
echo "[INFO] To use: conda activate ${ENV_TARGET}"