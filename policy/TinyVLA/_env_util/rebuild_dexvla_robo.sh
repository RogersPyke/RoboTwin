#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLICY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROBOTWIN_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
LOCK_FILE="${SCRIPT_DIR}/dexvla-robo.conda-lock.yml"
ENV_NAME="dexvla-robo"
POLICY_HEADS_DIR="${POLICY_DIR}/policy_heads"
ENV_DIR="${ROBOTWIN_ROOT}/envs"
CUROBO_DIR="${ENV_DIR}/curobo"

if [[ ! -f "${LOCK_FILE}" ]]; then
  echo "[ERROR] Missing lock file: ${LOCK_FILE}"
  echo "[ERROR] This script is lock-only. Prepare lock file first."
  exit 1
fi

if command -v conda-lock >/dev/null 2>&1; then
  CONDA_LOCK_CMD=(conda-lock)
else
  CONDA_LOCK_CMD=(conda run -n robotwin-act conda-lock)
fi

echo "[INFO] Recreate conda env: ${ENV_NAME}"
if conda env list | awk -v env_name="${ENV_NAME}" '$1 == env_name {found=1} END{exit found?0:1}'; then
  conda env remove -n "${ENV_NAME}" -y
fi
"${CONDA_LOCK_CMD[@]}" install -n "${ENV_NAME}" "${LOCK_FILE}"

echo "[INFO] Install curobo from source"
mkdir -p "${ENV_DIR}"
if [[ ! -d "${CUROBO_DIR}" ]]; then
  git clone "https://github.com/NVlabs/curobo.git" "${CUROBO_DIR}"
fi
conda run -n "${ENV_NAME}" pip install -e "${CUROBO_DIR}" --no-build-isolation

if [[ -d "${POLICY_HEADS_DIR}" ]]; then
  echo "[INFO] Install policy_heads in editable mode"
  conda run -n "${ENV_NAME}" pip install -e "${POLICY_HEADS_DIR}"
else
  echo "[WARN] policy_heads directory not found: ${POLICY_HEADS_DIR}"
fi

echo "[INFO] Done. Activate: conda activate ${ENV_NAME}"
