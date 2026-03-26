#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/env.yml"
LOCK_FILE="${SCRIPT_DIR}/conda-lock.yml"
ENV_NAME="robotwin-act"
OPEN3D_VERSION="0.18.0"
PYTORCH3D_GIT_URL="git+https://github.com/facebookresearch/pytorch3d.git"
ENV_DIR="${SCRIPT_DIR}/../../../envs"
CUROBO_DIR="${ENV_DIR}/curobo"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "[ERROR] Missing file: ${ENV_FILE}"
  exit 1
fi

echo "[INFO] Generate lock file from env.yml"
conda-lock -f "${ENV_FILE}" -p linux-64 --kind lock --lockfile "${LOCK_FILE}"

if [[ ! -f "${LOCK_FILE}" ]]; then
  echo "[ERROR] Failed to generate lock file: ${LOCK_FILE}"
  exit 1
fi

echo "[INFO] Recreate conda env: ${ENV_NAME}"
if conda env list | awk -v env_name="${ENV_NAME}" '$1 == env_name {found=1} END{exit found?0:1}'; then
  conda env remove -n "${ENV_NAME}" -y
fi
conda-lock install -n "${ENV_NAME}" "${LOCK_FILE}"

echo "[INFO] Install open3d from PyPI"
conda run -n "${ENV_NAME}" pip install "open3d==${OPEN3D_VERSION}"

echo "[INFO] Install pytorch3d from Git"
conda run -n "${ENV_NAME}" pip install "${PYTORCH3D_GIT_URL}"

echo "[INFO] Install curobo from source"
mkdir -p "${ENV_DIR}"
if [[ ! -d "${CUROBO_DIR}" ]]; then
  git clone "https://github.com/NVlabs/curobo.git" "${CUROBO_DIR}"
fi
conda run -n "${ENV_NAME}" pip install -e "${CUROBO_DIR}" --no-build-isolation

echo "[INFO] Done. Activate: conda activate ${ENV_NAME}"
