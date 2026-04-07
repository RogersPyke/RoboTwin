#!/usr/bin/env bash
set -euo pipefail

export PIP_USER=0
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PIP_NO_BUILD_ISOLATION=1

ENV_SOURCE="twin"
ENV_TARGET="tvla-ev"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLICY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROBOTWIN_DIR="$(cd "${POLICY_DIR}/../.." && pwd)"
REQ_FILE="${POLICY_DIR}/Eval_Tiny_DexVLA_requirements.txt"
FLASH_ATTN_DIR="${ROBOTWIN_DIR}/envs/flash-attention"
CONTACT_GRASPNET_DIR="${ROBOTWIN_DIR}/envs/contact_graspnet"
CONTACT_GRASPNET_REPO="https://github.com/NVlabs/contact_graspnet.git"
CUROBO_DIR="${ROBOTWIN_DIR}/envs/curobo"
# PyAV: prebuilt on conda-forge (no local ffmpeg/Cython build; CUDA-agnostic).
AV_CONDA_SPEC="av=14.4.0"
AV_CONDA_CHANNEL="conda-forge"
eval "$(conda shell.bash hook)"

if ! conda env list | awk '{print $1}' | grep -Fxq "${ENV_SOURCE}"; then
    echo "[ERROR] Missing source conda env: ${ENV_SOURCE}"
    echo "[INFO] Please create it first, then re-run this script."
    exit 1
fi

if [[ ! -f "${REQ_FILE}" ]]; then
    echo "[ERROR] Missing requirements file: ${REQ_FILE}"
    exit 1
fi

if [[ ! -d "${FLASH_ATTN_DIR}" ]]; then
    echo "[ERROR] Missing flash-attention directory: ${FLASH_ATTN_DIR}"
    exit 1
fi

echo "[INFO] Rebuilding ${ENV_TARGET} from ${ENV_SOURCE}"
conda env remove -n "${ENV_TARGET}" -y >/dev/null 2>&1 || true
conda create -n "${ENV_TARGET}" --clone "${ENV_SOURCE}" -y

# Pip must not overwrite conda-installed wheels for packages conda still tracks.
# Otherwise conda-pack fails with "Files managed by conda were found to have been
# deleted/overwritten" (common for tqdm/idna when pip upgrades them).
echo "[INFO] Dropping conda-managed tqdm/idna so pip install -r owns them (conda-pack safe)"
conda remove -n "${ENV_TARGET}" --force tqdm idna -y 2>/dev/null || true

# Stage-1 requirements:
#   Install directly from Eval_Tiny_DexVLA_requirements.txt.
# Not listed in Eval_Tiny_DexVLA_requirements.txt on purpose:
#   - chamfer==2.0.0 (manually removed)
#   - contact-graspnet==0.0.0 (manually removed)
#      + https://github.com/NVlabs/contact_graspnet.git
#   - emd_ext==0.0.0 (manually removed)
#   - av (manually removed)
# Stage-2 manual install here:
#   1) av via conda-forge (prebuilt)
#   2) contact_graspnet via git clone + site-packages .pth (upstream has no setup.py)
#   3) curobo via envs/curobo (git clone default branch if missing) + pip install .
#   4) flash-attn via local wheel file

shopt -s nullglob
FLASH_ATTN_WHEELS=("${FLASH_ATTN_DIR}"/flash_attn-*.whl)
shopt -u nullglob
if [[ ${#FLASH_ATTN_WHEELS[@]} -eq 0 ]]; then
    echo "[ERROR] No flash-attention wheel found in: ${FLASH_ATTN_DIR}"
    exit 1
fi
FLASH_ATTN_WHEEL="${FLASH_ATTN_WHEELS[0]}"

echo "[INFO] Installing TinyVLA eval dependencies from requirements"
conda run -n "${ENV_TARGET}" python -m pip install --no-cache-dir --no-user --no-build-isolation -r "${REQ_FILE}"

if [[ ! -d "${CONTACT_GRASPNET_DIR}" ]]; then
    echo "[INFO] Cloning contact_graspnet into envs/: ${CONTACT_GRASPNET_REPO}"
    git clone "${CONTACT_GRASPNET_REPO}" "${CONTACT_GRASPNET_DIR}"
else
    echo "[INFO] Reusing existing contact_graspnet repo: ${CONTACT_GRASPNET_DIR}"
fi

echo "[INFO] Installing av in stage-2 via conda (-c ${AV_CONDA_CHANNEL}): ${AV_CONDA_SPEC}"
conda install -n "${ENV_TARGET}" -y -c "${AV_CONDA_CHANNEL}" "${AV_CONDA_SPEC}"

echo "[INFO] Registering contact_graspnet repo on sys.path (stage-2): ${CONTACT_GRASPNET_DIR}"
conda run -n "${ENV_TARGET}" env CONTACT_GRASPNET_ROOT="${CONTACT_GRASPNET_DIR}" python -c \
'import os, site
from pathlib import Path
root = Path(os.environ["CONTACT_GRASPNET_ROOT"]).resolve()
for sp in site.getsitepackages():
    (Path(sp) / "contact_graspnet_repo.pth").write_text(str(root) + "\n", encoding="ascii")'

if [[ ! -d "${CUROBO_DIR}" ]]; then
    echo "[INFO] Cloning curobo into envs/: https://github.com/NVlabs/curobo.git"
    git clone "https://github.com/NVlabs/curobo.git" "${CUROBO_DIR}"
else
    echo "[INFO] Reusing existing curobo repo: ${CUROBO_DIR}"
fi

# vcs-versioning registers setuptools_scm.parse_scm .git handlers that expect newer
# setuptools-scm Configuration (config.scm). Our pin is setuptools-scm 8.3.1; if
# vcs-versioning is present (e.g. from cloned env), metadata prep fails with:
# AttributeError: 'Configuration' object has no attribute 'scm'
echo "[INFO] Ensuring built-in setuptools-scm git backend (remove vcs-versioning if present)"
conda run -n "${ENV_TARGET}" python -m pip uninstall -y vcs-versioning 2>/dev/null || true

echo "[INFO] Installing curobo in stage-2 (minimal, no extras): ${CUROBO_DIR}"
conda run -n "${ENV_TARGET}" bash -lc "cd \"${CUROBO_DIR}\" && python -m pip install --no-cache-dir --no-user --no-build-isolation ."

echo "[INFO] Installing flash-attn wheel in stage-2: ${FLASH_ATTN_WHEEL}"
conda run -n "${ENV_TARGET}" python -m pip install --no-cache-dir --no-user --no-build-isolation "${FLASH_ATTN_WHEEL}"

echo "[OK] ${ENV_TARGET} ready"
echo "[INFO] To use: conda activate ${ENV_TARGET}"