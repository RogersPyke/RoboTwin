#!/usr/bin/env bash
#
# Script purpose:
# - Install dexvla-robo from lock file directly.
# - The lock file intentionally excludes flash-attn and av.
# - Reason: av source build may fail due to FFmpeg API/header mismatch.
# - deepspeed is kept in lock and installed by conda-lock.
# - flash-attn and av are installed manually at the end.
# - flash-attn is installed ONLY from local wheel with --no-deps.
#
set -euo pipefail
export PYTHONNOUSERSITE=1
export PIP_USER=0
export CONDA_VERBOSITY="${CONDA_VERBOSITY:-1}"
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLICY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROBOTWIN_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
LOCK_FILE="${SCRIPT_DIR}/dexvla-robo.conda-lock.yml"
ENV_NAME="dexvla-robo"
crun() {
  mamba run -n "${ENV_NAME}" "$@"
}
TORCH_VERSION="2.3.0"
FLASH_ATTN_CUDA_MAJOR="12"
FLASH_ATTN_CXX11_ABI="FALSE"
POLICY_HEADS_DIR="${POLICY_DIR}/policy_heads"
ENV_DIR="${ROBOTWIN_ROOT}/envs"
FLASH_ATTN_DIR="${ENV_DIR}/flash-attention"
CUROBO_DIR="${ENV_DIR}/curobo"
LOG_DIR="${POLICY_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/rebuild_dexvla_robo_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

LAST_CMD=""
CURRENT_STAGE="init"

log_banner() {
  echo ""
  echo "[INFO] ---------- $1 ----------"
}

on_error() {
  local exit_code="$1"
  local line_no="$2"
  echo "[ERROR] Stage: ${CURRENT_STAGE}"
  echo "[ERROR] Failed command (line ${line_no}): ${LAST_CMD}"
  echo "[ERROR] BASH_COMMAND: ${BASH_COMMAND:-}"
  echo "[ERROR] Exit code: ${exit_code}"
  echo "[ERROR] Debug snapshot:"
  echo "[ERROR]   env name: ${ENV_NAME}"
  echo "[ERROR]   lock file: ${LOCK_FILE}"
  mamba env list || true
  crun python -V || true
  crun python -m pip --version || true
  crun python -m pip show torch psutil setuptools wheel ninja packaging || true
  crun python -c "import sys,site; print('sys.executable=', sys.executable); print('user_site=', site.getusersitepackages())" || true
  crun python -c "import sys; [print('path', i, p) for i,p in enumerate(sys.path)]" || true
  echo "[ERROR] Full log: ${LOG_FILE}"
}
trap 'on_error $? $LINENO' ERR

run_step() {
  local stage="$1"
  shift
  CURRENT_STAGE="${stage}"
  LAST_CMD=$(printf '%q ' "$@")
  echo ""
  echo "[INFO] ===== STAGE: ${CURRENT_STAGE} ====="
  echo -n "[INFO] Command:"
  local a
  for a in "$@"; do
    printf ' %q' "$a"
  done
  echo ""
  local t0 t1 ec
  t0=$(date +%s)
  "$@"
  ec=$?
  t1=$(date +%s)
  echo "[INFO] ===== STAGE '${CURRENT_STAGE}' exit=${ec} duration_sec=$((t1 - t0)) ====="
  return "${ec}"
}

install_flash_attn_from_local_wheel_or_exit() {
  local py_tag
  local torch_mm
  local cuda_major
  local abi_flag
  local wheel_file=""
  local expected_glob_plus
  local expected_glob_pct
  local candidates_plus=()
  local candidates_pct=()

  py_tag="$(mamba run -n "${ENV_NAME}" python -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")"
  torch_mm="$(echo "${TORCH_VERSION}" | awk -F. '{print $1 "." $2}')"
  cuda_major="${FLASH_ATTN_CUDA_MAJOR}"
  abi_flag="${FLASH_ATTN_CXX11_ABI}"

  echo "[INFO] FlashAttn required versions:"
  echo "[INFO]   torch: ${torch_mm}"
  echo "[INFO]   python tag: ${py_tag}"
  echo "[INFO]   cuda runtime major: ${cuda_major}"
  echo "[INFO]   cxx11 abi: ${abi_flag}"

  expected_glob_plus="${FLASH_ATTN_DIR}/flash_attn-*+cu${cuda_major}torch${torch_mm}cxx11abi${abi_flag}-${py_tag}-${py_tag}-linux_x86_64.whl"
  expected_glob_pct="${FLASH_ATTN_DIR}/flash_attn-*%2Bcu${cuda_major}torch${torch_mm}cxx11abi${abi_flag}-${py_tag}-${py_tag}-linux_x86_64.whl"
  shopt -s nullglob
  candidates_plus=( ${expected_glob_plus} )
  candidates_pct=( ${expected_glob_pct} )
  shopt -u nullglob
  if [[ ${#candidates_plus[@]} -gt 0 ]]; then
    wheel_file="${candidates_plus[0]}"
  elif [[ ${#candidates_pct[@]} -gt 0 ]]; then
    wheel_file="${candidates_pct[0]}"
  fi

  if [[ -z "${wheel_file}" ]]; then
    echo "[ERROR] No compatible flash-attn wheel found."
    echo "[ERROR] Required pattern: ${expected_glob_plus}"
    echo "[ERROR] Alternate encoded pattern: ${expected_glob_pct}"
    echo "[ERROR] Place the wheel manually under: ${FLASH_ATTN_DIR}"
    echo "[ERROR] Removing incomplete environment: ${ENV_NAME}"
    mamba env remove -n "${ENV_NAME}" -y || true
    exit 1
  fi

  echo "[INFO] Using local flash-attn wheel: ${wheel_file}"
  run_step "install_flash_attn_local_wheel" crun python -m pip install --no-deps "${wheel_file}"
}

echo "[INFO] Log file: ${LOG_FILE}"
echo "[INFO] Time UTC: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "[INFO] Time Asia/Shanghai: $(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S %Z')"
echo "[INFO] Host: $(hostname 2>/dev/null || echo unknown) uname: $(uname -srvmo 2>/dev/null || echo unknown)"
command -v conda >/dev/null 2>&1 && echo "[INFO] $(conda --version 2>&1)" || echo "[WARN] conda not on PATH"
if ! command -v mamba >/dev/null 2>&1; then
  run_step "conda_install_mamba_base" conda install -n base -c conda-forge mamba -y
fi
if ! command -v conda-lock >/dev/null 2>&1; then
  run_step "mamba_install_conda_lock_base" mamba install -n base -c conda-forge conda-lock -y
fi
command -v mamba >/dev/null 2>&1 && echo "[INFO] $(mamba --version 2>&1)" || echo "[WARN] mamba not on PATH"
command -v conda-lock >/dev/null 2>&1 && echo "[INFO] conda-lock: $(conda-lock --version 2>&1 || true)" || echo "[WARN] conda-lock not on PATH"

if [[ ! -f "${LOCK_FILE}" ]]; then
  echo "[ERROR] Missing lock file: ${LOCK_FILE}"
  echo "[ERROR] This script is lock-only. Prepare lock file first."
  exit 1
fi

echo "[INFO] Recreate conda env: ${ENV_NAME}"
if mamba env list | awk -v env_name="${ENV_NAME}" '$1 == env_name {found=1} END{exit found?0:1}'; then
  run_step "remove_env" mamba env remove -n "${ENV_NAME}" -y
fi

echo "[INFO] Create bootstrap env for torch preinstall"
run_step "create_env" mamba create -n "${ENV_NAME}" python=3.10 -y
echo "[INFO] Preinstall torch==${TORCH_VERSION} before lock install"
run_step "bootstrap_pip" crun python -m pip install --upgrade pip setuptools wheel
run_step "bootstrap_build_deps" crun python -m pip install "ninja" "packaging" "psutil"
run_step "bootstrap_torch" crun python -m pip install "torch==${TORCH_VERSION}"
echo "[INFO] Verify torch import from env before lock install"
run_step "verify_torch" crun env EXPECTED_ENV_FRAGMENT="envs/${ENV_NAME}" python -c 'import os,site,torch; expected=os.environ["EXPECTED_ENV_FRAGMENT"]; print(torch.__file__); print(site.getusersitepackages()); assert expected in torch.__file__, "torch is not from target conda env"; assert not os.path.exists(site.getusersitepackages()) or "site-packages" in site.getusersitepackages()'

echo "[INFO] Install lock packages (lock excludes deferred pip packages)"
run_step "conda_lock_install" conda-lock install --mamba -n "${ENV_NAME}" "${LOCK_FILE}"

log_banner "post-lock: conda list"
mamba list -n "${ENV_NAME}" || true
log_banner "post-lock: pip list"
crun python -m pip list || true

echo "[INFO] Install deferred package at end"
run_step "prepare_env_dir" mkdir -p "${ENV_DIR}"
install_flash_attn_from_local_wheel_or_exit

echo "[INFO] Install curobo from source"
run_step "prepare_env_dir_for_curobo" mkdir -p "${ENV_DIR}"
if [[ ! -d "${CUROBO_DIR}" ]]; then
  run_step "clone_curobo" git clone --progress "https://github.com/NVlabs/curobo.git" "${CUROBO_DIR}"
fi
run_step "install_curobo_editable" crun python -m pip install -e "${CUROBO_DIR}" --no-build-isolation

if [[ -d "${POLICY_HEADS_DIR}" ]]; then
  echo "[INFO] Install policy_heads in editable mode"
  run_step "install_policy_heads_editable" crun python -m pip install -e "${POLICY_HEADS_DIR}"
else
  echo "[WARN] policy_heads directory not found: ${POLICY_HEADS_DIR}"
fi

log_banner "final: pip list (torch / deepspeed / flash if present)"
crun python -m pip list | grep -iE 'torch|deepspeed|flash|cuda|curobo' || true

echo "[INFO] Done. Activate: conda activate ${ENV_NAME}"
echo "[INFO] Full log: ${LOG_FILE}"
