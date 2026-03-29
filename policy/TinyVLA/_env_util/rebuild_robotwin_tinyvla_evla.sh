#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export CONDA_VERBOSITY="${CONDA_VERBOSITY:-1}"
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLICY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROBOTWIN_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
LOCK_FILE="${SCRIPT_DIR}/robotwin-tinyvla-evla.conda-lock.yml"
ENV_NAME="robotwin-tinyvla-evla"
crun() {
  mamba run -n "${ENV_NAME}" "$@"
}
ENV_DIR="${ROBOTWIN_ROOT}/envs"
CUROBO_DIR="${ENV_DIR}/curobo"
LOG_DIR="${POLICY_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/rebuild_robotwin_tinyvla_evla_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

log_banner() {
  echo ""
  echo "[INFO] ---------- $1 ----------"
}

run_step() {
  local st="$1"
  shift
  echo ""
  echo "[INFO] ===== STAGE: ${st} ====="
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
  echo "[INFO] ===== STAGE '${st}' exit=${ec} duration_sec=$((t1 - t0)) ====="
  return "${ec}"
}

on_err() {
  echo "[ERROR] ERR trap: BASH_COMMAND=${BASH_COMMAND:-}"
  echo "[ERROR] Log: ${LOG_FILE}"
  mamba env list 2>&1 | head -120 || true
}
trap on_err ERR

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

log_banner "remove existing env if present"
if mamba env list | awk -v env_name="${ENV_NAME}" '$1 == env_name {found=1} END{exit found?0:1}'; then
  run_step "conda_env_remove" mamba env remove -n "${ENV_NAME}" -y
else
  echo "[INFO] Env ${ENV_NAME} not present, skip remove"
fi

run_step "conda_lock_install" conda-lock install --mamba -n "${ENV_NAME}" "${LOCK_FILE}"

log_banner "post-lock: conda list"
mamba list -n "${ENV_NAME}" || true
log_banner "post-lock: pip list"
crun python -m pip list || true

run_step "mkdir_env_dir" mkdir -p "${ENV_DIR}"
if [[ ! -d "${CUROBO_DIR}" ]]; then
  run_step "git_clone_curobo" git clone --progress "https://github.com/NVlabs/curobo.git" "${CUROBO_DIR}"
else
  echo "[INFO] curobo dir exists: ${CUROBO_DIR}"
fi
run_step "pip_install_curobo_editable" crun python -m pip install -e "${CUROBO_DIR}" --no-build-isolation

log_banner "verify curobo import"
crun python -c "import curobo; print(curobo.__file__)" || true

echo "[INFO] Done. Activate: conda activate ${ENV_NAME}"
echo "[INFO] Full log: ${LOG_FILE}"
