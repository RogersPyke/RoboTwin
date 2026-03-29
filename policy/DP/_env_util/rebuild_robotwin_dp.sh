#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export CONDA_VERBOSITY="${CONDA_VERBOSITY:-1}"
# Real-time pip/conda child output (conda run otherwise buffers/captures).
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLICY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOCK_FILE="${SCRIPT_DIR}/robotwin-dp.conda-lock.yml"
ENV_NAME="robotwin-dp"
crun() {
  mamba run -n "${ENV_NAME}" "$@"
}
LOG_DIR="${POLICY_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/rebuild_robotwin_dp_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S).log"
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

# x264 is omitted from the lock file (see robotwin-dp.conda-lock.yml.backup) because some mirrors
# lack the pinned build. Install any resolvable x264 after the locked set; ffmpeg may need it at runtime.
# Do not use run_step here: optional failure must not trigger set -e exit.
log_banner "post-lock: relaxed conda install x264 (not from lock)"
echo "[INFO] Command: mamba install -y -n ${ENV_NAME} -c conda-forge x264"
if mamba install -y -n "${ENV_NAME}" -c conda-forge x264; then
  echo "[INFO] x264 installed via mamba."
else
  echo "[WARN] mamba install x264 failed; trying PyPI fallback (may not match ffmpeg system deps)."
  crun python -m pip install --no-input x264 || echo "[WARN] pip install x264 also failed; fix manually."
fi

log_banner "post-lock: conda list"
mamba list -n "${ENV_NAME}" || true
log_banner "post-lock: pip list"
crun python -m pip list || true
log_banner "post-lock: python sys.executable and sys.path"
crun python -c "import sys,site; print('executable', sys.executable); print('prefix', sys.prefix); print('user_site', site.getusersitepackages()); [print('path', i, p) for i,p in enumerate(sys.path)]" || true

run_step "pip_install_editable_dp" crun python -m pip install -e "${POLICY_DIR}"

log_banner "final: pip show diffusion_policy (if import name differs, check setup.cfg)"
crun python -m pip list | grep -i diff || true

echo "[INFO] Done. Activate: conda activate ${ENV_NAME}"
echo "[INFO] Full log: ${LOG_FILE}"
