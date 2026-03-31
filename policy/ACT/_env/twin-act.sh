#!/usr/bin/env bash
set -euo pipefail

[[ -d detr ]] || { echo "[ERROR] cd to policy/ACT first"; exit 1; }
command -v conda >/dev/null || { echo "[ERROR] no conda"; exit 1; }
eval "$(conda shell.bash hook)"

N=$(conda env list | awk '!/^#/ && NF { if ($1 == "*") print $2; else print $1 }')
echo "$N" | grep -Fxq twin || { echo "[ERROR] create env twin first"; exit 1; }

if echo "$N" | grep -Fxq twin-act; then
  read -rp "recreate twin-act? [y/N] " a
  [[ $a == y ]] || exit 0
  conda env remove -n twin-act -y
fi

conda create -n twin-act --clone twin -y
conda activate twin-act
export PIP_USER=0 PYTHONNOUSERSITE=1

python -m pip install --no-user \
  pyquaternion pyyaml rospkg pexpect mujoco==2.3.7 dm_control==1.0.14 \
  opencv-python matplotlib einops packaging h5py ipython absl-py pyopengl glfw
(cd detr && python -m pip install --no-user .)

echo "[OK] done"
