#!/bin/bash
export PYTHONNOUSERSITE=1

# == keep unchanged ==
policy_name=ACT

# ====== Wrapper Modification ======
# Joint (multi-task) checkpoint eval: MUST use --config <name> with _ev_cfg/<name>.yaml.
# Each eval_result/.../run folder gets a copy _ev_cfg_<name>.yaml via ACT_EV_CFG_SNAPSHOT_SRC.
# Positional-args eval below is for single-task checkpoints only (act-<one_task>/...).
if [[ "$1" == "--config" && -n "${2:-}" ]]; then
    cd "$(dirname "$0")"
    python3 ./_ev_wrapper.py --config "$2"
    exit 0
fi
# ==================

task_name=${1}
task_config=${2}
ckpt_setting=${3}
expert_data_num=${4}
seed=${5}
gpu_id=${6}
# temporal_agg=${5} # use temporal_agg
DEBUG=False

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

cd ../..

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --ckpt_dir policy/ACT/act_ckpt/act-${task_name}/${ckpt_setting}-${expert_data_num} \
    --seed ${seed} \
    --temporal_agg true