#!/bin/bash
export PYTHONNOUSERSITE=1

# == keep unchanged ==
policy_name=ACT

# ====== Wrapper Modification ======
# Deprecated:
#   bash eval.sh --config <name>
#
# Use:
#   bash _eval.sh <name>
if [[ "${1:-}" == "--config" ]]; then
    echo -e "\033[31m[ERROR] Deprecated: bash eval.sh --config <name>\033[0m"
    echo -e "\033[33mUse: bash _eval.sh <name>\033[0m"
    exit 2
fi

task_name=${1}
task_config=${2}
ckpt_setting=${3}
expert_data_num=${4}
seed=${5}
gpu_id=${6}
unused_early_stop_patience_epochs=${7:-}
unused_early_stop_rel_tol=${8:-}
# temporal_agg=${5} # use temporal_agg
DEBUG=False

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
if [[ -n "${unused_early_stop_patience_epochs}" || -n "${unused_early_stop_rel_tol}" ]]; then
    echo -e "\033[31m[WARN] eval.sh received early-stop args but evaluation does not use them. Ignored.\033[0m"
fi

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