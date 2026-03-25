#!/bin/bash
export PYTHONNOUSERSITE=1

# Deprecated:
#   bash train.sh --config <name>
#
# Use:
#   bash _train.sh <name>
if [[ "${1:-}" == "--config" ]]; then
    echo -e "\033[31m[ERROR] Deprecated: bash train.sh --config <name>\033[0m"
    echo -e "\033[33mUse: bash _train.sh <name>\033[0m"
    exit 2
fi

task_name=${1}
task_config=${2}
expert_data_num=${3}
seed=${4}
gpu_id=${5}
early_stop_patience_evals=${6:-}
early_stop_rel_tol=${7:-}
eval_steps_for_early_stop=${8:-}

DEBUG=False
save_ckpt=True

export CUDA_VISIBLE_DEVICES=${gpu_id}

python3 imitate_episodes.py \
    --task_name sim-${task_name}-${task_config}-${expert_data_num} \
    --ckpt_dir ./act_ckpt/act-${task_name}/${task_config}-${expert_data_num} \
    --policy_class ACT \
    --kl_weight 10 \
    --chunk_size 50 \
    --hidden_dim 512 \
    --batch_size 8 \
    --dim_feedforward 3200 \
    --num_epochs 6000 \
    --lr 1e-5 \
    --save_freq 2000 \
    --state_dim 14 \
    --seed ${seed} \
    ${early_stop_patience_evals:+--early_stop_patience_evals ${early_stop_patience_evals}} \
    ${early_stop_rel_tol:+--early_stop_rel_tol ${early_stop_rel_tol}} \
    ${eval_steps_for_early_stop:+--eval_steps_for_early_stop ${eval_steps_for_early_stop}}
