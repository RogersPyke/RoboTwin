#!/bin/bash
export PYTHONNOUSERSITE=1
# ====== Wrapper Modification ======
# Multi-task: must use --config <name>; config file under _tr_cfg/
# On start, _tr_wrapper.py copies that YAML plus training_run_manifest.txt into ckpt_dir
# so each checkpoint tree records which config and paths were used (not only folder names).
if [[ "$1" == "--config" && -n "${2:-}" ]]; then
    cd "$(dirname "$0")"
    python3 ./_tr_wrapper.py --config "$2"
    exit 0
fi
# ==================

task_name=${1}
task_config=${2}
expert_data_num=${3}
seed=${4}
gpu_id=${5}

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
    --seed ${seed}
