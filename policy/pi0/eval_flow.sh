#!/bin/bash

set -euo pipefail

# Usage:
#   bash eval_flow.sh [task_config] [train_config_name] [model_name] [seed_gpu0] [seed_gpu1]
#
# Example:
#   bash eval_flow.sh franka_clean pi0_base_franka_robotwin_lora ckpt 0 0
#
# This script runs 11 tasks on two GPUs in parallel:
# - GPU 0: first 6 tasks
# - GPU 1: last 5 tasks

task_config="${1:-franka_clean}"
train_config_name="${2:-pi0_base_franka_robotwin_lora}"
model_name="${3:-ckpt}"
seed_gpu0="${4:-0}"
seed_gpu1="${5:-0}"

tasks_gpu0=(
  "grab_roller"
  "handover_mic"
  "place_bread_basket"
  "stack_bowls_two"
  "stack_blocks_two"
  "blocks_ranking_rgb"
)

tasks_gpu1=(
  "lift_pot"
  "place_can_basket"
  "place_dual_shoes"
  "put_object_cabinet"
  "scan_object"
)

run_group() {
  local gpu_id="$1"
  local seed="$2"
  shift 2
  local tasks=("$@")

  for task_name in "${tasks[@]}"; do
    echo "[GPU ${gpu_id}] Start task: ${task_name}"
    bash eval.sh "${task_name}" "${task_config}" "${train_config_name}" "${model_name}" "${seed}" "${gpu_id}"
    echo "[GPU ${gpu_id}] Done task: ${task_name}"
  done
}

run_group 0 "${seed_gpu0}" "${tasks_gpu0[@]}" &
pid0=$!

run_group 1 "${seed_gpu1}" "${tasks_gpu1[@]}" &
pid1=$!

wait "${pid0}"
wait "${pid1}"

echo "All 11 tasks evaluation finished."
