#!/usr/bin/env bash

set -euo pipefail
export PYTHONNOUSERSITE=1
cd "$(dirname "$0")"

TASK_CONFIG="demo_clean"
EXPERT_NUM="100"

TASKS=(
  "move_pillbottle_pad"
  "unmove_pillbottle_pad"
  "stack_bowls_three"
  "unstack_bowls_three"
  "stack_blocks_three"
  "unstack_blocks_three"
  "hanging_mug"
  "unhanging_mug"
)

STEMS=(
  "flow_single_move_pillbottle_pad"
  "flow_single_unmove_pillbottle_pad"
  "flow_joint_move_pillbottle_pad"
  "flow_single_stack_bowls_three"
  "flow_single_unstack_bowls_three"
  "flow_joint_stack_bowls_three"
  "flow_single_stack_blocks_three"
  "flow_single_unstack_blocks_three"
  "flow_joint_stack_blocks_three"
  "flow_single_hanging_mug"
  "flow_single_unhanging_mug"
  "flow_joint_hanging_mug"
)

for stem in "${STEMS[@]}"; do
  if [[ ! -f "_tr_cfg/${stem}.yaml" ]]; then
    echo "[ERROR] missing _tr_cfg/${stem}.yaml" >&2
    exit 1
  fi
  if [[ ! -f "_ev_cfg/${stem}.yaml" ]]; then
    echo "[ERROR] missing _ev_cfg/${stem}.yaml" >&2
    exit 1
  fi
done


for task in "${TASKS[@]}"; do
  bash process_data.sh "$task" "$TASK_CONFIG" "$EXPERT_NUM"
done

for stem in "${STEMS[@]}"; do
  bash _train.sh "$stem"
  bash _eval.sh "$stem"
done
