#!/usr/bin/env bash
# Thin wrapper: logic lives in __flow.py (easier to maintain).
#
# Other env: MS_TOKEN, ACT_FLOW_GPU, ACT_FLOW_DRY_RUN=1
#
# PAIRS is a shell-only list (not exported): one line per pair, "pos_task,neg_task".
# Line order is run order. Must match _tr_cfg/_ev_cfg flow_joint_<pos>.yaml in this tree.
PAIRS=(
    "move_pillbottle_pad,unmove_pillbottle_pad"
    "stack_bowls_three,unstack_bowls_three"
    "stack_blocks_three,unstack_blocks_three"
    "hanging_mug,unhanging_mug"
)

set -euo pipefail
cd "$(dirname "$0")"
exec python3 __flow.py --pairs "$PAIRS" "$@"
