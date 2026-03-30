#!/usr/bin/env bash
# Thin wrapper: logic lives in __flow.py (mirrors policy/ACT/__flow.sh).
#
# Other env: MS_TOKEN, DP_FLOW_GPU, DP_FLOW_DRY_RUN=1
#
# PAIRS: one line per pair, "pos_task,neg_task". Run order matches _tr_cfg/_ev_cfg flow_joint_<pos>.yaml.
PAIRS=(
    "move_pillbottle_pad,unmove_pillbottle_pad"
    "stack_bowls_three,unstack_bowls_three"
    "stack_blocks_three,unstack_blocks_three"
    "hanging_mug,unhanging_mug"
)

set -euo pipefail
cd "$(dirname "$0")"
PAIRS_STR=$(printf '%s\n' "${PAIRS[@]}")
exec python3 __flow.py --pairs "$PAIRS_STR" "$@"
