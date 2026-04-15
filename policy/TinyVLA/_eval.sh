#!/bin/bash
set -euo pipefail
export PYTHONNOUSERSITE=1

# Single-task evaluation entrypoint for TinyVLA.
#
# Usage modes:
#
# 1. Legacy mode (requires _ev_cfg/<cfg_name>.yaml):
#    bash _eval.sh <cfg_name>
#    bash _eval.sh --config <cfg_name>
#
# 2. Task-id mode (uses ev_tasks.yaml):
#    bash _eval.sh --task-id <task_id>
#
# 3. Direct mode (specify output dir and task directly):
#    bash _eval.sh --output-dir <path> --task-name <name> --task-config <config>
#    bash _eval.sh --output-dir policy/TinyVLA/tinyvla_ckpt/tinyvla-task/demo_clean-100 \
#        --task-name move_pillbottle_pad --task-config demo_clean
#
# Optional arguments:
#    --gpu-id <id>       GPU ID (default: 0)
#    --seed <n>          Random seed (default: 0)
#    --test-num <n>      Number of test rollouts (default: 50)
#    --end-reset-to-init <true|false>
#
# Notes:
# - This script runs from TinyVLA/ directory (it cd's to its own folder).
# - For batch evaluation of all tasks, use: bash __flow.sh

cd "$(dirname "$0")"
python3 ./_ev_wrapper.py "$@"
