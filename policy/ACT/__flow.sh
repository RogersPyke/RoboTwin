#!/usr/bin/env bash

# ============================================================
# ACT Batch Evaluation Entry Point
# ============================================================
# 
# Usage:
#   cd policy/ACT && bash __flow.sh
#
# Description:
#   Runs all evaluation tasks defined in _ev_cfg/ev_tasks.yaml
#   in parallel according to gpu_parallel configuration.
#
# Configuration:
#   Edit _ev_cfg/ev_tasks.yaml to:
#   - Add/remove evaluation tasks
#   - Change GPU allocation (gpu_parallel)
#   - Modify test_num, seed, etc.
#
# For single-task evaluation, use:
#   bash _eval.sh --task-id <task_id>
#   bash _eval.sh --ckpt-dir <path> --task-name <name> --task-config <config>
#
# ============================================================

set -euo pipefail
cd "$(dirname "$0")"
exec python3 __flow.py
