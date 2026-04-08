#!/bin/bash
# CLI entry for script/collect_data_flow.py. Config lives in Python file.
# Usage: ./collect_data_flow.sh
#
# ---------------------------------------------------------------------------
# PERTURBATION (Pert) mode -- same-seed diverse data collection
# ---------------------------------------------------------------------------
# To collect pert data (diverse trajectories) alongside classic data:
#
#   1. Classic collection (run first to generate seed.txt files):
#        TASK_TO_COLL="hanging_mug, stack_blocks_three" CFG_TO_COLL="demo_clean" ...
#
#   2. Pert collection:
#        Run pert tasks directly (they manage their own local seed.txt
#        discovery/resume flow just like classic tasks).
#
#   Supported pert tasks (must pair with their classic counterparts):
#     hanging_mug_pert         <- hanging_mug
#     unhanging_mug_pert       <- unhanging_mug        (conservative mode)
#     stack_blocks_three_pert  <- stack_blocks_three
#     unstack_blocks_three_pert<- unstack_blocks_three  (conservative mode)
#     stack_bowls_three_pert   <- stack_bowls_three
#     unstack_bowls_three_pert <- unstack_bowls_three   (conservative mode)
#     move_pillbottle_pad_pert <- move_pillbottle_pad
#     unmove_pillbottle_pad_pert<- unmove_pillbottle_pad(conservative mode)
#
#   Pert config is in task_config/demo_clean_pert.yml (perturbation: block).
#
# NOTE: Cross-task seed reuse is removed.
# Pert tasks no longer read seed.txt from classic task directories.
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1
exec python3 script/collect_data_flow.py
