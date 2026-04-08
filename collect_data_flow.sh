#!/bin/bash
# CLI entry for script/collect_data_flow.py. Config below; no CLI args.
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

# ---------------------------------------------------------------------------
# Config (edit here)
# ---------------------------------------------------------------------------
# Run order (enforced by script/collect_data_flow.py):
#   (1) TASK order = order in TASK_TO_COLL below.
#   (2) For each CFG, collect all TASKs first, then next CFG.
#   (3) GPU parallel: same order, multiple jobs at a time.
# Task names to collect (comma-separated).
TASK_TO_COLL="grab_roller, handover_block, 
handover_mic, hanging_mug, 
pick_dual_bottles, place_bread_basket, 
place_bread_skillet, place_burger_fries, 
place_cans_plasticbox"
# Task config names to collect (comma-separated).
CFG_TO_COLL="franka_clean"
# GPU IDs for parallel workers: length = number of processes; each value = Vulkan/CUDA GPU ID.
# Example: "0,0" = 2 processes on GPU 0; "0,1" = 2 processes on GPU 0 and 1.
GPU_PARALLEL="0"

SUBPROCESS_PRINT=true
# Data-collection-only end-reset switch: true/false.
# When true, collected trajectories include explicit return-to-init behavior.
END_RESET_TO_INIT=true

export TASK_TO_COLL CFG_TO_COLL GPU_PARALLEL SUBPROCESS_PRINT END_RESET_TO_INIT

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1
exec python3 script/collect_data_flow.py
