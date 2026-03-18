#!/bin/bash
# CLI entry for script/collect_data_flow.py. Config below; no CLI args.
# Usage: ./collect_data_flow.sh

# ---------------------------------------------------------------------------
# Config (edit here)
# ---------------------------------------------------------------------------
# Run order (enforced by script/collect_data_flow.py):
#   (1) TASK order = order in TASK_TO_COLL below.
#   (2) For each CFG, collect all TASKs first, then next CFG.
#   (3) GPU parallel: same order, multiple jobs at a time.
# Task names to collect (comma-separated).
TASK_TO_COLL="unhanging_mug,hanging_mug,
    unmove_pillbottle_pad,move_pillbottle_pad,
    unstack_bowls_three,stack_bowls_three,
    unstack_blocks_three,stack_blocks_three"
# Task config names to collect (comma-separated).
CFG_TO_COLL="demo_clean,demo_randomized"
# GPU IDs for parallel workers: length = number of processes; each value = Vulkan/CUDA GPU ID.
# Example: "0,0" = 2 processes on GPU 0; "0,1" = 2 processes on GPU 0 and 1.
GPU_PARALLEL="0,0"

SUBPROCESS_PRINT=false

export TASK_TO_COLL CFG_TO_COLL GPU_PARALLEL SUBPROCESS_PRINT

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1
exec python3 script/collect_data_flow.py
