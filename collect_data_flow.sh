#!/bin/bash
# CLI batch entry for collect_data.sh. Config below; no CLI args.
# Usage: ./collect_data_flow.sh

set -euo pipefail

PARENT_PID="${PPID}"
declare -a CHILD_PIDS=()
WATCHDOG_PID=""
SHUTTING_DOWN=0

cleanup_all() {
  if [[ "${SHUTTING_DOWN}" -eq 1 ]]; then
    return
  fi
  SHUTTING_DOWN=1

  if [[ -n "${WATCHDOG_PID}" ]]; then
    kill "${WATCHDOG_PID}" >/dev/null 2>&1 || true
  fi

  for pid in "${CHILD_PIDS[@]:-}"; do
    if kill -0 "${pid}" >/dev/null 2>&1; then
      kill -TERM "${pid}" >/dev/null 2>&1 || true
    fi
  done

  sleep 0.5
  for pid in "${CHILD_PIDS[@]:-}"; do
    if kill -0 "${pid}" >/dev/null 2>&1; then
      kill -KILL "${pid}" >/dev/null 2>&1 || true
    fi
  done
}

on_signal() {
  cleanup_all
  exit 143
}

monitor_parent_alive() {
  while true; do
    if ! kill -0 "${PARENT_PID}" >/dev/null 2>&1; then
      echo "[collect_data_flow] Parent process ${PARENT_PID} is gone. Terminating all child jobs."
      cleanup_all
      exit 0
    fi
    sleep 1
  done
}

trap on_signal SIGINT SIGTERM

# ---------------------------------------------------------------------------
# Config (edit here)
# ---------------------------------------------------------------------------
# Run order:
#   (1) TASK order = order in TASK_TO_COLL below.
#   (2) For each CFG, collect all TASKs first, then next CFG.
#   (3) GPU parallel: same order, multiple jobs at a time.
# Task names to collect (comma-separated).
TASK_TO_COLL="grab_roller,
handover_mic,
place_bread_basket,
stack_bowls_two,
stack_blocks_two,
blocks_ranking_rgb,
lift_pot,
place_can_basket,
place_dual_shoes,
put_object_cabinet,
scan_object,
handover_block,
hanging_mug,
pick_dual_bottles,
place_bread_skillet,
place_burger_fries,
place_cans_plasticbox"

# "grab_roller, handover_block, 
# handover_mic, hanging_mug, 
# pick_dual_bottles, place_bread_basket, 
# place_bread_skillet, place_burger_fries, 
# place_cans_plasticbox"

# Task config names to collect (comma-separated).
CFG_TO_COLL="arx_clean"
# GPU IDs for parallel workers: length = number of processes; each value = Vulkan/CUDA GPU ID.
# Example: "0,0" = 2 processes on GPU 0; "0,1" = 2 processes on GPU 0 and 1.
GPU_PARALLEL="0,1"

SUBPROCESS_PRINT=true
# If one task's seed-fail count exceeds this threshold, skip that task.
FAIL_SEED_SKIP_THRESHOLD=100

export TASK_TO_COLL CFG_TO_COLL GPU_PARALLEL SUBPROCESS_PRINT FAIL_SEED_SKIP_THRESHOLD

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

split_csv_trimmed() {
  local input="$1"
  local -n out_ref="$2"
  out_ref=()
  IFS=',' read -r -a raw_items <<< "$input"
  for raw in "${raw_items[@]}"; do
    local item
    item="$(echo "$raw" | xargs)"
    if [[ -n "$item" ]]; then
      out_ref+=("$item")
    fi
  done
}

declare -a TASKS CFGS GPUS
split_csv_trimmed "${TASK_TO_COLL}" TASKS
split_csv_trimmed "${CFG_TO_COLL}" CFGS
split_csv_trimmed "${GPU_PARALLEL}" GPUS

if [[ ${#TASKS[@]} -eq 0 || ${#CFGS[@]} -eq 0 || ${#GPUS[@]} -eq 0 ]]; then
  echo "[collect_data_flow] Invalid config: TASK_TO_COLL/CFG_TO_COLL/GPU_PARALLEL cannot be empty."
  exit 1
fi

max_parallel=${#GPUS[@]}
running_count=0
next_gpu_idx=0

monitor_parent_alive &
WATCHDOG_PID="$!"

run_one_job() {
  local task_name="$1"
  local task_cfg="$2"
  local gpu_id="$3"
  if [[ "${SUBPROCESS_PRINT}" == "true" ]]; then
    echo "[collect_data_flow] Start task=${task_name}, cfg=${task_cfg}, gpu=${gpu_id}"
  fi
  bash ./collect_data.sh "${task_name}" "${task_cfg}" "${gpu_id}"
  if [[ "${SUBPROCESS_PRINT}" == "true" ]]; then
    echo "[collect_data_flow] Done  task=${task_name}, cfg=${task_cfg}, gpu=${gpu_id}"
  fi
}

# Delegate real scheduling to Python wrapper so fail-seed threshold and dynamic refill
# logic always applies, even when users invoke this shell entry directly.
python3 ./script/collect_data_flow.py &
CHILD_PIDS+=("$!")
running_count=1

while [[ ${running_count} -gt 0 ]]; do
  if ! wait -n; then
    echo "[collect_data_flow] collect_data_flow.py exited non-zero."
  fi
  ((running_count-=1))
done

cleanup_all
echo "[collect_data_flow] All jobs finished."
