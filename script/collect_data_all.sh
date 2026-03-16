#!/bin/bash
# Run all four data-collection tasks in sequence. Continue to the next if any fails.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR" || exit 1

LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/collect_data_all_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG_FILE") 2>&1
echo "Logging to $LOG_FILE"

GPU_ID="${1:-0}"

run_one() {
  local task_name="$1"
  local task_config="$2"
  echo "===== Collecting: $task_name / $task_config (GPU $GPU_ID) ====="
  if bash ./collect_data.sh "$task_name" "$task_config" "$GPU_ID"; then
    echo "===== OK: $task_name / $task_config ====="
    return 0
  else
    echo "===== FAILED: $task_name / $task_config (continuing) ====="
    return 1
  fi
}

FAILED=0

# run_one unhanging_mug demo_clean || ((FAILED++))
# run_one unhanging_mug demo_randomized || ((FAILED++))
run_one hanging_mug demo_clean || ((FAILED++))
run_one hanging_mug demo_randomized || ((FAILED++))

if [[ $FAILED -gt 0 ]]; then
  echo "Done. $FAILED task(s) failed."
  exit 1
fi
echo "Done. All tasks completed successfully."
exit 0
