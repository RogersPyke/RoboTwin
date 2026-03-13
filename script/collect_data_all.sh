#!/bin/bash
# Run all four data-collection tasks in sequence. Continue to the next if any fails.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR" || exit 1

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

run_one unstack_bowls_three demo_clean || ((FAILED++))
run_one unstack_bowls_three demo_randomized || ((FAILED++))
run_one stack_bowls_three demo_clean || ((FAILED++))
run_one stack_bowls_three demo_randomized || ((FAILED++))

if [[ $FAILED -gt 0 ]]; then
  echo "Done. $FAILED task(s) failed."
  exit 1
fi
echo "Done. All tasks completed successfully."
exit 0
