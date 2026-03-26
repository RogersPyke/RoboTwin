#!/bin/bash
set -euo pipefail
export PYTHONNOUSERSITE=1

# Recommended entrypoint for TinyVLA evaluation (config under _ev_cfg/<cfg_name>.yaml).
#
# Usage:
#   bash _eval.sh <cfg_name>
#
# Notes:
# - This script runs from TinyVLA/ directory (it cd's to its own folder).
# - The config file must exist at: _ev_cfg/<cfg_name>.yaml

if [[ $# -lt 1 || -z "${1:-}" ]]; then
  echo "Usage: bash _eval.sh <cfg_name>"
  echo "Example: bash _eval.sh tinyvla_single_task_example"
  exit 2
fi

cfg_name="$1"

cd "$(dirname "$0")"
python3 ./_ev_wrapper.py "${cfg_name}"

