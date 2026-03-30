#!/bin/bash
set -euo pipefail
export PYTHONNOUSERSITE=1

# Recommended entrypoint for ACT training (single- or multi-task via YAML config).
#
# Usage:
#   bash _train.sh <cfg_name>
#
# Example:
#   bash _train.sh hanging_mug_pair
#
# Notes:
# - This script must be run inside the ACT/ folder or from anywhere; it cd's to its own directory.
# - The config file must exist at: _tr_cfg/<cfg_name>.yaml

if [[ $# -lt 1 || -z "${1:-}" ]]; then
  echo "Usage: bash _train.sh <cfg_name>"
  echo "Example: bash _train.sh hanging_mug_pair"
  exit 2
fi

cfg_name="$1"

cd "$(dirname "$0")"
python3 ./_tr_wrapper.py "${cfg_name}"

