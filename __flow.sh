#!/usr/bin/env bash
#
# Unified Training Flow Entry Point
#
# Usage:
#   bash __flow.sh ACT
#   bash __flow.sh DP
#   bash __flow.sh TinyVLA
#   bash __flow.sh ACT --gpu-parallel 0,1
#   bash __flow.sh DP --config path/to/custom.yaml
#
# This script delegates to script/__flow.py for actual execution.
#

set -euo pipefail
cd "$(dirname "$0")"

if [[ $# -lt 1 || -z "${1:-}" ]]; then
    echo "Usage: bash __flow.sh <model> [options]"
    echo "  model: ACT, DP, or TinyVLA"
    echo "  options:"
    echo "    --config PATH         Custom config file path"
    echo "    --gpu-parallel IDS    Override GPU slots (e.g., '0,1')"
    echo "    --seed N              Override random seed"
    exit 2
fi

exec python3 script/__flow.py "$@"
