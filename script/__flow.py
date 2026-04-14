#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Training Flow Entry Point.

@input: [str, model name (ACT/DP/TinyVLA)], [optional CLI args]
@output: [int, exit code]
@scenario: [Parse config, run training scheduler for specified model]

Usage:
    python script/__flow.py ACT
    python script/__flow.py DP --gpu-parallel 0,1
    python script/__flow.py TinyVLA --config path/to/custom.yaml
"""

import argparse
import atexit
import os
import signal
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROBOTWIN_ROOT = SCRIPT_DIR.parent
POLICY_UTIL_ROOT = ROBOTWIN_ROOT / "policy_util"

if str(POLICY_UTIL_ROOT) not in sys.path:
    sys.path.insert(0, str(POLICY_UTIL_ROOT))

from flow_util.tr_cfg_parser import load_tr_config
from flow_util.unified_scheduler import (
    cleanup_all_jobs,
    on_signal,
    run_scheduler,
)


def parse_gpu_parallel(value: str) -> list:
    """
    @input: [str, comma-separated GPU IDs]
    @output: [list[int], list of GPU IDs]
    @scenario: [Parse --gpu-parallel CLI argument]
    """
    return [int(x.strip()) for x in value.split(",")]


def main(argv: list) -> int:
    """
    @input: [list[str], command line arguments]
    @output: [int, exit code]
    @scenario: [Parse CLI, load config, run scheduler]
    """
    parser = argparse.ArgumentParser(
        description="Unified training flow scheduler for ACT/DP/TinyVLA"
    )
    parser.add_argument(
        "model",
        type=str,
        help="Model to train: ACT, DP, or TinyVLA",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom tr_tasks.yaml config file",
    )
    parser.add_argument(
        "--gpu-parallel",
        type=parse_gpu_parallel,
        default=None,
        help="Override GPU parallel slots (comma-separated, e.g., '0,1')",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed",
    )

    args = parser.parse_args(argv[1:])

    model = args.model
    print(f"[flow] Loading config for model: {model}")

    try:
        cfg = load_tr_config(model, config_path=args.config)
    except Exception as e:
        print(f"[flow] Failed to load config: {e}", file=sys.stderr)
        return 1

    if args.gpu_parallel is not None:
        cfg["gpu_parallel"] = args.gpu_parallel
        print(f"[flow] Overriding gpu_parallel: {args.gpu_parallel}")

    if args.seed is not None:
        cfg["seed"] = args.seed
        print(f"[flow] Overriding seed: {args.seed}")

    print(f"[flow] Config loaded: {len(cfg['tr_tasks'])} tasks")
    print(f"[flow] GPU slots: {cfg['gpu_parallel']}")

    return run_scheduler(cfg)


if __name__ == "__main__":
    atexit.register(cleanup_all_jobs)
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, on_signal)

    os.chdir(ROBOTWIN_ROOT)
    raise SystemExit(main(sys.argv))
