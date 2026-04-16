#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flow scheduler for TinyVLA train/eval pipeline.

Usage:
    cd policy/TinyVLA
    bash __flow.sh
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR.parent.parent.parent / "policy_util"))

from flow_util.unified_flow import BaseFlowScheduler


class TinyVLAFlowScheduler(BaseFlowScheduler):
    MODEL_NAME = "TinyVLA"

    def get_process_data_cmd(
        self, task_name: str, task_config: str, expert_num: str
    ) -> list:
        return [sys.executable, "process_data.py", task_name, task_config, expert_num]


if __name__ == "__main__":
    import os

    os.chdir(BASE_DIR)
    scheduler = TinyVLAFlowScheduler(BASE_DIR)
    raise SystemExit(scheduler.run())
