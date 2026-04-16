#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flow scheduler for DP train/eval pipeline.

Usage:
    cd policy/DP
    bash __flow.sh
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR.parent.parent / "policy_util"))

from flow_util.unified_flow import BaseFlowScheduler


class DPFlowScheduler(BaseFlowScheduler):
    MODEL_NAME = "DP"

    def get_process_data_cmd(
        self, task_name: str, task_config: str, expert_num: str
    ) -> list:
        return ["bash", "process_data.sh", task_name, task_config, expert_num]


if __name__ == "__main__":
    import os

    os.chdir(BASE_DIR)
    scheduler = DPFlowScheduler(BASE_DIR)
    raise SystemExit(scheduler.run())
