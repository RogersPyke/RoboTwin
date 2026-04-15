#!/usr/bin/env python3

"""
TinyVLA Eval-only Flow Scheduler (Batch Evaluation)

============================================================
Usage:
    cd policy/TinyVLA && bash __flow.sh
    # or directly:
    python3 __flow.py

Description:
    Runs all evaluation tasks defined in _ev_cfg/ev_tasks.yaml.
    Tasks run in parallel according to gpu_parallel configuration.

Configuration (_ev_cfg/ev_tasks.yaml):
    gpu_parallel: [0]           # GPU IDs for parallel execution
    model_defaults:
      test_num: 50
      end_reset_to_init: true
      seed: 0
    ev_tasks:
      - task_id: move_pillbottle_pad
        output_dir: policy/TinyVLA/tinyvla_ckpt/tinyvla-move_pillbottle_pad/demo_clean-100
        eval_tasks:
          - [move_pillbottle_pad, demo_clean, 100]

Runtime precedence:
    CLI args > FLOW env > YAML config defaults

For single-task evaluation:
    bash _eval.sh --task-id <task_id>
    bash _eval.sh --output-dir <path> --task-name <name> --task-config <config>
============================================================
"""

import atexit
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO
