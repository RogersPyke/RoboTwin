#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TinyVLA Evaluation Wrapper (Single-Task Evaluation)

============================================================
Usage modes:

1. Task-id mode (uses ev_tasks.yaml):
    python3 _ev_wrapper.py --task-id <id> --yaml <shared.yaml> --yaml <model.yaml>
    bash _eval.sh --task-id move_pillbottle_pad

2. Direct mode (specify output dir and task directly):
    python3 _ev_wrapper.py --output-dir <path> --task-name <name> --task-config <config>
    bash _eval.sh --output-dir policy/TinyVLA/tinyvla_ckpt/tinyvla-task/demo_clean-100 \
        --task-name move_pillbottle_pad --task-config demo_clean

3. Legacy mode (requires _ev_cfg/<cfg_name>.yaml):
    python3 _ev_wrapper.py <cfg_name>
    bash _eval.sh flow_single_move_pillbottle_pad

Optional arguments:
    --gpu-id <id>           GPU ID (default: 0)
    --seed <n>              Random seed (default: 0)
    --test-num <n>          Number of test rollouts (default: 50)
    --end-reset-to-init <true|false>

For batch evaluation:
    bash __flow.sh
============================================================
"""

import argparse
import logging
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
