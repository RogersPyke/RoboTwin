#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ACT Eval wrapper - Refactored to use BaseEvWrapper.

Usage:
    python3 _ev_wrapper.py <cfg_name>
    python3 _ev_wrapper.py --config <cfg_name>   # (legacy)

Config file: _ev_cfg/<cfg_name>.yaml
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add policy_util to path for imports
_POLICY_UTIL_ROOT = Path(__file__).resolve().parent.parent.parent / "policy_util"
if str(_POLICY_UTIL_ROOT) not in sys.path:
    sys.path.insert(0, str(_POLICY_UTIL_ROOT))

from wrapper_base import BaseEvWrapper


class ACTEvWrapper(BaseEvWrapper):
    """
    ACT-specific evaluation wrapper.

    Model-specific features:
    - Checkpoint directory format: act_ckpt/act-{task_slug}/{config_slug}-{total_episodes}
    - Uses script/eval_policy.py with deploy_policy.yml
    - Supports temporal aggregation
    """

    MODEL_NAME = "ACT"
    ENV_PREFIX = "ACT_FLOW"

    def _get_checkpoint_dir(
        self,
        task_slug: str,
        config_slug: str,
        total_episodes: int,
        cfg: Dict[str, Any],
    ) -> str:
        """
        @input: [str, task slug], [str, config slug], [int, total episodes], [dict, config]
        @output: [str, checkpoint directory path]
        @scenario: [Return ACT checkpoint directory]
        """
        return f"policy/ACT/act_ckpt/act-{task_slug}/{config_slug}-{total_episodes}"

    def _build_eval_command(
        self,
        task_name: str,
        task_config: str,
        ckpt_dir: str,
        runtime: Dict[str, Any],
        cfg: Dict[str, Any],
        train_slug: str,
        config_slug: str,
        total_episodes: int,
    ) -> List[str]:
        """
        @input: [various evaluation parameters]
        @output: [list, command tokens for script/eval_policy.py]
        @scenario: [Build ACT evaluation command]
        """
        end_reset = self._resolve_eval_end_reset_to_init(cfg)

        cmd = [
            sys.executable,
            "script/eval_policy.py",
            "--config", "policy/ACT/deploy_policy.yml",
            "--overrides",
            "--task_name", task_name,
            "--task_config", task_config,
            "--ckpt_setting", config_slug,
            "--ckpt_dir", ckpt_dir,
            "--seed", str(runtime["seed"]),
            "--test_num", str(runtime["test_num"]),
            "--temporal_agg", "true",
            "--END_RESET_TO_INIT", end_reset,
        ]
        return cmd


def main(argv: list) -> int:
    """CLI entry point."""
    return ACTEvWrapper.main(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
