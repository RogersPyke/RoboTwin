#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DP Eval wrapper - Refactored to use BaseEvWrapper.

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


class DPEvWrapper(BaseEvWrapper):
    """
    DP-specific evaluation wrapper.

    Model-specific features:
    - Checkpoint directory format: checkpoints/{task_slug}/{config_slug}-{total_episodes}
    - Uses script/eval_policy.py with deploy_policy.yml
    - Supports head_camera_type and checkpoint_num parameters
    """

    MODEL_NAME = "DP"
    ENV_PREFIX = "DP_FLOW"

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
        @scenario: [Return DP checkpoint directory]
        """
        # DP uses different checkpoint path structure
        return f"policy/DP/checkpoints/{task_slug}/{config_slug}-{total_episodes}"

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
        @scenario: [Build DP evaluation command]
        """
        end_reset = self._resolve_eval_end_reset_to_init(cfg)

        # DP-specific parameters
        checkpoint_num = int(cfg.get("CHECKPOINT_NUM", 600))
        head_camera_type = str(cfg.get("EVAL_HEAD_CAMERA_TYPE", "D435"))
        checkpoint_expert_data_num = int(cfg.get("CHECKPOINT_EXPERT_DATA_NUM", total_episodes))

        # Get expert_num from eval task (for multi-task eval)
        eval_rows = self._parse_task_rows(cfg, "EVAL_TASKS")
        # Find the matching eval task
        eval_expert_num = total_episodes
        for en, ec, ee in eval_rows:
            if en == task_name and ec == task_config:
                eval_expert_num = ee
                break

        cmd = [
            sys.executable,
            "script/eval_policy.py",
            "--config", "policy/DP/deploy_policy.yml",
            "--overrides",
            "--task_name", task_name,
            "--task_config", task_config,
            "--ckpt_setting", config_slug,
            "--expert_data_num", str(checkpoint_expert_data_num),
            "--seed", str(runtime["seed"]),
            "--checkpoint_num", str(checkpoint_num),
            "--head_camera_type", head_camera_type,
            "--train_task_name", train_slug,
            "--eval_expert_data_num", str(eval_expert_num),
            "--test_num", str(runtime["test_num"]),
            "--END_RESET_TO_INIT", end_reset,
        ]
        return cmd

    def setup_env(self, gpu_id: str, cfg_name: str) -> dict:
        """
        @input: [str, GPU ID], [str, config name]
        @output: [dict, environment variables for subprocess]
        @scenario: [Setup environment with LD_LIBRARY_PATH for conda]
        """
        env = super().setup_env(gpu_id, cfg_name)

        # Add conda lib to LD_LIBRARY_PATH if available
        conda_prefix = str(env.get("CONDA_PREFIX", "")).strip()
        if conda_prefix:
            conda_lib = os.path.join(conda_prefix, "lib")
            prev_ld = str(env.get("LD_LIBRARY_PATH", ""))
            env["LD_LIBRARY_PATH"] = conda_lib if not prev_ld else f"{conda_lib}:{prev_ld}"

        return env


def main(argv: list) -> int:
    """CLI entry point."""
    return DPEvWrapper.main(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
