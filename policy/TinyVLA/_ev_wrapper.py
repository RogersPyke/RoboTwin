#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TinyVLA Eval wrapper - Refactored to use BaseEvWrapper.

Usage:
    python3 _ev_wrapper.py <cfg_name>
    python3 _ev_wrapper.py --config <cfg_name>   # (legacy)

Config file: _ev_cfg/<cfg_name>.yaml

Supports:
- ACT-aligned YAML with TRAIN_TASKS + EVAL_TASKS
- Legacy single-task YAML with TASK_NAME/TASK_CONFIG/OUTPUT_DIR
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add policy_util to path for imports
_POLICY_UTIL_ROOT = Path(__file__).resolve().parent.parent.parent / "policy_util"
if str(_POLICY_UTIL_ROOT) not in sys.path:
    sys.path.insert(0, str(_POLICY_UTIL_ROOT))

from wrapper_base import BaseEvWrapper


class TinyVLAEvWrapper(BaseEvWrapper):
    """
    TinyVLA-specific evaluation wrapper.

    Model-specific features:
    - Checkpoint directory format: tinyvla_ckpt/tinyvla-{task_slug}/{config_slug}-{total_episodes}
    - Uses script/eval_policy.py with deploy_policy.yml
    - Supports policy_best selection
    - Uses model_path and state_path parameters
    """

    MODEL_NAME = "TinyVLA"
    ENV_PREFIX = "TVLA_FLOW"

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
        @scenario: [Return TinyVLA checkpoint directory]
        """
        return os.path.join(
            self.policy_dir,
            "tinyvla_ckpt",
            f"tinyvla-{task_slug}",
            f"{config_slug}-{total_episodes}",
        )

    def _get_required_config_keys(self) -> List[str]:
        """
        @input: [None]
        @output: [list, required config keys]
        @scenario: [TinyVLA requires additional keys for model configuration]
        """
        return [
            "EVAL_SEED", "EVAL_GPU_ID", "MODEL_BASE",
            "TRAIN_TASKS", "EVAL_TASKS", "END_RESET_TO_INIT"
        ]

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
        @scenario: [Build TinyVLA evaluation command]
        """
        end_reset = self._resolve_eval_end_reset_to_init(cfg)

        # TinyVLA-specific parameters
        model_base = str(cfg.get("MODEL_BASE", ""))
        use_policy_best = bool(cfg.get("USE_POLICY_BEST", False))
        enable_lore = bool(cfg.get("ENABLE_LORE", False))
        instruction_type = cfg.get("INSTRUCTION_TYPE")

        # Resolve model_path and state_path
        model_path = os.path.join(ckpt_dir, "policy_best") if use_policy_best else ckpt_dir
        state_path = os.path.join(ckpt_dir, "dataset_stats.pkl")

        # Build overrides
        overrides: List[str] = []
        self._add_pair(overrides, "task_name", task_name)
        self._add_pair(overrides, "task_config", task_config)
        self._add_pair(overrides, "ckpt_setting", config_slug)
        self._add_pair(overrides, "expert_data_num", total_episodes)
        self._add_pair(overrides, "seed", runtime["seed"])
        self._add_pair(overrides, "model_base", model_base)
        self._add_pair(overrides, "model_path", model_path)
        self._add_pair(overrides, "state_path", state_path)
        self._add_pair(overrides, "enable_lore", enable_lore)
        self._add_pair(overrides, "instruction_type", instruction_type)
        self._add_pair(overrides, "test_num", runtime["test_num"])
        overrides.extend(["--END_RESET_TO_INIT", end_reset])

        cmd = [
            sys.executable,
            "script/eval_policy.py",
            "--config", "policy/TinyVLA/deploy_policy.yml",
            "--overrides",
        ] + overrides

        return cmd

    def _add_pair(self, overrides: List[str], key: str, value: Any) -> None:
        """Add key-value pair to overrides if value is not None."""
        if value is None:
            return
        overrides.extend([f"--{key}", str(value)])

    def run(
        self,
        cfg_name: str,
        gpu_id: Optional[str] = None,
        seed: Optional[int] = None,
        test_num: Optional[int] = None,
    ) -> int:
        """
        @input: [str, config name], [optional overrides]
        @output: [int, exit code (0 for success)]
        @scenario: [Execute evaluation - supports both ACT-style and legacy configs]
        """
        try:
            cfg = self._load_ev_cfg(cfg_name)

            # Check for legacy single-task config
            if "TRAIN_TASKS" not in cfg and "EVAL_TASKS" not in cfg:
                return self._run_legacy_eval(cfg, cfg_name, gpu_id, seed, test_num)

            # Standard multi-task eval
            return super().run(cfg_name, gpu_id, seed, test_num)

        except Exception as exc:
            self.logger.error("Wrapper failed: %s", str(exc))
            import traceback
            self.logger.error("Stack trace:\n%s", traceback.format_exc())
            return 1

    def _run_legacy_eval(
        self,
        cfg: Dict[str, Any],
        cfg_name: str,
        gpu_id: Optional[str],
        seed: Optional[int],
        test_num: Optional[int],
    ) -> int:
        """Handle legacy single-task config format."""
        self._ensure_required_keys(cfg, [
            "EVAL_SEED", "EVAL_GPU_ID", "TASK_NAME", "TASK_CONFIG",
            "MODEL_BASE", "OUTPUT_DIR", "END_RESET_TO_INIT"
        ])

        runtime = self._resolve_runtime(cfg, seed, gpu_id, test_num)

        self.logger.info(
            "Resolved runtime: seed=%s (%s), gpu_id=%s (%s), test_num=%s (%s)",
            runtime["seed"], runtime["seed_source"],
            runtime["gpu_id"], runtime["gpu_source"],
            runtime["test_num"], runtime["test_num_source"],
        )

        task_name = str(cfg["TASK_NAME"]).strip()
        task_config = str(cfg["TASK_CONFIG"]).strip()
        output_dir = str(cfg["OUTPUT_DIR"])

        use_policy_best = bool(cfg.get("USE_POLICY_BEST", False))
        model_path = os.path.join(output_dir, "policy_best") if use_policy_best else output_dir
        state_path = os.path.join(output_dir, "dataset_stats.pkl")

        end_reset = self._resolve_eval_end_reset_to_init(cfg)

        overrides: List[str] = []
        self._add_pair(overrides, "task_name", task_name)
        self._add_pair(overrides, "task_config", task_config)
        self._add_pair(overrides, "ckpt_setting", cfg.get("CKPT_SETTING", "legacy"))
        self._add_pair(overrides, "expert_data_num", cfg.get("EXPERT_DATA_NUM", 0))
        self._add_pair(overrides, "seed", runtime["seed"])
        self._add_pair(overrides, "model_base", cfg.get("MODEL_BASE"))
        self._add_pair(overrides, "model_path", model_path)
        self._add_pair(overrides, "state_path", state_path)
        self._add_pair(overrides, "enable_lore", cfg.get("ENABLE_LORE", False))
        self._add_pair(overrides, "instruction_type", cfg.get("INSTRUCTION_TYPE"))
        self._add_pair(overrides, "test_num", runtime["test_num"])
        overrides.extend(["--END_RESET_TO_INIT", end_reset])

        cmd = [
            sys.executable,
            "script/eval_policy.py",
            "--config", "policy/TinyVLA/deploy_policy.yml",
            "--overrides",
        ] + overrides

        repo_root = os.path.abspath(os.path.join(self.policy_dir, "..", ".."))
        env = self.setup_env(runtime["gpu_id"], cfg_name)

        self.logger.info(
            "Eval: task_name=%s task_config=%s model_path=%s test_num=%s",
            task_name, task_config, model_path, runtime["test_num"]
        )
        self.run_eval(cmd, env, repo_root)

        return 0


def main(argv: list) -> int:
    """CLI entry point."""
    return TinyVLAEvWrapper.main(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
