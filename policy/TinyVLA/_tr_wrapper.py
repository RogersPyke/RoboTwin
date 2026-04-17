#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TinyVLA Train wrapper - Refactored to use BaseTrWrapper.

Usage:
    python3 _tr_wrapper.py --task-id <id> --yaml <config.yaml> [--gpu-id N] [--seed N]
"""

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add policy_util to path for imports
_POLICY_UTIL_ROOT = Path(__file__).resolve().parent.parent.parent / "policy_util"
if str(_POLICY_UTIL_ROOT) not in sys.path:
    sys.path.insert(0, str(_POLICY_UTIL_ROOT))

from wrapper_base import BaseTrWrapper
from wrapper_base.config_util import build_argv_from_args_dict


class TinyVLATrWrapper(BaseTrWrapper):
    """
    TinyVLA-specific training wrapper.

    Model-specific features:
    - Uses HuggingFace Trainer
    - Supports DeepSpeed multi-GPU training
    - Uses dataset_dirs list (multiple data directories)
    - Checkpoint format is directories (checkpoint-N)
    - Uses joint_task_spec.json for task configuration
    """

    MODEL_NAME = "TinyVLA"

    def _prepare_dataset(
        self,
        cfg: Dict[str, Any],
        task: Dict[str, Any],
        task_id: str,
        task_names: List[str],
        task_configs: List[str],
        expert_counts: List[int],
    ) -> Tuple[str, Dict[str, Any]]:
        """
        @input: [various config and task parameters]
        @output: [tuple, (dataset_dirs_json, dataset_info dict)]
        @scenario: [Prepare TinyVLA dataset - resolve data directories]
        """
        robotwin_root = Path(self.policy_dir).parent.parent
        dataset_dirs: List[str] = []

        if "data_folder" in task:
            data_folder = Path(task["data_folder"])
            if not data_folder.is_absolute():
                data_folder = robotwin_root / data_folder
            dataset_dirs.append(str(data_folder))
        elif "data_sources" in task:
            for src in task["data_sources"]:
                data_folder = Path(src["data_folder"])
                if not data_folder.is_absolute():
                    data_folder = robotwin_root / data_folder
                dataset_dirs.append(str(data_folder))
        else:
            raise ValueError(f"Task {task_id} must have data_folder or data_sources")

        combined_task_slug = "__".join(task_names)
        combined_config_slug = "__".join(task_configs)
        combined_total_episodes = int(sum(expert_counts))

        # Create joint_task_spec
        joint_task_spec = {
            "task_name": combined_task_slug,
            "dataset_dir": dataset_dirs,
            "camera_names": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
            "episode_len": 0,
        }

        dataset_info = {
            "task_slug": combined_task_slug,
            "config_slug": combined_config_slug,
            "total_episodes": combined_total_episodes,
            "dataset_dirs": dataset_dirs,
            "joint_task_spec": joint_task_spec,
        }
        return json.dumps(dataset_dirs), dataset_info

    def _build_train_command(
        self,
        cfg: Dict[str, Any],
        task: Dict[str, Any],
        params: Dict[str, Any],
        task_id: str,
        task_names: List[str],
        task_configs: List[str],
        expert_counts: List[int],
        dataset_info: Dict[str, Any],
        ckpt_dir: str,
        gpu_id: int,
        seed: int,
    ) -> List[str]:
        """
        @input: [various config and task parameters]
        @output: [list, command tokens for train_vla.py]
        @scenario: [Build TinyVLA training command]
        """
        combined_task_slug = dataset_info["task_slug"]
        output_dir = os.path.join(self.policy_dir, "tinyvla_ckpt", task_id)

        train_args: Dict[str, Any] = dict(params)
        train_args["task_name"] = combined_task_slug
        train_args["output_dir"] = output_dir
        train_args.setdefault("logging_dir", os.path.join(output_dir, "log"))
        train_args["joint_task_spec"] = json.dumps(dataset_info["joint_task_spec"])

        # Early stop parameters
        early_stop = cfg.get("early_stop", {})
        early_stop_enabled = bool(early_stop.get("enabled", True))
        if early_stop_enabled:
            if early_stop.get("eval_steps") is not None:
                train_args["eval_steps_for_early_stop"] = early_stop["eval_steps"]
            if early_stop.get("patience_evals") is not None:
                train_args["early_stop_patience_evals"] = early_stop["patience_evals"]
            if early_stop.get("rel_tol") is not None:
                train_args["early_stop_rel_tol"] = early_stop["rel_tol"]
        else:
            train_args["early_stop_patience_evals"] = 0
            train_args["early_stop_rel_tol"] = 0.0

        # Limits
        limits = cfg.get("limits", {})
        if limits.get("max_tr_steps"):
            train_args["max_steps"] = limits["max_tr_steps"]
        if limits.get("save_interval"):
            train_args["save_steps"] = limits["save_interval"]
            train_args["save_strategy"] = "steps"

        # DeepSpeed
        deepspeed_cfg = cfg.get("deepspeed", {})
        use_deepspeed = deepspeed_cfg.get("enabled", False)

        if use_deepspeed:
            num_gpus = int(deepspeed_cfg.get("num_gpus", 1))
            master_port = int(deepspeed_cfg.get("master_port", 29604))
            zero2_json = str(deepspeed_cfg.get("zero2_json", "scripts/zero2.json"))
            train_args["deepspeed"] = zero2_json
            self._use_deepspeed = True
            self._num_gpus = num_gpus
            self._master_port = master_port
        else:
            self._use_deepspeed = False

        # Store output_dir for manifest
        self._output_dir = output_dir

        # Build argv
        argv_tokens = build_argv_from_args_dict(train_args, override_seed=seed)

        if use_deepspeed:
            cmd = [
                "deepspeed",
                "--master_port", str(self._master_port),
                f"--num_gpus={self._num_gpus}",
                "--num_nodes=1",
                "./train_vla.py",
            ] + argv_tokens
        else:
            cmd = [sys.executable, "./train_vla.py"] + argv_tokens

        return cmd

    def _get_checkpoint_dir(self, task_id: str) -> str:
        """
        @input: [str, task_id]
        @output: [str, checkpoint directory path]
        @scenario: [Return TinyVLA checkpoint directory]
        """
        return os.path.join(self.policy_dir, "tinyvla_ckpt", task_id)

    def _package_checkpoints(self, ckpt_dir: str) -> None:
        """
        @input: [str, checkpoint directory]
        @output: [None]
        @scenario: [Package TinyVLA checkpoints - directories not files]
        """
        if not os.path.isdir(ckpt_dir):
            return

        candidates = []
        for name in sorted(os.listdir(ckpt_dir)):
            path = os.path.join(ckpt_dir, name)
            if name.startswith("checkpoint-") and os.path.isdir(path):
                candidates.append((name, path))
            elif name == "policy_best" and os.path.isdir(path):
                candidates.append((name, path))

        if not candidates:
            return

        bundle_root = os.path.join(ckpt_dir, "step_packages")
        os.makedirs(bundle_root, exist_ok=True)
        passthrough_files = [
            "training_run_manifest.txt",
            "steps.txt",
            "joint_task_spec.json",
        ]

        for name, src_path in candidates:
            step_id = self._extract_step_from_tinyvla_name(name)
            if step_id >= 0:
                folder_name = f"step_{step_id}"
            elif name == "policy_best":
                folder_name = "step_best"
            else:
                folder_name = f"step_misc_{name.replace('/', '_')}"

            dst_dir = os.path.join(bundle_root, folder_name)
            os.makedirs(dst_dir, exist_ok=True)

            # TinyVLA uses directories, copy tree
            model_dir = os.path.join(dst_dir, "model")
            shutil.copytree(src_path, model_dir, dirs_exist_ok=True)

            # Copy passthrough files
            for keep in passthrough_files:
                src_keep = os.path.join(ckpt_dir, keep)
                if os.path.isfile(src_keep):
                    shutil.copy2(src_keep, os.path.join(dst_dir, keep))

        self.logger.info("Packaged %d TinyVLA checkpoints to %s", len(candidates), bundle_root)

    def _extract_step_from_tinyvla_name(self, name: str) -> int:
        """Extract step from TinyVLA checkpoint directory name."""
        import re
        m = re.search(r"checkpoint-(\d+)$", name)
        if m:
            return int(m.group(1))
        m = re.search(r"step_(\d+)$", name)
        if m:
            return int(m.group(1))
        return -1

    def _cleanup_after_training(
        self,
        cfg: Dict[str, Any],
        task: Dict[str, Any],
        dataset_info: Dict[str, Any],
        success: bool,
    ) -> None:
        """
        @input: [various parameters], [bool, training success]
        @output: [None]
        @scenario: [Cleanup manifest if training failed]
        """
        if not success:
            manifest_path = os.path.join(
                self._get_checkpoint_dir(dataset_info.get("task_id", "")),
                "training_run_manifest.txt"
            )
            if os.path.isfile(manifest_path):
                os.remove(manifest_path)
                self.logger.info("Removed manifest after failed training: %s", manifest_path)

    def write_manifest(
        self, ckpt_dir: str, task_id: str, task_slug: str, config_slug: str, total_episodes: int
    ) -> str:
        """
        @input: [various parameters]
        @output: [str, manifest file path]
        @scenario: [Write TinyVLA-specific training manifest]
        """
        os.makedirs(ckpt_dir, exist_ok=True)
        manifest_path = os.path.join(ckpt_dir, "training_run_manifest.txt")
        with open(manifest_path, "w", encoding="ascii") as mf:
            mf.write(f"task_id={task_id}\n")
            mf.write(f"tinyvla_policy_dir={self.policy_dir}\n")
            mf.write(f"combined_task_slug={task_slug}\n")
            mf.write(f"combined_config_slug={config_slug}\n")
            mf.write(f"combined_total_episodes={total_episodes}\n")
        self.logger.info("Saved training manifest to %s", manifest_path)
        return manifest_path


def main(argv: list) -> int:
    """CLI entry point."""
    policy_dir = os.path.dirname(os.path.abspath(__file__))
    return TinyVLATrWrapper.main(argv, policy_dir=policy_dir)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
