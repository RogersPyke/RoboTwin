#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ACT Train wrapper - Refactored to use BaseTrWrapper.

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
from wrapper_base.config_util import maybe_add_arg


class ACTTrWrapper(BaseTrWrapper):
    """
    ACT-specific training wrapper.

    Model-specific features:
    - Uses SIM_TASK_CONFIGS.json for dataset management
    - Supports multi-task training with symlink-based dataset merging
    - Supports offline ResNet weights via TORCH_HOME
    - Cleans up symlink-only directories after training
    """

    MODEL_NAME = "ACT"

    def _parse_single_task(
        self, task: Dict[str, Any], task_id: str, robotwin_root: Path
    ) -> Tuple[List[str], List[str], List[int]]:
        """
        @input: [dict, task config], [str, task_id], [Path, robotwin root]
        @output: [tuple, (task_names, task_configs, expert_counts)]
        @scenario: [Parse ACT-specific single-task data_folder format]

        ACT data_folder format:
        - Parent dir starting with 'sim-' indicates task_name
        - Folder name format: {task_config}-{expert_num}
        """
        data_folder = Path(task["data_folder"])
        if not data_folder.is_absolute():
            data_folder = robotwin_root / data_folder

        folder_name = data_folder.name
        parent_name = data_folder.parent.name

        if parent_name.startswith("sim-"):
            task_names = [parent_name[4:]]
        else:
            task_names = [task.get("task_name", task_id)]

        parts = folder_name.rsplit("-", 1)
        if len(parts) == 2 and parts[1].isdigit():
            task_configs = [parts[0]]
            expert_counts = [int(parts[1])]
        else:
            task_configs = [task.get("task_config", "demo_clean")]
            expert_counts = [task.get("expert_num", 100)]

        return task_names, task_configs, expert_counts

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
        @output: [tuple, (combined_key, dataset_info dict)]
        @scenario: [Prepare ACT dataset - validate and optionally merge via symlinks]
        """
        combined_task_slug = "__".join(task_names)
        combined_config_slug = "__".join(task_configs)
        combined_total_episodes = int(sum(expert_counts))
        combined_key = f"sim-{combined_task_slug}-{combined_config_slug}-{combined_total_episodes}"
        combined_dataset_dir = f"./processed_data/sim-{combined_task_slug}/{combined_config_slug}-{combined_total_episodes}"

        self.logger.info("Combined key: %s", combined_key)

        # Load SIM_TASK_CONFIGS.json
        sim_cfg_path = "./SIM_TASK_CONFIGS.json"
        if not os.path.isfile(sim_cfg_path):
            raise FileNotFoundError(
                f"Missing {sim_cfg_path}. Please run process_data.sh first."
            )
        with open(sim_cfg_path, "r") as f:
            sim_task_configs = json.load(f)

        # Validate and collect episode info
        episode_len = None
        camera_names = None
        offsets = []
        offset = 0
        for cnt in expert_counts:
            offsets.append(offset)
            offset += cnt

        for i in range(len(task_names)):
            sub_key = f"sim-{task_names[i]}-{task_configs[i]}-{expert_counts[i]}"
            if sub_key not in sim_task_configs:
                raise KeyError(
                    f"Missing SIM_TASK_CONFIGS entry for {sub_key}. Run process_data.sh first."
                )
            sub_entry = sim_task_configs[sub_key]
            if episode_len is None:
                episode_len = sub_entry["episode_len"]
                camera_names = sub_entry["camera_names"]
            else:
                if (
                    sub_entry["episode_len"] != episode_len
                    or sub_entry["camera_names"] != camera_names
                ):
                    raise ValueError(
                        "All subtasks must share episode_len and camera_names."
                    )
            src_dir = sub_entry["dataset_dir"]
            for j in range(expert_counts[i]):
                src_ep = os.path.join(src_dir, f"episode_{j}.hdf5")
                if not os.path.isfile(src_ep):
                    raise FileNotFoundError(f"Missing episode file: {src_ep}")

        # Handle single-task vs multi-task
        if len(task_names) == 1:
            existing_one = sim_task_configs.get(combined_key)
            if existing_one is None:
                raise KeyError(
                    f"Missing SIM_TASK_CONFIGS entry for {combined_key}. Run process_data.sh first."
                )
            self.logger.info(
                "Single-task mode: using existing dataset_dir=%s", combined_dataset_dir
            )
        else:
            # Multi-task: create symlinks
            os.makedirs(combined_dataset_dir, exist_ok=True)
            for i in range(len(task_names)):
                sub_key = f"sim-{task_names[i]}-{task_configs[i]}-{expert_counts[i]}"
                src_dir = sim_task_configs[sub_key]["dataset_dir"]
                base = offsets[i]
                for j in range(expert_counts[i]):
                    dst_idx = base + j
                    dst_ep = os.path.join(combined_dataset_dir, f"episode_{dst_idx}.hdf5")
                    src_ep = os.path.join(src_dir, f"episode_{j}.hdf5")
                    if os.path.lexists(dst_ep):
                        if os.path.islink(dst_ep):
                            continue
                        raise FileExistsError(
                            f"Destination exists and is not a symlink: {dst_ep}"
                        )
                    rel_src_ep = os.path.relpath(src_ep, start=os.path.dirname(dst_ep))
                    os.symlink(rel_src_ep, dst_ep)

            # Update SIM_TASK_CONFIGS.json
            sim_task_configs[combined_key] = {
                "dataset_dir": combined_dataset_dir,
                "num_episodes": combined_total_episodes,
                "episode_len": episode_len,
                "camera_names": camera_names,
            }
            with open(sim_cfg_path, "w") as f:
                json.dump(sim_task_configs, f, indent=4)

        dataset_info = {
            "combined_task_slug": combined_task_slug,
            "combined_config_slug": combined_config_slug,
            "combined_total_episodes": combined_total_episodes,
            "combined_dataset_dir": combined_dataset_dir,
        }
        return combined_key, dataset_info

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
        @output: [list, command tokens for imitate_episodes.py]
        @scenario: [Build ACT training command]
        """
        # Extract ACT-specific parameters
        train_num_epochs = int(params.get("num_epochs", 6000))
        train_save_freq = int(params.get("save_freq", 2000))
        train_batch_size = int(params.get("batch_size", 32))
        train_lr = float(params.get("lr", 1.0e-5))
        train_state_dim = int(params.get("state_dim", 14))
        act_kl_weight = int(params.get("kl_weight", 10))
        act_chunk_size = int(params.get("chunk_size", 50))
        act_hidden_dim = int(params.get("hidden_dim", 512))
        act_dim_feedforward = int(params.get("dim_feedforward", 3200))

        # Early stop parameters
        early_stop = cfg.get("early_stop", {})
        early_stop_enabled = bool(early_stop.get("enabled", True))
        early_stop_patience_evals = (
            early_stop.get("patience_evals") if early_stop_enabled else None
        )
        early_stop_rel_tol = early_stop.get("rel_tol") if early_stop_enabled else None
        eval_steps_for_early_stop = (
            early_stop.get("eval_steps") if early_stop_enabled else None
        )

        # Limits
        limits = cfg.get("limits", {})
        max_tr_steps = limits.get("max_tr_steps")
        save_interval = limits.get("save_interval")

        cmd = [
            sys.executable,
            "imitate_episodes.py",
            "--task_name", dataset_info["dataset_path"],
            "--ckpt_dir", ckpt_dir,
            "--policy_class", "ACT",
            "--kl_weight", str(act_kl_weight),
            "--chunk_size", str(act_chunk_size),
            "--hidden_dim", str(act_hidden_dim),
            "--batch_size", str(train_batch_size),
            "--dim_feedforward", str(act_dim_feedforward),
            "--num_epochs", str(train_num_epochs),
            "--lr", str(train_lr),
            "--save_freq", str(train_save_freq),
            "--state_dim", str(train_state_dim),
            "--seed", str(seed),
        ]
        maybe_add_arg(cmd, "--early_stop_rel_tol", early_stop_rel_tol)
        maybe_add_arg(cmd, "--early_stop_patience_evals", early_stop_patience_evals)
        maybe_add_arg(cmd, "--eval_steps_for_early_stop", eval_steps_for_early_stop)
        maybe_add_arg(cmd, "--max_tr_steps", max_tr_steps)
        maybe_add_arg(cmd, "--save_interval", save_interval)

        return cmd

    def _get_checkpoint_dir(self, task_id: str) -> str:
        """
        @input: [str, task_id]
        @output: [str, checkpoint directory path]
        @scenario: [Return ACT checkpoint directory]
        """
        return f"./act_ckpt/{task_id}"

    def setup_env(self, gpu_id: int) -> dict:
        """
        @input: [int, GPU ID]
        @output: [dict, environment variables for subprocess]
        @scenario: [Setup environment with offline ResNet weights if available]
        """
        env = super().setup_env(gpu_id)

        # Check for offline ResNet weights
        offline_torch_home = os.path.join(self.policy_dir, "torch_offline")
        offline_resnet = os.path.join(
            offline_torch_home, "hub", "checkpoints", "resnet18-f37072fd.pth"
        )
        if os.path.isfile(offline_resnet):
            env["TORCH_HOME"] = offline_torch_home
            self.logger.info(
                "Offline ResNet18 weights found; TORCH_HOME=%s", offline_torch_home
            )

        return env

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
        @scenario: [Cleanup symlink-only dataset directories after training]
        """
        combined_task_slug = dataset_info.get("combined_task_slug", "")
        if not combined_task_slug:
            return

        sim_task_root_dir = os.path.join("./processed_data", f"sim-{combined_task_slug}")
        if os.path.isdir(sim_task_root_dir) and self._has_only_symlink_files(sim_task_root_dir):
            shutil.rmtree(sim_task_root_dir)
            self.logger.info("Removed symlink-only dataset directory: %s", sim_task_root_dir)

    def _has_only_symlink_files(self, path: str) -> bool:
        """Check if directory contains only symlink files (no real data)."""
        if not os.path.isdir(path):
            return False
        for root, dir_names, file_names in os.walk(path, followlinks=False):
            for file_name in file_names:
                file_path = os.path.join(root, file_name)
                if not os.path.islink(file_path):
                    return False
            for dir_name in dir_names:
                dir_path = os.path.join(root, dir_name)
                if os.path.islink(dir_path):
                    continue
        return True

    def write_manifest(
        self, ckpt_dir: str, task_id: str, task_slug: str, config_slug: str, total_episodes: int
    ) -> str:
        """
        @input: [various parameters]
        @output: [str, manifest file path]
        @scenario: [Write ACT-specific training manifest]
        """
        os.makedirs(ckpt_dir, exist_ok=True)
        manifest_path = os.path.join(ckpt_dir, "training_run_manifest.txt")
        with open(manifest_path, "w", encoding="ascii") as mf:
            mf.write(f"task_id={task_id}\n")
            mf.write(f"act_policy_dir={self.policy_dir}\n")
            mf.write(f"combined_task_slug={task_slug}\n")
            mf.write(f"combined_config_slug={config_slug}\n")
            mf.write(f"combined_total_episodes={total_episodes}\n")
        self.logger.info("Saved training manifest to %s", manifest_path)
        return manifest_path


def main(argv: list) -> int:
    """CLI entry point."""
    policy_dir = os.path.dirname(os.path.abspath(__file__))
    return ACTTrWrapper.main(argv, policy_dir=policy_dir)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
