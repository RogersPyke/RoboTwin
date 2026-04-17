#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DP Train wrapper - Refactored to use BaseTrWrapper.

Usage:
    python3 _tr_wrapper.py --task-id <id> --yaml <config.yaml> [--gpu-id N] [--seed N]
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import zarr

# Add policy_util to path for imports
_POLICY_UTIL_ROOT = Path(__file__).resolve().parent.parent.parent / "policy_util"
if str(_POLICY_UTIL_ROOT) not in sys.path:
    sys.path.insert(0, str(_POLICY_UTIL_ROOT))

from wrapper_base import BaseTrWrapper
from wrapper_base.checkpoint_util import find_workspace_checkpoint_dir


class DPTrWrapper(BaseTrWrapper):
    """
    DP-specific training wrapper.

    Model-specific features:
    - Uses zarr format for dataset storage
    - Supports multi-task training with zarr concatenation
    - Uses Hydra configuration system
    - Checkpoints saved in data/outputs/{save_name}-{seed}/checkpoints/
    """

    MODEL_NAME = "DP"

    def _parse_single_task(
        self, task: Dict[str, Any], task_id: str, robotwin_root: Path
    ) -> Tuple[List[str], List[str], List[int]]:
        """
        @input: [dict, task config], [str, task_id], [Path, robotwin root]
        @output: [tuple, (task_names, task_configs, expert_counts)]
        @scenario: [Parse DP-specific single-task data_folder format]

        DP data_folder format:
        - Folder name: {task_name}-{task_config}-{expert_num}.zarr
        - Or: {task_config}-{expert_num}.zarr (task_name inferred)
        """
        data_folder = Path(task["data_folder"])
        folder_name = data_folder.name
        if folder_name.endswith(".zarr"):
            folder_name = folder_name[:-5]

        parts = folder_name.rsplit("-", 1)
        if len(parts) == 2 and parts[1].isdigit():
            task_names = [parts[0].rsplit("-", 1)[0] if "-" in parts[0] else parts[0]]
            task_configs = [
                parts[0].rsplit("-", 1)[1] if "-" in parts[0] else "demo_clean"
            ]
            expert_counts = [int(parts[1])]
        else:
            task_names = [task.get("task_name", task_id)]
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
        @output: [tuple, (zarr_path, dataset_info dict)]
        @scenario: [Prepare DP dataset - ensure zarr files and concatenate if multi-task]
        """
        task_slug = "__".join(task_names)
        config_slug = "__".join(task_configs)
        total_episodes = int(sum(expert_counts))

        combined_rel_path = os.path.join(
            "data", f"{task_slug}-{config_slug}-{total_episodes}.zarr"
        )
        combined_abs_path = os.path.join(self.policy_dir, combined_rel_path)

        # Resolve source zarr paths
        rows = list(zip(task_names, task_configs, expert_counts))
        src_paths = self._resolve_src_zarr_paths(rows)

        # Concatenate zarrs
        self._concat_zarrs(src_paths, combined_abs_path, combined_rel_path)

        dataset_info = {
            "task_slug": task_slug,
            "config_slug": config_slug,
            "total_episodes": total_episodes,
            "combined_rel_path": combined_rel_path,
            "combined_abs_path": combined_abs_path,
            "save_name": f"{task_slug}-{config_slug}-{total_episodes}",
        }
        return combined_rel_path, dataset_info

    def _resolve_src_zarr_paths(
        self, rows: List[Tuple[str, str, int]]
    ) -> List[Tuple[str, str, int, str, str]]:
        """Resolve and ensure source zarr paths exist."""
        src_paths = []
        for task_name, task_config, expert_num in rows:
            src_rel_path = self._ensure_single_task_zarr(task_name, task_config, expert_num)
            src_abs_path = os.path.join(self.policy_dir, src_rel_path)
            if not os.path.isdir(src_abs_path):
                raise FileNotFoundError(f"Source zarr not found: {src_rel_path}")
            src_paths.append(
                (task_name, task_config, expert_num, src_rel_path, src_abs_path)
            )
        return src_paths

    def _ensure_single_task_zarr(
        self, task_name: str, task_config: str, expert_num: int
    ) -> str:
        """Ensure single-task zarr exists, build if necessary."""
        rel_path = os.path.join("data", f"{task_name}-{task_config}-{expert_num}.zarr")
        abs_path = os.path.join(self.policy_dir, rel_path)
        if os.path.isdir(abs_path):
            return rel_path
        self.logger.info("Single-task zarr missing, building: %s", rel_path)
        cmd = [sys.executable, "process_data.py", task_name, task_config, str(expert_num)]
        subprocess.run(cmd, check=True, cwd=self.policy_dir, env=os.environ.copy())
        if not os.path.isdir(abs_path):
            raise FileNotFoundError(f"Failed to build expected zarr: {rel_path}")
        return rel_path

    def _concat_zarrs(
        self,
        src_paths: List[Tuple[str, str, int, str, str]],
        combined_abs_path: str,
        combined_rel_path: str,
    ) -> None:
        """Concatenate multiple zarr files into one."""
        src_meta = []
        src_head = []
        src_state = []
        src_action = []

        for task_name, task_config, expert_num, src_rel_path, src_abs_path in src_paths:
            root = zarr.open(src_abs_path, mode="r")
            src_head_arr = root["data"]["head_camera"][:]
            src_state_arr = root["data"]["state"][:]
            src_action_arr = root["data"]["action"][:]
            src_ep_ends = root["meta"]["episode_ends"][:]
            if not (len(src_head_arr) == len(src_state_arr) == len(src_action_arr)):
                raise ValueError(f"Inconsistent data length in {src_rel_path}")
            src_meta.append(
                (task_name, task_config, expert_num, src_rel_path, src_ep_ends.copy())
            )
            src_head.append(src_head_arr)
            src_state.append(src_state_arr)
            src_action.append(src_action_arr)

        all_head = np.concatenate(src_head, axis=0)
        all_state = np.concatenate(src_state, axis=0)
        all_action = np.concatenate(src_action, axis=0)

        episode_ends_list = []
        offset = 0
        for _, _, _, _, src_ep_ends in src_meta:
            episode_ends_list.append(src_ep_ends + offset)
            offset += int(src_ep_ends[-1])
        all_episode_ends = np.concatenate(episode_ends_list, axis=0).astype(np.int64)

        if os.path.isdir(combined_abs_path):
            shutil.rmtree(combined_abs_path)

        compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
        root = zarr.group(combined_abs_path)
        data_group = root.create_group("data")
        meta_group = root.create_group("meta")

        data_group.create_dataset(
            "head_camera",
            data=all_head,
            chunks=(100, *all_head.shape[1:]),
            overwrite=True,
            compressor=compressor,
        )
        data_group.create_dataset(
            "state",
            data=all_state,
            chunks=(100, all_state.shape[1]),
            dtype="float32",
            overwrite=True,
            compressor=compressor,
        )
        data_group.create_dataset(
            "action",
            data=all_action,
            chunks=(100, all_action.shape[1]),
            dtype="float32",
            overwrite=True,
            compressor=compressor,
        )
        meta_group.create_dataset(
            "episode_ends",
            data=all_episode_ends,
            dtype="int64",
            overwrite=True,
            compressor=compressor,
        )
        self.logger.info("Built combined zarr: %s", combined_rel_path)

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
        @output: [list, command tokens for train.py with Hydra overrides]
        @scenario: [Build DP training command]
        """
        action_dim = int(params.get("action_dim", 14))
        head_camera_type = str(params.get("head_camera_type", "D435"))
        batch_size = int(params.get("batch_size", 64))
        num_epochs = int(params.get("num_epochs", 600))
        checkpoint_every = int(params.get("checkpoint_every", 300))
        val_ratio = float(params.get("val_ratio", 0.02))
        learning_rate = float(params.get("lr", 1.0e-4))

        early_stop = cfg.get("early_stop", {})
        early_stop_enabled = bool(early_stop.get("enabled", True))
        early_stop_patience_evals = (
            early_stop.get("patience_evals", 0) if early_stop_enabled else 0
        )
        early_stop_rel_tol = early_stop.get("rel_tol", 0.0) if early_stop_enabled else 0.0
        eval_steps_for_early_stop = (
            early_stop.get("eval_steps", 1) if early_stop_enabled else 1
        )

        limits = cfg.get("limits", {})
        max_tr_steps = limits.get("max_tr_steps")
        save_interval = limits.get("save_interval")

        task_slug = dataset_info["task_slug"]
        config_slug = dataset_info["config_slug"]
        total_episodes = dataset_info["total_episodes"]
        zarr_path = dataset_info["dataset_path"]

        cmd = [
            sys.executable,
            "train.py",
            f"--config-name=robot_dp_{action_dim}.yaml",
            f"task.name={task_slug}",
            f"task.dataset.zarr_path={zarr_path}",
            "training.debug=False",
            f"training.seed={seed}",
            "training.device=cuda:0",
            f"dataloader.batch_size={batch_size}",
            f"val_dataloader.batch_size={batch_size}",
            f"task.dataset.val_ratio={val_ratio}",
            f"optimizer.lr={learning_rate}",
            f"training.num_epochs={num_epochs}",
            f"training.checkpoint_every={checkpoint_every}",
            f"training.early_stop_patience_evals={early_stop_patience_evals}",
            f"training.early_stop_rel_tol={early_stop_rel_tol}",
            f"training.eval_steps_for_early_stop={eval_steps_for_early_stop}",
            f"training.max_tr_steps={max_tr_steps}",
            f"training.save_interval={save_interval}",
            f"setting={config_slug}",
            f"expert_data_num={total_episodes}",
            f"head_camera_type={head_camera_type}",
            f"hydra.run.dir={ckpt_dir}",
            f"hydra.sweep.dir={ckpt_dir}",
            "hydra.sweep.subdir=multirun_${hydra.job.num}",
            f"multi_run.run_dir={ckpt_dir}",
        ]
        return cmd

    def _get_checkpoint_dir(self, task_id: str) -> str:
        """
        @input: [str, task_id]
        @output: [str, checkpoint directory path]
        @scenario: [Return DP checkpoint directory]
        """
        return os.path.join(self.policy_dir, "checkpoints", task_id)

    def _package_checkpoints(self, ckpt_dir: str) -> None:
        """
        @input: [str, checkpoint directory]
        @output: [None]
        @scenario: [Package DP checkpoints from workspace outputs directory]

        DP saves checkpoints in data/outputs/{save_name}-{seed}/checkpoints/.
        """
        # Use the save_name from dataset_info stored during run
        if not hasattr(self, "_dataset_info"):
            return

        save_name = self._dataset_info.get("save_name", "")
        if not save_name:
            return

        # Get seed from the run
        workspace_ckpt_dir = find_workspace_checkpoint_dir(
            self.policy_dir, save_name, self._seed
        )
        if workspace_ckpt_dir:
            from wrapper_base.checkpoint_util import package_checkpoints_for_eval
            package_checkpoints_for_eval(workspace_ckpt_dir, self.MODEL_NAME, self.logger)

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
        @scenario: [Cleanup combined zarr and manifest if training failed]
        """
        combined_abs_path = dataset_info.get("combined_abs_path", "")
        manifest_path = os.path.join(self._get_checkpoint_dir(dataset_info.get("task_id", "")), "training_run_manifest.txt")

        if not success:
            if os.path.isfile(manifest_path):
                os.remove(manifest_path)
                self.logger.info("Removed manifest after failed training: %s", manifest_path)

        if combined_abs_path and os.path.isdir(combined_abs_path):
            shutil.rmtree(combined_abs_path)
            self.logger.info("Deleted combined zarr: %s", dataset_info.get("combined_rel_path", ""))

    def write_manifest(
        self, ckpt_dir: str, task_id: str, task_slug: str, config_slug: str, total_episodes: int
    ) -> str:
        """
        @input: [various parameters]
        @output: [str, manifest file path]
        @scenario: [Write DP-specific training manifest]
        """
        os.makedirs(ckpt_dir, exist_ok=True)
        manifest_path = os.path.join(ckpt_dir, "training_run_manifest.txt")
        with open(manifest_path, "w", encoding="ascii") as mf:
            mf.write(f"task_id={task_id}\n")
            mf.write(f"dp_policy_dir={self.policy_dir}\n")
            mf.write(f"combined_task_slug={task_slug}\n")
            mf.write(f"combined_config_slug={config_slug}\n")
            mf.write(f"combined_total_episodes={total_episodes}\n")
        self.logger.info("Saved training manifest to %s", manifest_path)
        return manifest_path

    def run(
        self,
        task_id: str,
        yaml_path: str,
        gpu_id: int = 0,
        seed: Optional[int] = None,
    ) -> int:
        """
        Override to store dataset_info and seed for checkpoint packaging.
        """
        self._seed = seed if seed is not None else 0
        return super().run(task_id, yaml_path, gpu_id, seed)


def main(argv: list) -> int:
    """CLI entry point."""
    policy_dir = os.path.dirname(os.path.abspath(__file__))
    return DPTrWrapper.main(argv, policy_dir=policy_dir)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
