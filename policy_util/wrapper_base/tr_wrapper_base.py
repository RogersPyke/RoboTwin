#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Abstract base class for training wrappers (ACT/DP/TinyVLA).

@input: [str, task_id], [str, yaml_path], [int, gpu_id], [int, seed]
@output: [int, exit code (0 for success)]
@scenario: [Execute training task from unified config]

Design:
- Subclasses define MODEL_NAME and implement abstract methods
- Common functionality (logging, config loading) handled by base class
- Model-specific logic (data prep, training command) in subclasses

Dependencies:
- PyYAML

Usage:
    class ACTTrWrapper(BaseTrWrapper):
        MODEL_NAME = "ACT"

        def _prepare_dataset(self, cfg, task, task_id):
            # ACT-specific data preparation
            ...

        def _build_train_command(self, cfg, task, gpu_id, seed, ckpt_dir):
            # ACT-specific training command
            ...
"""

import argparse
import logging
import os
import subprocess
import sys
import traceback
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config_util import (
    deep_merge,
    load_unified_config,
    merge_global_and_model_config,
    resolve_task_config,
)


class BaseTrWrapper(ABC):
    """
    Abstract base class for model-specific training wrappers.

    Subclasses must implement:
        - MODEL_NAME: Model identifier (ACT, DP, TinyVLA)
        - _prepare_dataset(): Prepare data for training
        - _build_train_command(): Build training command
        - _get_checkpoint_dir(): Return checkpoint directory path
        - _cleanup_after_training(): Cleanup temporary files (optional)

    The main() method handles:
        - Argument parsing
        - Logger setup
        - Config loading and merging
        - Training execution
        - Checkpoint packaging
    """

    MODEL_NAME: str = ""

    def __init__(self, policy_dir: str):
        """
        @input: [str, policy directory path]
        @output: [BaseTrWrapper instance]
        @scenario: [Initialize wrapper with policy directory and setup logger]
        """
        self.policy_dir = os.path.abspath(policy_dir)
        os.chdir(self.policy_dir)
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """
        @input: [None]
        @output: [logging.Logger, configured logger]
        @scenario: [Setup logger with file and console handlers]

        Creates logs/ directory and writes to timestamped log file.
        """
        logs_dir = os.path.join(self.policy_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)

        script_name = self.__class__.__name__
        tz_8 = timezone(timedelta(hours=8))
        ts = datetime.now(tz_8).strftime("%Y%m%d%H%M%S")
        log_path = os.path.join(logs_dir, f"{script_name}_{ts}.log")

        logger = logging.getLogger(script_name)
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            formatter = logging.Formatter("[Wrapper] [%(levelname)s] %(message)s")
            fh = logging.FileHandler(log_path)
            fh.setFormatter(formatter)
            sh = logging.StreamHandler(sys.stdout)
            sh.setFormatter(formatter)
            logger.addHandler(fh)
            logger.addHandler(sh)

        return logger

    def load_config(self, yaml_path: str) -> Dict[str, Any]:
        """
        @input: [str, path to unified YAML config]
        @output: [dict, merged config for this model]
        @scenario: [Load unified config and extract model-specific settings]
        """
        unified_cfg = load_unified_config(yaml_path)
        return merge_global_and_model_config(unified_cfg, self.MODEL_NAME)

    def get_task_config(self, cfg: Dict[str, Any], task_id: str) -> Optional[Dict[str, Any]]:
        """
        @input: [dict, config dict], [str, task_id]
        @output: [Optional[dict], task config if found]
        @scenario: [Find task config by task_id]
        """
        return resolve_task_config(cfg, task_id)

    def merge_task_params(
        self, cfg: Dict[str, Any], task: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        @input: [dict, model config], [dict, task config]
        @output: [dict, merged params with task overrides]
        @scenario: [Merge model_defaults with task-specific params]
        """
        model_defaults = cfg.get("model_defaults", {})
        params = dict(model_defaults)
        excluded_keys = {"task_id", "data_folder", "data_sources", "_resolved_data"}
        for key, value in task.items():
            if key not in excluded_keys:
                params[key] = value
        return params

    def parse_task_data_sources(
        self, task: Dict[str, Any], task_id: str, robotwin_root: Path
    ) -> Tuple[List[str], List[str], List[int]]:
        """
        @input: [dict, task config], [str, task_id], [Path, robotwin root]
        @output: [tuple, (task_names, task_configs, expert_counts)]
        @scenario: [Extract task names, configs, and episode counts from task config]

        Handles both single-task (data_folder) and multi-task (data_sources) configs.
        """
        if "data_folder" in task:
            task_names, task_configs, expert_counts = self._parse_single_task(
                task, task_id, robotwin_root
            )
        elif "data_sources" in task:
            task_names, task_configs, expert_counts = self._parse_multi_task(task)
        else:
            raise ValueError(f"Task {task_id} must have data_folder or data_sources")

        return task_names, task_configs, expert_counts

    def _parse_single_task(
        self, task: Dict[str, Any], task_id: str, robotwin_root: Path
    ) -> Tuple[List[str], List[str], List[int]]:
        """Parse single-task data_folder. Override in subclass for model-specific parsing."""
        task_names = [task.get("task_name", task_id)]
        task_configs = [task.get("task_config", "demo_clean")]
        expert_counts = [task.get("expert_num", 100)]
        return task_names, task_configs, expert_counts

    def _parse_multi_task(
        self, task: Dict[str, Any]
    ) -> Tuple[List[str], List[str], List[int]]:
        """Parse multi-task data_sources."""
        task_names = [src["task_name"] for src in task["data_sources"]]
        task_configs = [src["task_config"] for src in task["data_sources"]]
        expert_counts = [src["expert_num"] for src in task["data_sources"]]
        return task_names, task_configs, expert_counts

    def write_manifest(
        self, ckpt_dir: str, task_id: str, task_slug: str, config_slug: str, total_episodes: int
    ) -> str:
        """
        @input: [str, checkpoint dir], [str, task_id], [str, task slug], [str, config slug], [int, total episodes]
        @output: [str, manifest file path]
        @scenario: [Write training run manifest for later reference]
        """
        os.makedirs(ckpt_dir, exist_ok=True)
        manifest_path = os.path.join(ckpt_dir, "training_run_manifest.txt")
        with open(manifest_path, "w", encoding="ascii") as mf:
            mf.write(f"task_id={task_id}\n")
            mf.write(f"{self.MODEL_NAME.lower()}_policy_dir={self.policy_dir}\n")
            mf.write(f"combined_task_slug={task_slug}\n")
            mf.write(f"combined_config_slug={config_slug}\n")
            mf.write(f"combined_total_episodes={total_episodes}\n")
        self.logger.info("Saved training manifest to %s", manifest_path)
        return manifest_path

    def setup_env(self, gpu_id: int) -> dict:
        """
        @input: [int, GPU ID]
        @output: [dict, environment variables for subprocess]
        @scenario: [Setup environment for training subprocess]
        """
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["PYTHONNOUSERSITE"] = "1"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        return env

    def run_training(self, cmd: List[str], env: dict) -> None:
        """
        @input: [list, command tokens], [dict, environment]
        @output: [None, raises on failure]
        @scenario: [Execute training command in subprocess]
        """
        self.logger.info("Launching training: %s", " ".join(cmd))
        subprocess.run(cmd, check=True, env=env)

    # ==================== Abstract Methods ====================

    @abstractmethod
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
        @input: [dict, model config], [dict, task config], [str, task_id], [list, task names/configs/counts]
        @output: [tuple, (dataset_path or key, additional info dict)]
        @scenario: [Prepare dataset for training - model-specific implementation]

        ACT: Create symlinks in processed_data/, return combined_key
        DP: Create combined zarr, return zarr_path
        TinyVLA: Return dataset_dirs list
        """
        pass

    @abstractmethod
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
        @output: [list, command tokens for subprocess]
        @scenario: [Build training command - model-specific implementation]
        """
        pass

    @abstractmethod
    def _get_checkpoint_dir(self, task_id: str) -> str:
        """
        @input: [str, task_id]
        @output: [str, checkpoint directory path]
        @scenario: [Return checkpoint directory for this task]
        """
        pass

    def _cleanup_after_training(
        self,
        cfg: Dict[str, Any],
        task: Dict[str, Any],
        dataset_info: Dict[str, Any],
        success: bool,
    ) -> None:
        """
        @input: [dict, config], [dict, task], [dict, dataset info], [bool, training success]
        @output: [None]
        @scenario: [Cleanup temporary files after training - optional override]

        Default implementation does nothing. Override for model-specific cleanup.
        """
        pass

    def _package_checkpoints(self, ckpt_dir: str) -> None:
        """
        @input: [str, checkpoint directory]
        @output: [None, creates step_packages subdirectory]
        @scenario: [Package checkpoints for evaluation - optional override]

        Default implementation uses checkpoint_util.package_checkpoints_for_eval.
        """
        from .checkpoint_util import package_checkpoints_for_eval

        package_checkpoints_for_eval(ckpt_dir, self.MODEL_NAME, self.logger)

    # ==================== Main Entry Point ====================

    def run(
        self,
        task_id: str,
        yaml_path: str,
        gpu_id: int = 0,
        seed: Optional[int] = None,
    ) -> int:
        """
        @input: [str, task_id], [str, yaml_path], [int, gpu_id], [Optional[int], seed]
        @output: [int, exit code (0 for success)]
        @scenario: [Execute training task from unified config]

        This is the main entry point called by main().
        """
        try:
            cfg = self.load_config(yaml_path)
            task = self.get_task_config(cfg, task_id)
            if task is None:
                raise ValueError(f"Task not found: {task_id}")

            actual_seed = seed if seed is not None else cfg.get("seed", 0)
            params = self.merge_task_params(cfg, task)

            robotwin_root = Path(self.policy_dir).parent.parent
            task_names, task_configs, expert_counts = self.parse_task_data_sources(
                task, task_id, robotwin_root
            )

            self.logger.info("Task ID: %s", task_id)
            self.logger.info("Tasks: %s", task_names)
            self.logger.info("Configs: %s", task_configs)
            self.logger.info("Counts: %s", expert_counts)
            self.logger.info("Seed: %s", actual_seed)
            self.logger.info("GPU ID: %s", gpu_id)

            # Prepare dataset (model-specific)
            dataset_path, dataset_info = self._prepare_dataset(
                cfg, task, task_id, task_names, task_configs, expert_counts
            )
            dataset_info["dataset_path"] = dataset_path

            # Setup checkpoint directory and manifest
            ckpt_dir = self._get_checkpoint_dir(task_id)
            task_slug = "__".join(task_names)
            config_slug = "__".join(task_configs)
            total_episodes = int(sum(expert_counts))
            self.write_manifest(ckpt_dir, task_id, task_slug, config_slug, total_episodes)

            # Build and run training command
            cmd = self._build_train_command(
                cfg, task, params, task_id, task_names, task_configs, expert_counts,
                dataset_info, ckpt_dir, gpu_id, actual_seed,
            )
            env = self.setup_env(gpu_id)

            train_success = False
            try:
                self.run_training(cmd, env)
                train_success = True
                self._package_checkpoints(ckpt_dir)
            finally:
                self._cleanup_after_training(cfg, task, dataset_info, train_success)

            return 0

        except Exception as exc:
            self.logger.error("Wrapper failed: %s", str(exc))
            self.logger.error("Stack trace:\n%s", traceback.format_exc())
            return 1

    @classmethod
    def main(cls, argv: List[str]) -> int:
        """
        @input: [list, command line arguments]
        @output: [int, exit code]
        @scenario: [CLI entry point for training wrapper]
        """
        parser = argparse.ArgumentParser(description=f"{cls.MODEL_NAME} train wrapper.")
        parser.add_argument(
            "--task-id",
            dest="task_id",
            type=str,
            required=True,
            help="Task ID from tr_tasks.",
        )
        parser.add_argument(
            "--yaml",
            dest="yaml_path",
            type=str,
            required=True,
            help="Unified YAML config path.",
        )
        parser.add_argument(
            "--gpu-id",
            dest="gpu_id",
            type=int,
            default=0,
            help="GPU ID.",
        )
        parser.add_argument(
            "--seed",
            dest="seed",
            type=int,
            default=None,
            help="Random seed.",
        )
        args = parser.parse_args(argv[1:])

        policy_dir = os.path.dirname(os.path.abspath(__file__))
        wrapper = cls(policy_dir)
        return wrapper.run(
            task_id=args.task_id,
            yaml_path=args.yaml_path,
            gpu_id=args.gpu_id,
            seed=args.seed,
        )
