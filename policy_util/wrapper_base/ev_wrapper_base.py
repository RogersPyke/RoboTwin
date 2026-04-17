#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Abstract base class for evaluation wrappers (ACT/DP/TinyVLA).

@input: [str, config name], [int, gpu_id], [int, seed], [int, test_num]
@output: [int, exit code (0 for success)]
@scenario: [Run evaluation on trained checkpoints]

Design:
- Subclasses define MODEL_NAME and implement abstract methods
- Common functionality (logging, config loading, task parsing) handled by base class
- Model-specific logic (checkpoint resolution, eval command) in subclasses
- Supports three modes:
  1. Legacy mode: python3 _ev_wrapper.py <cfg_name>
  2. Merged YAML mode: python3 _ev_wrapper.py --task-id <id> --yaml <a.yaml> --yaml <b.yaml>
  3. Direct mode: python3 _ev_wrapper.py --ckpt-dir <dir> --task-name <name> --task-config <config>

Dependencies:
- PyYAML

Usage:
    class ACTEvWrapper(BaseEvWrapper):
        MODEL_NAME = "ACT"
        ENV_PREFIX = "ACT_FLOW"

        def _get_checkpoint_dir(self, task_slug, config_slug, total_episodes, cfg):
            return f"policy/ACT/act_ckpt/act-{task_slug}/{config_slug}-{total_episodes}"

        def _build_eval_command(self, task_name, task_config, ckpt_dir, runtime, cfg, ...):
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

import yaml


class BaseEvWrapper(ABC):
    """
    Abstract base class for model-specific evaluation wrappers.

    Subclasses must implement:
        - MODEL_NAME: Model identifier (ACT, DP, TinyVLA)
        - ENV_PREFIX: Environment variable prefix (ACT_FLOW, DP_FLOW, TVLA_FLOW)
        - _get_checkpoint_dir(): Return checkpoint directory path
        - _build_eval_command(): Build evaluation command

    The main() method handles:
        - Argument parsing
        - Logger setup
        - Config loading
        - Runtime resolution (CLI > ENV > YAML)
        - Evaluation execution
    """

    MODEL_NAME: str = ""
    ENV_PREFIX: str = ""

    def __init__(self, policy_dir: str):
        """
        @input: [str, policy directory path]
        @output: [BaseEvWrapper instance]
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

    def _load_yaml(self, path: str) -> Dict[str, Any]:
        """
        @input: [str, path to YAML file]
        @output: [dict, parsed YAML content]
        @scenario: [Load and validate YAML file]
        """
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            raise ValueError(f"Invalid YAML config: {path}")
        return cfg

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        @input: [dict, base config], [dict, override config]
        @output: [dict, merged config]
        @scenario: [Recursively merge two dicts]
        """
        result = dict(base)
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def _merge_yamls(self, yaml_paths: List[str]) -> Dict[str, Any]:
        """
        @input: [list, paths to YAML files]
        @output: [dict, merged config]
        @scenario: [Merge multiple YAML configs, later overrides earlier]
        """
        merged: Dict[str, Any] = {}
        for path in yaml_paths:
            cfg = self._load_yaml(path)
            merged = self._deep_merge(merged, cfg)
        return merged

    def _load_ev_cfg(self, cfg_name: str) -> Dict[str, Any]:
        """
        @input: [str, config name (with or without .yaml)]
        @output: [dict, parsed eval config]
        @scenario: [Load eval config from _ev_cfg/<name>.yaml]
        """
        base = cfg_name if cfg_name.endswith(".yaml") else f"{cfg_name}.yaml"
        cfg_path = os.path.join(self.policy_dir, "_ev_cfg", base)
        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(f"No config found: _ev_cfg/{base}")
        return self._load_yaml(cfg_path)

    def _ensure_required_keys(self, cfg: Dict[str, Any], keys: List[str]) -> None:
        """
        @input: [dict, config], [list, required keys]
        @output: [None, raises KeyError if missing]
        @scenario: [Validate required config keys]
        """
        missing = [k for k in keys if k not in cfg]
        if missing:
            raise KeyError(f"Missing required keys in config: {missing}")

    def _parse_task_rows(
        self, cfg: Dict[str, Any], key: str
    ) -> List[Tuple[str, str, int]]:
        """
        @input: [dict, config], [str, key (TRAIN_TASKS or EVAL_TASKS)]
        @output: [list of tuples, (task_name, task_config, expert_num)]
        @scenario: [Parse task rows from config]
        """
        if key not in cfg:
            raise ValueError(f"Config must define {key} as a non-empty list of [task_name, task_config, expert_num].")
        rows = cfg[key]
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"{key} must be a non-empty list.")

        parsed = []
        for idx, row in enumerate(rows):
            if not isinstance(row, (list, tuple)) or len(row) != 3:
                raise ValueError(f"{key}[{idx}] must be [task_name, task_config, expert_num], got {row!r}")
            task_name = str(row[0]).strip()
            task_config = str(row[1]).strip()
            expert_num = int(row[2])
            if not task_name or not task_config:
                raise ValueError(f"{key}[{idx}] contains an empty task_name/task_config.")
            if expert_num < 0:
                raise ValueError(f"{key}[{idx}] expert_num must be >= 0, got {expert_num}.")
            parsed.append((task_name, task_config, expert_num))
        return parsed

    def _get_task_from_config(
        self, cfg: Dict[str, Any], task_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        @input: [dict, config with ev_tasks], [str, task_id]
        @output: [Optional[dict], task config if found]
        @scenario: [Find task by task_id in ev_tasks list]
        """
        for task in cfg.get("ev_tasks", []):
            if task.get("task_id") == task_id:
                return task
        return None

    def _to_cli_bool(self, v: Any) -> str:
        """
        @input: [Any, bool-like value]
        @output: [str, "true" or "false"]
        @scenario: [Convert python truthy values to CLI boolean string]
        """
        if isinstance(v, str):
            return "true" if v.strip().lower() in ("1", "true", "yes", "y", "on") else "false"
        return "true" if bool(v) else "false"

    def _resolve_eval_end_reset_to_init(self, cfg: Dict[str, Any]) -> str:
        """
        @input: [dict, eval config]
        @output: [str, "true" or "false"]
        @scenario: [Resolve END_RESET_TO_INIT from config]
        """
        return self._to_cli_bool(cfg.get("END_RESET_TO_INIT", True))

    def _resolve_runtime(
        self,
        cfg: Dict[str, Any],
        cli_seed: Optional[int] = None,
        cli_gpu_id: Optional[str] = None,
        cli_test_num: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        @input: [dict, config], [optional CLI overrides]
        @output: [dict, resolved runtime values with sources]
        @scenario: [Resolve seed, gpu_id, test_num with priority: CLI > ENV > YAML]
        """
        seed = int(cfg.get("EVAL_SEED", cfg.get("seed", 0)))
        gpu_id = str(cfg.get("EVAL_GPU_ID", cfg.get("gpu_id", "0"))).strip()
        test_num = int(cfg.get("TEST_NUM", cfg.get("test_num", 100)))

        seed_source = "YAML:EVAL_SEED"
        gpu_source = "YAML:EVAL_GPU_ID"
        test_num_source = "YAML:TEST_NUM/default"

        # Environment variables
        env_seed = os.environ.get(f"{self.ENV_PREFIX}_SEED", "").strip()
        env_gpu = os.environ.get(f"{self.ENV_PREFIX}_GPU", "").strip()
        env_test_num = os.environ.get(f"{self.ENV_PREFIX}_TEST_NUM", "").strip()

        # CLI overrides
        if cli_seed is not None:
            seed = int(cli_seed)
            seed_source = "CLI:--seed"
        elif env_seed:
            seed = int(env_seed)
            seed_source = f"FLOW_ENV:{self.ENV_PREFIX}_SEED"

        if cli_gpu_id is not None:
            gpu_id = str(cli_gpu_id).strip()
            gpu_source = "CLI:--gpu-id"
        elif env_gpu:
            gpu_id = env_gpu
            gpu_source = f"FLOW_ENV:{self.ENV_PREFIX}_GPU"

        if cli_test_num is not None:
            test_num = int(cli_test_num)
            test_num_source = "CLI:--test-num"
        elif env_test_num:
            test_num = int(env_test_num)
            test_num_source = f"FLOW_ENV:{self.ENV_PREFIX}_TEST_NUM"

        if test_num < 1:
            raise ValueError("TEST_NUM must be >= 1")

        return {
            "seed": seed,
            "gpu_id": gpu_id,
            "test_num": test_num,
            "seed_source": seed_source,
            "gpu_source": gpu_source,
            "test_num_source": test_num_source,
        }

    def setup_env(self, gpu_id: str, cfg_name: Optional[str] = None) -> dict:
        """
        @input: [str, GPU ID], [str, config name]
        @output: [dict, environment variables for subprocess]
        @scenario: [Setup environment for evaluation subprocess]
        """
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        env["PYTHONWARNINGS"] = "ignore::UserWarning"
        env["PYTHONNOUSERSITE"] = "1"

        # Set config snapshot path if cfg_name provided
        if cfg_name:
            base = cfg_name if cfg_name.endswith(".yaml") else f"{cfg_name}.yaml"
            ev_cfg_path = os.path.abspath(os.path.join(self.policy_dir, "_ev_cfg", base))
            env[f"{self.MODEL_NAME.upper()}_EV_CFG_SNAPSHOT_SRC"] = ev_cfg_path

        return env

    def run_eval(self, cmd: List[str], env: dict, repo_root: str) -> None:
        """
        @input: [list, command tokens], [dict, environment], [str, repo root]
        @output: [None, raises on failure]
        @scenario: [Execute evaluation command in subprocess]
        """
        self.logger.info("Run: %s", " ".join(cmd))
        subprocess.run(cmd, check=True, env=env, cwd=repo_root)

    # ==================== Abstract Methods ====================

    @abstractmethod
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
        @scenario: [Return checkpoint directory for the trained model]
        """
        pass

    @abstractmethod
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
        @output: [list, command tokens for subprocess]
        @scenario: [Build evaluation command - model-specific implementation]
        """
        pass

    def _get_required_config_keys(self) -> List[str]:
        """
        @input: [None]
        @output: [list, required config keys]
        @scenario: [Return required keys for validation - override for model-specific requirements]
        """
        return ["TRAIN_TASKS", "EVAL_TASKS", "EVAL_SEED", "EVAL_GPU_ID", "END_RESET_TO_INIT"]

    # ==================== Legacy Mode ====================

    def _run_legacy_mode(
        self,
        cfg_name: str,
        gpu_id: Optional[str] = None,
        seed: Optional[int] = None,
        test_num: Optional[int] = None,
    ) -> int:
        """
        @input: [str, config name], [optional overrides]
        @output: [int, exit code]
        @scenario: [Run evaluation from legacy config file]
        """
        cfg = self._load_ev_cfg(cfg_name)
        self._ensure_required_keys(cfg, self._get_required_config_keys())

        train_rows = self._parse_task_rows(cfg, "TRAIN_TASKS")
        eval_rows = self._parse_task_rows(cfg, "EVAL_TASKS")

        runtime = self._resolve_runtime(cfg, cli_seed=seed, cli_gpu_id=gpu_id, cli_test_num=test_num)

        self.logger.info(
            "Resolved runtime: seed=%s (%s), gpu_id=%s (%s), test_num=%s (%s)",
            runtime["seed"],
            runtime["seed_source"],
            runtime["gpu_id"],
            runtime["gpu_source"],
            runtime["test_num"],
            runtime["test_num_source"],
        )

        train_slug = "__".join([row[0] for row in train_rows])
        config_slug = "__".join([row[1] for row in train_rows])
        total_episodes = int(sum([row[2] for row in train_rows]))

        ckpt_dir = self._get_checkpoint_dir(train_slug, config_slug, total_episodes, cfg)

        repo_root = os.path.abspath(os.path.join(self.policy_dir, "..", ".."))
        env = self.setup_env(runtime["gpu_id"], cfg_name)

        end_reset = self._resolve_eval_end_reset_to_init(cfg)

        for task_name, task_config, _ in eval_rows:
            cmd = self._build_eval_command(
                task_name=task_name,
                task_config=task_config,
                ckpt_dir=ckpt_dir,
                runtime=runtime,
                cfg=cfg,
                train_slug=train_slug,
                config_slug=config_slug,
                total_episodes=total_episodes,
            )
            self.logger.info(
                "Eval: task_name=%s task_config=%s ckpt_dir=%s test_num=%s END_RESET_TO_INIT=%s",
                task_name,
                task_config,
                ckpt_dir,
                runtime["test_num"],
                end_reset,
            )
            self.run_eval(cmd, env, repo_root)

        return 0

    # ==================== Merged YAML Mode ====================

    def _run_merged_yaml_mode(
        self,
        cfg: Dict[str, Any],
        task_id: str,
        gpu_id: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> int:
        """
        @input: [dict, merged config], [str, task_id], [optional overrides]
        @output: [int, exit code]
        @scenario: [Run evaluation from merged YAML config]
        """
        task = self._get_task_from_config(cfg, task_id)
        if task is None:
            raise ValueError(f"Task not found: {task_id}")

        model_defaults = cfg.get("model_defaults", {})
        test_num = task.get("test_num", model_defaults.get("test_num", 50))

        eval_tasks = task.get("eval_tasks", [])
        if not eval_tasks:
            raise ValueError(f"Task {task_id} must have eval_tasks")

        eval_rows = self._parse_task_rows_from_list(eval_tasks)

        # Resolve checkpoint directory
        if "ckpt_dir" in task and task["ckpt_dir"]:
            ckpt_dir = task["ckpt_dir"]
            train_slug = ""
            config_slug = Path(ckpt_dir).name
            total_episodes = 0
        else:
            train_tasks = task.get("train_tasks", [])
            if not train_tasks:
                raise ValueError(f"Task {task_id} must have either ckpt_dir or train_tasks")
            train_rows = self._parse_task_rows_from_list(train_tasks)
            train_slug = "__".join([r[0] for r in train_rows])
            config_slug = "__".join([r[1] for r in train_rows])
            total_episodes = int(sum([r[2] for r in train_rows]))
            ckpt_dir = self._get_checkpoint_dir(train_slug, config_slug, total_episodes, cfg)

        # Resolve runtime
        resolved_gpu = gpu_id if gpu_id is not None else 0
        resolved_seed = seed if seed is not None else cfg.get("seed", 0)

        runtime = {
            "seed": resolved_seed,
            "gpu_id": str(resolved_gpu),
            "test_num": test_num,
            "seed_source": "CLI" if seed is not None else "YAML",
            "gpu_source": "CLI" if gpu_id is not None else "default",
            "test_num_source": "task config",
        }

        repo_root = os.path.abspath(os.path.join(self.policy_dir, "..", ".."))
        env = self.setup_env(runtime["gpu_id"])

        end_reset = self._to_cli_bool(task.get("end_reset_to_init", model_defaults.get("end_reset_to_init", True)))

        self.logger.info(
            "Task ID: %s | GPU: %s | Seed: %s | Test Num: %s | END_RESET_TO_INIT: %s",
            task_id,
            runtime["gpu_id"],
            runtime["seed"],
            runtime["test_num"],
            end_reset,
        )
        self.logger.info("Checkpoint: %s", ckpt_dir)

        for task_name, task_config, _ in eval_rows:
            cmd = self._build_eval_command(
                task_name=task_name,
                task_config=task_config,
                ckpt_dir=ckpt_dir,
                runtime=runtime,
                cfg=cfg,
                train_slug=train_slug,
                config_slug=config_slug,
                total_episodes=total_episodes,
            )
            self.logger.info("Eval: task_name=%s task_config=%s", task_name, task_config)
            self.run_eval(cmd, env, repo_root)

        return 0

    def _parse_task_rows_from_list(self, rows: List) -> List[Tuple[str, str, int]]:
        """
        @input: [list, task rows]
        @output: [list of tuples, (task_name, task_config, expert_num)]
        @scenario: [Parse task rows from a list]
        """
        if not rows:
            raise ValueError("Task rows must be a non-empty list")
        parsed = []
        for i, row in enumerate(rows):
            if not isinstance(row, (list, tuple)) or len(row) != 3:
                raise ValueError(f"Row {i} must be [task_name, task_config, expert_num], got {row!r}")
            parsed.append((str(row[0]).strip(), str(row[1]).strip(), int(row[2])))
        return parsed

    # ==================== Direct Mode ====================

    def _run_direct_mode(
        self,
        ckpt_dir: str,
        task_name: str,
        task_config: str,
        gpu_id: int,
        seed: int,
        test_num: int,
        **kwargs,
    ) -> int:
        """
        @input: [various direct parameters]
        @output: [int, exit code]
        @scenario: [Run evaluation with directly specified parameters]
        """
        config_slug = Path(ckpt_dir).name
        runtime = {
            "seed": seed,
            "gpu_id": str(gpu_id),
            "test_num": test_num,
            "seed_source": "CLI",
            "gpu_source": "CLI",
            "test_num_source": "CLI",
        }

        repo_root = os.path.abspath(os.path.join(self.policy_dir, "..", ".."))
        env = self.setup_env(runtime["gpu_id"])

        end_reset = self._to_cli_bool(kwargs.get("end_reset_to_init", True))

        self.logger.info(
            "Direct mode: ckpt_dir=%s, task_name=%s, task_config=%s, gpu=%d, seed=%d, test_num=%d, END_RESET_TO_INIT=%s",
            ckpt_dir,
            task_name,
            task_config,
            gpu_id,
            seed,
            test_num,
            end_reset,
        )

        cmd = self._build_eval_command(
            task_name=task_name,
            task_config=task_config,
            ckpt_dir=ckpt_dir,
            runtime=runtime,
            cfg={},
            train_slug="",
            config_slug=config_slug,
            total_episodes=0,
        )
        self.run_eval(cmd, env, repo_root)

        return 0

    # ==================== Main Entry Point ====================

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
        @scenario: [Execute evaluation from config (legacy mode)]
        """
        return self._run_legacy_mode(cfg_name, gpu_id, seed, test_num)

    @classmethod
    def main(cls, argv: List[str]) -> int:
        """
        @input: [list, command line arguments]
        @output: [int, exit code]
        @scenario: [CLI entry point for evaluation wrapper]
        """
        parser = argparse.ArgumentParser(
            description=f"{cls.MODEL_NAME} eval wrapper (config under _ev_cfg/*.yaml)"
        )

        # Legacy mode arguments
        parser.add_argument(
            "cfg_name",
            nargs="?",
            type=str,
            help="(legacy) Config name (file: _ev_cfg/<name>.yaml).",
        )
        parser.add_argument(
            "--config",
            dest="config",
            type=str,
            required=False,
            help="(legacy) Config name.",
        )

        # Merged YAML mode arguments
        parser.add_argument(
            "--task-id",
            dest="task_id",
            type=str,
            help="Task ID from ev_tasks.yaml (merged YAML mode).",
        )
        parser.add_argument(
            "--yaml",
            dest="yaml_paths",
            action="append",
            type=str,
            help="YAML config path (can specify multiple, merged YAML mode).",
        )

        # Direct mode arguments
        parser.add_argument(
            "--ckpt-dir",
            dest="ckpt_dir",
            type=str,
            help="Direct checkpoint directory path (direct mode).",
        )
        parser.add_argument(
            "--task-name",
            dest="task_name",
            type=str,
            help="Task name for evaluation (direct mode).",
        )
        parser.add_argument(
            "--task-config",
            dest="task_config",
            type=str,
            help="Task config for evaluation (direct mode).",
        )

        # Common overrides
        parser.add_argument(
            "--gpu-id",
            dest="gpu_id",
            type=int,
            required=False,
            help="Optional GPU id override.",
        )
        parser.add_argument(
            "--seed",
            dest="seed",
            type=int,
            required=False,
            help="Optional eval seed override.",
        )
        parser.add_argument(
            "--test-num",
            dest="test_num",
            type=int,
            required=False,
            help="Optional eval test_num override.",
        )
        parser.add_argument(
            "--end-reset-to-init",
            dest="end_reset_to_init",
            type=str,
            required=False,
            help="END_RESET_TO_INIT override: true/false",
        )

        args = parser.parse_args(argv[1:])

        policy_dir = os.path.dirname(os.path.abspath(argv[0]))
        wrapper = cls(policy_dir)

        try:
            # Direct mode
            if args.ckpt_dir and args.task_name and args.task_config:
                gpu_id = args.gpu_id if args.gpu_id is not None else 0
                seed = args.seed if args.seed is not None else 0
                test_num = args.test_num if args.test_num is not None else 50
                end_reset = True
                if args.end_reset_to_init is not None:
                    end_reset = wrapper._to_cli_bool(args.end_reset_to_init) == "true"
                return wrapper._run_direct_mode(
                    ckpt_dir=args.ckpt_dir,
                    task_name=args.task_name,
                    task_config=args.task_config,
                    gpu_id=gpu_id,
                    seed=seed,
                    test_num=test_num,
                    end_reset_to_init=end_reset,
                )

            # Merged YAML mode
            if args.task_id and args.yaml_paths:
                cfg = wrapper._merge_yamls(args.yaml_paths)
                gpu_id = args.gpu_id if args.gpu_id is not None else None
                seed = args.seed if args.seed is not None else None
                return wrapper._run_merged_yaml_mode(cfg, args.task_id, gpu_id, seed)

            # Legacy mode
            cfg_name = args.config if args.config is not None else args.cfg_name
            if cfg_name:
                return wrapper._run_legacy_mode(
                    cfg_name=cfg_name,
                    gpu_id=str(args.gpu_id) if args.gpu_id is not None else None,
                    seed=args.seed,
                    test_num=args.test_num,
                )

            parser.error(
                f"Require either --ckpt-dir + --task-name + --task-config, "
                f"or --task-id + --yaml, or legacy cfg_name."
            )

        except Exception as exc:
            wrapper.logger.error("Wrapper failed: %s", str(exc))
            wrapper.logger.error("Stack trace:\n%s", traceback.format_exc())
            return 1
