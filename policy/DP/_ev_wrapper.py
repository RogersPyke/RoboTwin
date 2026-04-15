#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DP Eval wrapper.

Usage (new multi-YAML mode):
    python3 _ev_wrapper.py --task-id <id> --yaml <shared.yaml> --yaml <model.yaml> [--gpu-id N] [--seed N]

Usage (legacy single-YAML mode):
    python3 _ev_wrapper.py <cfg_name>
    python3 _ev_wrapper.py --config <cfg_name>   # (legacy)

The multi-YAML mode merges configs with later overriding earlier.
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

import yaml


def _setup_logger(dp_dir: str) -> logging.Logger:
    logs_dir = os.path.join(dp_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    script_name = os.path.splitext(os.path.basename(__file__))[0]
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


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML config: {path}")
    return cfg


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _merge_yamls(yaml_paths: List[str]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for path in yaml_paths:
        cfg = _load_yaml(path)
        merged = _deep_merge(merged, cfg)
    return merged


def _get_task_from_config(
    cfg: Dict[str, Any], task_id: str
) -> Optional[Dict[str, Any]]:
    for task in cfg.get("ev_tasks", []):
        if task.get("task_id") == task_id:
            return task
    return None


def _parse_task_rows(rows: List) -> List[tuple]:
    if not rows:
        raise ValueError("Task rows must be a non-empty list")
    out = []
    for i, row in enumerate(rows):
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            raise ValueError(
                f"Row {i} must be [task_name, task_config, expert_num], got {row!r}"
            )
        out.append((str(row[0]).strip(), str(row[1]).strip(), int(row[2])))
    return out


def _to_cli_bool(v) -> str:
    if isinstance(v, str):
        return (
            "true" if v.strip().lower() in ("1", "true", "yes", "y", "on") else "false"
        )
    return "true" if bool(v) else "false"


def _run_from_merged_config(
    cfg: Dict[str, Any],
    task_id: str,
    gpu_id: int,
    seed: int,
    logger: logging.Logger,
    dp_dir: str,
) -> int:
    task = _get_task_from_config(cfg, task_id)
    if task is None:
        raise ValueError(f"Task not found: {task_id}")

    model_defaults = cfg.get("model_defaults", {})
    test_num = task.get("test_num", model_defaults.get("test_num", 50))
    end_reset_to_init = task.get(
        "end_reset_to_init", model_defaults.get("end_reset_to_init", True)
    )
    checkpoint_num = task.get(
        "checkpoint_num", model_defaults.get("checkpoint_num", 600)
    )
    head_camera_type = task.get(
        "eval_head_camera_type", model_defaults.get("eval_head_camera_type", "D435")
    )

    eval_tasks = task.get("eval_tasks", [])
    if not eval_tasks:
        raise ValueError(f"Task {task_id} must have eval_tasks")

    eval_rows = _parse_task_rows(eval_tasks)

    if "ckpt_dir" in task and task["ckpt_dir"]:
        ckpt_dir = task["ckpt_dir"]
    else:
        train_tasks = task.get("train_tasks", [])
        if not train_tasks:
            raise ValueError(f"Task {task_id} must have either ckpt_dir or train_tasks")
        train_rows = _parse_task_rows(train_tasks)
        names = [r[0] for r in train_rows]
        cfgs = [r[1] for r in train_rows]
        nums = [r[2] for r in train_rows]
        train_task_slug = "__".join(names)
        train_config_slug = "__".join(cfgs)
        train_total = int(sum(nums))
        ckpt_dir = (
            f"policy/DP/checkpoints/{train_task_slug}-{train_config_slug}-{train_total}"
        )

    repo_root = os.path.abspath(os.path.join(dp_dir, "..", ".."))

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONWARNINGS"] = "ignore::UserWarning"
    env["PYTHONNOUSERSITE"] = "1"

    logger.info(
        "Task ID: %s | GPU: %d | Seed: %d | Test Num: %d | Checkpoint Num: %d | Head Camera: %s | END_RESET_TO_INIT: %s",
        task_id,
        gpu_id,
        seed,
        test_num,
        checkpoint_num,
        head_camera_type,
        _to_cli_bool(end_reset_to_init),
    )
    logger.info("Checkpoint: %s", ckpt_dir)

    for task_name, task_config in eval_rows:
        cmd = [
            sys.executable,
            "script/eval_policy.py",
            "--config",
            "policy/DP/diffusion_policy/config/robot_dp_14.yaml",
            "--overrides",
            "--task_name",
            task_name,
            "--task_config",
            task_config,
            "--ckpt_dir",
            ckpt_dir,
            "--seed",
            str(seed),
            "--test_num",
            str(test_num),
            "--checkpoint_num",
            str(checkpoint_num),
            "--eval_head_camera_type",
            head_camera_type,
            "--END_RESET_TO_INIT",
            _to_cli_bool(end_reset_to_init),
        ]
        logger.info("Eval: task_name=%s task_config=%s", task_name, task_config)
        logger.info("Run: %s", " ".join(cmd))
        subprocess.run(cmd, check=True, env=env, cwd=repo_root)

    return 0


def _run_legacy_mode(cfg_name: str, dp_dir: str, logger: logging.Logger, args) -> int:
    base = cfg_name if cfg_name.endswith(".yaml") else f"{cfg_name}.yaml"
    cfg_path = os.path.join(dp_dir, "_ev_cfg", base)
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"No config found: _ev_cfg/{base}")

    cfg = _load_yaml(cfg_path)

    for key in (
        "TRAIN_TASKS",
        "EVAL_TASKS",
        "EVAL_SEED",
        "EVAL_GPU_ID",
        "END_RESET_TO_INIT",
    ):
        if key not in cfg:
            raise KeyError(f"Missing required key: {key}")

    train_tasks = cfg["TRAIN_TASKS"]
    eval_tasks = cfg["EVAL_TASKS"]

    train_rows = _parse_task_rows(train_tasks)
    eval_rows = _parse_task_rows(eval_tasks)

    names = [r[0] for r in train_rows]
    cfgs = [r[1] for r in train_rows]
    nums = [r[2] for r in train_rows]

    train_task_slug = "__".join(names)
    train_config_slug = "__".join(cfgs)
    train_total = int(sum(nums))

    seed = int(cfg["EVAL_SEED"])
    gpu_id = int(cfg["EVAL_GPU_ID"])
    test_num = int(cfg.get("TEST_NUM", 100))
    checkpoint_num = int(cfg.get("CHECKPOINT_NUM", 600))
    head_camera_type = cfg.get("EVAL_HEAD_CAMERA_TYPE", "D435")
    end_reset_to_init = _to_cli_bool(cfg["END_RESET_TO_INIT"])

    env_seed = os.environ.get("DP_FLOW_SEED", "").strip()
    env_gpu = os.environ.get("DP_FLOW_GPU", "").strip()
    env_test_num = os.environ.get("DP_FLOW_TEST_NUM", "").strip()
    env_checkpoint_num = os.environ.get("DP_FLOW_CHECKPOINT_NUM", "").strip()
    env_head_camera = os.environ.get("DP_FLOW_EVAL_HEAD_CAMERA_TYPE", "").strip()

    if args.seed is not None:
        seed = int(args.seed)
    elif env_seed:
        seed = int(env_seed)
    if args.gpu_id is not None:
        gpu_id = int(args.gpu_id)
    elif env_gpu:
        gpu_id = int(env_gpu)
    if args.test_num is not None:
        test_num = int(args.test_num)
    elif env_test_num:
        test_num = int(env_test_num)
    if args.checkpoint_num is not None:
        checkpoint_num = int(args.checkpoint_num)
    elif env_checkpoint_num:
        checkpoint_num = int(env_checkpoint_num)
    if args.head_camera_type is not None:
        head_camera_type = args.head_camera_type
    elif env_head_camera:
        head_camera_type = env_head_camera
    if args.end_reset_to_init is not None:
        end_reset_to_init = _to_cli_bool(args.end_reset_to_init)
    elif os.environ.get("DP_FLOW_END_RESET_TO_INIT", "").strip():
        end_reset_to_init = _to_cli_bool(os.environ["DP_FLOW_END_RESET_TO_INIT"])

    ckpt_dir = (
        f"policy/DP/checkpoints/{train_task_slug}-{train_config_slug}-{train_total}"
    )
    repo_root = os.path.abspath(os.path.join(dp_dir, "..", ".."))

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONWARNINGS"] = "ignore::UserWarning"
    env["PYTHONNOUSERSITE"] = "1"

    logger.info(
        "Legacy mode: seed=%d, gpu_id=%d, test_num=%d, checkpoint_num=%d, head_camera=%s, END_RESET_TO_INIT=%s",
        seed,
        gpu_id,
        test_num,
        checkpoint_num,
        head_camera_type,
        end_reset_to_init,
    )
    logger.info("Checkpoint: %s", ckpt_dir)

    for task_name, task_config in eval_rows:
        cmd = [
            sys.executable,
            "script/eval_policy.py",
            "--config",
            "policy/DP/diffusion_policy/config/robot_dp_14.yaml",
            "--overrides",
            "--task_name",
            task_name,
            "--task_config",
            task_config,
            "--ckpt_dir",
            ckpt_dir,
            "--seed",
            str(seed),
            "--test_num",
            str(test_num),
            "--checkpoint_num",
            str(checkpoint_num),
            "--eval_head_camera_type",
            head_camera_type,
            "--END_RESET_TO_INIT",
            end_reset_to_init,
        ]
        logger.info("Eval: task_name=%s task_config=%s", task_name, task_config)
        logger.info("Run: %s", " ".join(cmd))
        subprocess.run(cmd, check=True, env=env, cwd=repo_root)

    return 0


def _run_direct_mode(
    ckpt_dir: str,
    task_name: str,
    task_config: str,
    gpu_id: int,
    seed: int,
    test_num: int,
    checkpoint_num: int,
    head_camera_type: str,
    end_reset_to_init: bool,
    logger: logging.Logger,
    dp_dir: str,
) -> int:
    """
    Direct mode: specify checkpoint directory and task parameters directly.
    """
    repo_root = os.path.abspath(os.path.join(dp_dir, "..", ".."))

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONWARNINGS"] = "ignore::UserWarning"
    env["PYTHONNOUSERSITE"] = "1"

    logger.info(
        "Direct mode: ckpt_dir=%s, task_name=%s, task_config=%s, gpu=%d, seed=%d, test_num=%d, checkpoint_num=%d, head_camera=%s, END_RESET_TO_INIT=%s",
        ckpt_dir,
        task_name,
        task_config,
        gpu_id,
        seed,
        test_num,
        checkpoint_num,
        head_camera_type,
        _to_cli_bool(end_reset_to_init),
    )

    cmd = [
        sys.executable,
        "script/eval_policy.py",
        "--config",
        "policy/DP/diffusion_policy/config/robot_dp_14.yaml",
        "--overrides",
        "--task_name",
        task_name,
        "--task_config",
        task_config,
        "--ckpt_dir",
        ckpt_dir,
        "--seed",
        str(seed),
        "--test_num",
        str(test_num),
        "--checkpoint_num",
        str(checkpoint_num),
        "--eval_head_camera_type",
        head_camera_type,
        "--END_RESET_TO_INIT",
        _to_cli_bool(end_reset_to_init),
    ]
    logger.info("Run: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env, cwd=repo_root)

    return 0


def main(argv: list) -> int:
    parser = argparse.ArgumentParser(description="DP eval wrapper.")
    parser.add_argument(
        "--task-id", dest="task_id", type=str, help="Task ID from ev_tasks.yaml."
    )
    parser.add_argument(
        "--yaml",
        dest="yaml_paths",
        action="append",
        type=str,
        help="YAML config path (can specify multiple).",
    )
    parser.add_argument(
        "--ckpt-dir",
        dest="ckpt_dir",
        type=str,
        default=None,
        help="Direct checkpoint directory path (direct mode).",
    )
    parser.add_argument(
        "--task-name",
        dest="task_name",
        type=str,
        default=None,
        help="Task name for evaluation (direct mode).",
    )
    parser.add_argument(
        "--task-config",
        dest="task_config",
        type=str,
        default=None,
        help="Task config for evaluation (direct mode).",
    )
    parser.add_argument(
        "--gpu-id", dest="gpu_id", type=int, default=None, help="GPU ID."
    )
    parser.add_argument(
        "--seed", dest="seed", type=int, default=None, help="Random seed."
    )
    parser.add_argument(
        "--test-num", dest="test_num", type=int, default=None, help="Test rollouts."
    )
    parser.add_argument(
        "--checkpoint-num",
        dest="checkpoint_num",
        type=int,
        default=None,
        help="Checkpoint number.",
    )
    parser.add_argument(
        "--head-camera-type",
        dest="head_camera_type",
        type=str,
        default=None,
        help="Head camera type.",
    )
    parser.add_argument(
        "--end-reset-to-init",
        dest="end_reset_to_init",
        type=str,
        default=None,
        help="END_RESET_TO_INIT override: true/false",
    )
    parser.add_argument("cfg_name", nargs="?", type=str, help="(legacy) Config name.")
    parser.add_argument(
        "--config",
        dest="config",
        type=str,
        required=False,
        help="(legacy) Config name (file: _ev_cfg/<name>.yaml).",
    )

    args = parser.parse_args(argv[1:])

    dp_dir = os.path.dirname(os.path.abspath(__file__))
    logger = _setup_logger(dp_dir)

    try:
        if args.ckpt_dir and args.task_name and args.task_config:
            gpu_id = args.gpu_id if args.gpu_id is not None else 0
            seed = args.seed if args.seed is not None else 0
            test_num = args.test_num if args.test_num is not None else 50
            checkpoint_num = (
                args.checkpoint_num if args.checkpoint_num is not None else 600
            )
            head_camera_type = (
                args.head_camera_type if args.head_camera_type is not None else "D435"
            )
            end_reset = True
            if args.end_reset_to_init is not None:
                end_reset = _to_cli_bool(args.end_reset_to_init) == "true"
            return _run_direct_mode(
                args.ckpt_dir,
                args.task_name,
                args.task_config,
                gpu_id,
                seed,
                test_num,
                checkpoint_num,
                head_camera_type,
                end_reset,
                logger,
                dp_dir,
            )
        elif args.task_id and args.yaml_paths:
            cfg = _merge_yamls(args.yaml_paths)
            gpu_id = args.gpu_id if args.gpu_id is not None else 0
            seed = args.seed if args.seed is not None else cfg.get("seed", 0)
            return _run_from_merged_config(
                cfg, args.task_id, gpu_id, seed, logger, dp_dir
            )
        elif args.config is not None or args.cfg_name:
            cfg_name = args.config if args.config is not None else args.cfg_name
            if not cfg_name:
                parser.error("Missing cfg_name.")
            return _run_legacy_mode(cfg_name, dp_dir, logger, args)
        else:
            parser.error(
                "Require either --ckpt-dir + --task-name + --task-config, or --task-id + --yaml, or legacy cfg_name."
            )
    except Exception as exc:
        logger.error("Wrapper failed: %s", str(exc))
        logger.error("Stack trace:\n%s", traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
