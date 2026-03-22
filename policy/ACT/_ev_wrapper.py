#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-task eval wrapper: reads _ev_cfg/<name>.yaml, assembles ckpt path and args, runs eval.
Usage: python3 _ev_wrapper.py --config <name>
Config name is without extension; file must be _ev_cfg/<name>.yaml.
"""
import argparse
import logging
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone, timedelta

import yaml


def _setup_logger(act_dir: str) -> logging.Logger:
    logs_dir = os.path.join(act_dir, "logs")
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


def _parse_train_tasks_for_ckpt(cfg: dict) -> tuple:
    """
    @input: [dict, eval or train YAML]
    @output: [tuple, (ckpt_setting str, expert_data_num int) matching multi-task train layout]
    @scenario: [Build checkpoint slug from TRAIN_TASKS or legacy TASK_CFG_TO_TRAIN / TASK_NUM_TO_TRAIN]
    """
    if "TRAIN_TASKS" in cfg:
        rows = cfg["TRAIN_TASKS"]
        if not rows or len(rows) < 2:
            raise ValueError("TRAIN_TASKS must list at least 2 rows [task_name, task_config, expert_num].")
        cfgs, nums = [], []
        for i, row in enumerate(rows):
            if not isinstance(row, (list, tuple)) or len(row) != 3:
                raise ValueError(f"TRAIN_TASKS[{i}] must be [task_name, task_config, expert_num], got {row!r}")
            cfgs.append(str(row[1]).strip())
            nums.append(int(row[2]))
        return "__".join(cfgs), int(sum(nums))
    for k in ("TASK_CFG_TO_TRAIN", "TASK_NUM_TO_TRAIN"):
        if k not in cfg:
            raise ValueError(
                "Config must define TRAIN_TASKS or legacy TASK_CFG_TO_TRAIN and TASK_NUM_TO_TRAIN."
            )
    task_cfgs = [str(x).strip() for x in cfg["TASK_CFG_TO_TRAIN"]]
    task_nums = [int(x) for x in cfg["TASK_NUM_TO_TRAIN"]]
    if len(task_cfgs) != len(task_nums):
        raise ValueError("TASK_CFG_TO_TRAIN and TASK_NUM_TO_TRAIN must have same length.")
    return "__".join(task_cfgs), int(sum(task_nums))


def _parse_eval_runs(cfg: dict) -> list:
    """
    @input: [dict, loaded eval YAML]
    @output: [list, [(task_name, task_config), ...]]
    @scenario: [EVAL_TASKS rows or legacy EVAL_TASK / EVAL_TASK_CFG]
    """
    if "EVAL_TASKS" in cfg:
        rows = cfg["EVAL_TASKS"]
        if not rows:
            raise ValueError("EVAL_TASKS must be a non-empty list.")
        out = []
        for i, row in enumerate(rows):
            if not isinstance(row, (list, tuple)) or len(row) != 3:
                raise ValueError(f"EVAL_TASKS[{i}] must be [task_name, task_config, expert_num], got {row!r}")
            out.append((str(row[0]).strip(), str(row[1]).strip()))
        return out
    for k in ("EVAL_TASK", "EVAL_TASK_CFG"):
        if k not in cfg:
            raise ValueError("Config must define EVAL_TASKS or legacy EVAL_TASK and EVAL_TASK_CFG.")
    return [(str(cfg["EVAL_TASK"]).strip(), str(cfg["EVAL_TASK_CFG"]).strip())]


def _load_ev_cfg(act_dir: str, name: str) -> dict:
    """
    @input: [str, act_dir], [str, config name without .yaml]
    @output: [dict, normalized eval config fields present or derivable]
    @scenario: [Load _ev_cfg/<name>.yaml for multi-task evaluation]
    """
    base = name if name.endswith(".yaml") else f"{name}.yaml"
    path = os.path.join(act_dir, "_ev_cfg", base)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No config found: _ev_cfg/{base}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if not cfg:
        raise ValueError("Config file is empty.")
    for k in ("EVAL_SEED", "EVAL_GPU_ID"):
        if k not in cfg:
            raise ValueError(f"Missing required key in config: {k}")
    _parse_train_tasks_for_ckpt(cfg)
    _parse_eval_runs(cfg)
    return cfg


def main(argv: list) -> int:
    parser = argparse.ArgumentParser(description="Multi-task eval wrapper (config under _ev_cfg/*.yaml)")
    parser.add_argument("--config", type=str, required=True, help="Config name (file: _ev_cfg/<name>.yaml)")
    args = parser.parse_args(argv[1:])

    act_dir = os.path.dirname(os.path.abspath(__file__))
    logger = _setup_logger(act_dir)

    try:
        cfg = _load_ev_cfg(act_dir, args.config)
        ckpt_setting, expert_data_num = _parse_train_tasks_for_ckpt(cfg)
        eval_runs = _parse_eval_runs(cfg)
        seed = str(cfg["EVAL_SEED"]).strip()
        gpu_id = str(cfg["EVAL_GPU_ID"]).strip()

        repo_root = os.path.abspath(os.path.join(act_dir, "..", ".."))
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        env["PYTHONWARNINGS"] = "ignore::UserWarning"
        env["PYTHONNOUSERSITE"] = "1"

        for task_name, task_config in eval_runs:
            ckpt_dir = f"policy/ACT/act_ckpt/act-{task_name}/{ckpt_setting}-{expert_data_num}"
            cmd = [
                sys.executable,
                "script/eval_policy.py",
                "--config", "policy/ACT/deploy_policy.yml",
                "--overrides",
                "--task_name", task_name,
                "--task_config", task_config,
                "--ckpt_setting", ckpt_setting,
                "--ckpt_dir", ckpt_dir,
                "--seed", seed,
                "--temporal_agg", "true",
            ]
            logger.info(
                "Eval: task_name=%s task_config=%s ckpt_setting=%s expert_data_num=%s",
                task_name,
                task_config,
                ckpt_setting,
                expert_data_num,
            )
            logger.info("Run: %s", " ".join(cmd))
            subprocess.run(cmd, check=True, env=env, cwd=repo_root)
        return 0
    except Exception as exc:
        logger.error("Wrapper failed: %s", str(exc))
        logger.error("Stack trace:\n%s", traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
