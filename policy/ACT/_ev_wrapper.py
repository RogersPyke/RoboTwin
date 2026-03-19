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


def _load_ev_cfg(act_dir: str, name: str) -> dict:
    """
    @input: [str, act_dir], [str, config name without .yaml]
    @output: [dict, keys TASK_CFG_TO_TRAIN, TASK_NUM_TO_TRAIN, EVAL_TASK, EVAL_TASK_CFG, EVAL_SEED, EVAL_GPU_ID]
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
    required = ("TASK_CFG_TO_TRAIN", "TASK_NUM_TO_TRAIN", "EVAL_TASK", "EVAL_TASK_CFG", "EVAL_SEED", "EVAL_GPU_ID")
    for k in required:
        if k not in cfg:
            raise ValueError(f"Missing required key in config: {k}")
    return cfg


def main(argv: list) -> int:
    parser = argparse.ArgumentParser(description="Multi-task eval wrapper (config under _ev_cfg/*.yaml)")
    parser.add_argument("--config", type=str, required=True, help="Config name (file: _ev_cfg/<name>.yaml)")
    args = parser.parse_args(argv[1:])

    act_dir = os.path.dirname(os.path.abspath(__file__))
    logger = _setup_logger(act_dir)

    try:
        cfg = _load_ev_cfg(act_dir, args.config)
        task_cfgs = [str(x).strip() for x in cfg["TASK_CFG_TO_TRAIN"]]
        task_nums = [int(x) for x in cfg["TASK_NUM_TO_TRAIN"]]
        ckpt_setting = "__".join(task_cfgs)
        expert_data_num = sum(task_nums)
        task_name = str(cfg["EVAL_TASK"]).strip()
        task_config = str(cfg["EVAL_TASK_CFG"]).strip()
        seed = str(cfg["EVAL_SEED"]).strip()
        gpu_id = str(cfg["EVAL_GPU_ID"]).strip()

        repo_root = os.path.abspath(os.path.join(act_dir, "..", ".."))
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
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        env["PYTHONWARNINGS"] = "ignore::UserWarning"

        logger.info("Eval: task_name=%s task_config=%s ckpt_setting=%s expert_data_num=%s", task_name, task_config, ckpt_setting, expert_data_num)
        logger.info("Run: %s", " ".join(cmd))
        subprocess.run(cmd, check=True, env=env, cwd=repo_root)
        return 0
    except Exception as exc:
        logger.error("Wrapper failed: %s", str(exc))
        logger.error("Stack trace:\n%s", traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
