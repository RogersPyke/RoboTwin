#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DP multi-task eval wrapper.
Usage: python3 _ev_wrapper.py <cfg_name>
       python3 _ev_wrapper.py --config <cfg_name>
Config file: _ev_cfg/<cfg_name>.yaml
"""

import argparse
import logging
import os
import subprocess
import sys
import traceback
from datetime import datetime, timedelta, timezone

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


def _load_ev_cfg(dp_dir: str, cfg_name: str) -> dict:
    base = cfg_name if cfg_name.endswith(".yaml") else f"{cfg_name}.yaml"
    cfg_path = os.path.join(dp_dir, "_ev_cfg", base)
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"No config found: _ev_cfg/{base}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict) or not cfg:
        raise ValueError("Config file is empty or invalid.")
    for key in ("TRAIN_TASKS", "EVAL_TASKS", "EVAL_SEED", "EVAL_GPU_ID", "END_RESET_TO_INIT"):
        if key not in cfg:
            raise KeyError(f"Missing required key: {key}")
    return cfg


def _to_cli_bool(v) -> str:
    if isinstance(v, str):
        return "true" if v.strip().lower() in ("1", "true", "yes", "y", "on") else "false"
    return "true" if bool(v) else "false"


def _resolve_eval_force_end_reset_to_init(cfg: dict) -> str:
    return _to_cli_bool(cfg["END_RESET_TO_INIT"])


def _parse_task_rows(cfg: dict, key: str) -> list:
    rows = cfg.get(key)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{key} must be a non-empty list of [task_name, task_config, expert_num].")
    out = []
    for i, row in enumerate(rows):
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            raise ValueError(f"{key}[{i}] must be [task_name, task_config, expert_num], got {row!r}")
        out.append((str(row[0]).strip(), str(row[1]).strip(), int(row[2])))
    return out


def main(argv: list) -> int:
    parser = argparse.ArgumentParser(description="DP multi-task eval wrapper (_ev_cfg/*.yaml)")
    parser.add_argument("cfg_name", nargs="?", type=str, help="Config name (_ev_cfg/<name>.yaml).")
    parser.add_argument("--config", dest="config", type=str, required=False, help="Legacy cfg arg.")
    args = parser.parse_args(argv[1:])

    dp_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(dp_dir, "..", ".."))
    logger = _setup_logger(dp_dir)
    cfg_name = args.config if args.config is not None else args.cfg_name
    if not cfg_name:
        parser.error("Missing cfg_name. Use: python3 _ev_wrapper.py <cfg_name>")

    try:
        cfg = _load_ev_cfg(dp_dir, cfg_name)
        train_rows = _parse_task_rows(cfg, "TRAIN_TASKS")
        eval_rows = _parse_task_rows(cfg, "EVAL_TASKS")
        seed = int(cfg["EVAL_SEED"])
        gpu_id = str(cfg["EVAL_GPU_ID"])
        env_gpu = os.environ.get("DP_FLOW_GPU", "").strip()
        if env_gpu:
            gpu_id = env_gpu
        checkpoint_num = int(cfg.get("CHECKPOINT_NUM", 600))
        head_camera_type = str(cfg.get("EVAL_HEAD_CAMERA_TYPE", "D435"))
        test_num = int(cfg.get("TEST_NUM", 100))
        if test_num < 1:
            raise ValueError("TEST_NUM must be >= 1")
        force_end_reset_to_init = _resolve_eval_force_end_reset_to_init(cfg)

        train_task_slug = "__".join([row[0] for row in train_rows])
        train_config_slug = "__".join([row[1] for row in train_rows])
        train_total_episodes = int(sum([row[2] for row in train_rows]))
        checkpoint_expert_data_num = int(cfg.get("CHECKPOINT_EXPERT_DATA_NUM", train_total_episodes))

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        env["PYTHONWARNINGS"] = "ignore::UserWarning"
        env["PYTHONNOUSERSITE"] = "1"
        conda_prefix = str(env.get("CONDA_PREFIX", "")).strip()
        if conda_prefix:
            conda_lib = os.path.join(conda_prefix, "lib")
            prev_ld = str(env.get("LD_LIBRARY_PATH", ""))
            env["LD_LIBRARY_PATH"] = conda_lib if not prev_ld else f"{conda_lib}:{prev_ld}"
        base = cfg_name if cfg_name.endswith(".yaml") else f"{cfg_name}.yaml"
        env["ACT_EV_CFG_SNAPSHOT_SRC"] = os.path.abspath(os.path.join(dp_dir, "_ev_cfg", base))

        for task_name, task_config, expert_num in eval_rows:
            cmd = [
                sys.executable,
                "script/eval_policy.py",
                "--config",
                "policy/DP/deploy_policy.yml",
                "--overrides",
                "--task_name",
                task_name,
                "--task_config",
                task_config,
                "--ckpt_setting",
                train_config_slug,
                "--expert_data_num",
                str(checkpoint_expert_data_num),
                "--seed",
                str(seed),
                "--checkpoint_num",
                str(checkpoint_num),
                "--head_camera_type",
                head_camera_type,
                "--train_task_name",
                train_task_slug,
                "--eval_expert_data_num",
                str(expert_num),
                "--test_num",
                str(test_num),
                "--force_end_reset_to_init",
                force_end_reset_to_init,
            ]
            logger.info(
                "Eval task=%s config=%s ckpt=%s/%s test_num=%s force_end_reset_to_init=%s",
                task_name,
                task_config,
                train_task_slug,
                train_config_slug,
                test_num,
                force_end_reset_to_init,
            )
            logger.info("Run: %s", " ".join(cmd))
            subprocess.run(cmd, check=True, cwd=repo_root, env=env)
        return 0
    except Exception as exc:
        logger.error("Wrapper failed: %s", str(exc))
        logger.error("Stack trace:\n%s", traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
