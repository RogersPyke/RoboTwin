#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DP multi-task training wrapper.
Usage: python3 _tr_wrapper.py <cfg_name>
       python3 _tr_wrapper.py --config <cfg_name>
Config file: _tr_cfg/<cfg_name>.yaml
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timedelta, timezone

import numpy as np
import yaml
import zarr


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


def _load_tr_cfg(dp_dir: str, cfg_name: str) -> tuple:
    base = cfg_name if cfg_name.endswith(".yaml") else f"{cfg_name}.yaml"
    cfg_path = os.path.join(dp_dir, "_tr_cfg", base)
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"No config found: _tr_cfg/{base}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict) or not cfg:
        raise ValueError("Config file is empty or invalid.")
    for key in ("TRAIN_TASKS", "TRAIN_SEED", "TRAIN_GPU_ID", "TRAIN_ACTION_DIM"):
        if key not in cfg:
            raise KeyError(f"Missing required key: {key}")
    return cfg, os.path.abspath(cfg_path)


def _parse_task_rows(cfg: dict, key: str) -> list:
    rows = cfg.get(key)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{key} must be a non-empty list of [task_name, task_config, expert_num].")
    out = []
    for i, row in enumerate(rows):
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            raise ValueError(f"{key}[{i}] must be [task_name, task_config, expert_num], got {row!r}")
        task_name = str(row[0]).strip()
        task_config = str(row[1]).strip()
        expert_num = int(row[2])
        if not task_name or not task_config:
            raise ValueError(f"{key}[{i}] has empty task_name/task_config.")
        if expert_num <= 0:
            raise ValueError(f"{key}[{i}] expert_num must be > 0.")
        out.append((task_name, task_config, expert_num))
    return out


def _ensure_single_task_zarr(dp_dir: str, task_name: str, task_config: str, expert_num: int, logger: logging.Logger) -> str:
    rel_path = os.path.join("data", f"{task_name}-{task_config}-{expert_num}.zarr")
    abs_path = os.path.join(dp_dir, rel_path)
    if os.path.isdir(abs_path):
        return rel_path
    logger.info("Single-task zarr missing, building: %s", rel_path)
    cmd = [sys.executable, "process_data.py", task_name, task_config, str(expert_num)]
    subprocess.run(cmd, check=True, cwd=dp_dir, env=os.environ.copy())
    if not os.path.isdir(abs_path):
        raise FileNotFoundError(f"Failed to build expected zarr: {rel_path}")
    return rel_path


def _concat_zarrs(dp_dir: str, rows: list, combined_rel_path: str, logger: logging.Logger) -> None:
    combined_abs_path = os.path.join(dp_dir, combined_rel_path)
    if os.path.isdir(combined_abs_path):
        shutil.rmtree(combined_abs_path)

    src_meta = []
    src_head = []
    src_state = []
    src_action = []
    for task_name, task_config, expert_num in rows:
        src_rel_path = os.path.join("data", f"{task_name}-{task_config}-{expert_num}.zarr")
        src_abs_path = os.path.join(dp_dir, src_rel_path)
        root = zarr.open(src_abs_path, mode="r")
        src_head_arr = root["data"]["head_camera"][:]
        src_state_arr = root["data"]["state"][:]
        src_action_arr = root["data"]["action"][:]
        src_ep_ends = root["meta"]["episode_ends"][:]
        if not (len(src_head_arr) == len(src_state_arr) == len(src_action_arr)):
            raise ValueError(f"Inconsistent data length in {src_rel_path}")
        src_meta.append((task_name, task_config, expert_num, src_rel_path, src_ep_ends.copy()))
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
        offset = int(src_ep_ends[-1])
    all_episode_ends = np.concatenate(episode_ends_list, axis=0).astype(np.int64)

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
    logger.info("Built combined zarr: %s", combined_rel_path)


def main(argv: list) -> int:
    parser = argparse.ArgumentParser(description="DP multi-task train wrapper (_tr_cfg/*.yaml)")
    parser.add_argument("cfg_name", nargs="?", type=str, help="Config name (_tr_cfg/<name>.yaml).")
    parser.add_argument("--config", dest="config", type=str, required=False, help="Legacy cfg arg.")
    args = parser.parse_args(argv[1:])

    dp_dir = os.path.dirname(os.path.abspath(__file__))
    logger = _setup_logger(dp_dir)
    cfg_name = args.config if args.config is not None else args.cfg_name
    if not cfg_name:
        parser.error("Missing cfg_name. Use: python3 _tr_wrapper.py <cfg_name>")

    try:
        cfg, cfg_src_abspath = _load_tr_cfg(dp_dir, cfg_name)
        rows = _parse_task_rows(cfg, "TRAIN_TASKS")
        seed = int(cfg["TRAIN_SEED"])
        gpu_id = str(cfg["TRAIN_GPU_ID"])
        env_gpu = os.environ.get("DP_FLOW_GPU", "").strip()
        if env_gpu:
            gpu_id = env_gpu
        action_dim = int(cfg["TRAIN_ACTION_DIM"])
        head_camera_type = str(cfg.get("TRAIN_HEAD_CAMERA_TYPE", "D435"))
        batch_size = int(cfg.get("TRAIN_BATCH_SIZE", 128))
        num_epochs = int(cfg.get("TRAIN_NUM_EPOCHS", 600))
        checkpoint_every = int(cfg.get("TRAIN_CHECKPOINT_EVERY", 300))
        val_ratio = float(cfg.get("TRAIN_VAL_RATIO", 0.02))
        learning_rate = float(cfg.get("TRAIN_LR", 1.0e-4))
        early_stop_patience_evals = int(cfg.get("EARLY_STOP_PATIENCE_EVALS", 0))
        early_stop_rel_tol = float(cfg.get("EARLY_STOP_REL_TOL", 0.0))
        eval_steps_for_early_stop = int(cfg.get("EVAL_STEPS_FOR_EARLY_STOP", 1))
        delete_combined_zarr_after_train = bool(cfg.get("TRAIN_DELETE_COMBINED_ZARR_AFTER_TRAIN", True))

        task_slug = "__".join([row[0] for row in rows])
        config_slug = "__".join([row[1] for row in rows])
        total_episodes = int(sum([row[2] for row in rows]))
        combined_rel_path = os.path.join("data", f"{task_slug}-{config_slug}-{total_episodes}.zarr")
        combined_abs_path = os.path.join(dp_dir, combined_rel_path)

        for task_name, task_config, expert_num in rows:
            _ensure_single_task_zarr(dp_dir, task_name, task_config, expert_num, logger)
        _concat_zarrs(dp_dir, rows, combined_rel_path, logger)

        ckpt_dir = os.path.join(dp_dir, "checkpoints", f"{task_slug}-{config_slug}-{total_episodes}-{seed}")
        os.makedirs(ckpt_dir, exist_ok=True)
        cfg_basename = os.path.basename(cfg_src_abspath)
        shutil.copy2(cfg_src_abspath, os.path.join(ckpt_dir, cfg_basename))
        with open(os.path.join(ckpt_dir, "training_run_manifest.txt"), "w", encoding="ascii") as mf:
            mf.write(f"training_config_source={cfg_src_abspath}\n")
            mf.write(f"dp_policy_dir={dp_dir}\n")
            mf.write(f"combined_task_slug={task_slug}\n")
            mf.write(f"combined_config_slug={config_slug}\n")
            mf.write(f"combined_total_episodes={total_episodes}\n")
            mf.write(f"combined_zarr={combined_rel_path}\n")
            mf.write(f"copied_yaml={cfg_basename}\n")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        env["PYTHONNOUSERSITE"] = "1"

        cmd = [
            sys.executable,
            "train.py",
            f"--config-name=robot_dp_{action_dim}.yaml",
            f"task.name={task_slug}",
            f"task.dataset.zarr_path={combined_rel_path}",
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
            f"setting={config_slug}",
            f"expert_data_num={total_episodes}",
            f"head_camera_type={head_camera_type}",
        ]
        logger.info("Launch training: %s", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True, cwd=dp_dir, env=env)
        finally:
            if delete_combined_zarr_after_train and os.path.isdir(combined_abs_path):
                shutil.rmtree(combined_abs_path)
                logger.info("Deleted combined zarr after train: %s", combined_rel_path)
        return 0
    except Exception as exc:
        logger.error("Wrapper failed: %s", str(exc))
        logger.error("Stack trace:\n%s", traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
