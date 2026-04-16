#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DP Train wrapper.

Usage (new multi-YAML mode):
    python3 _tr_wrapper.py --task-id <id> --yaml <shared.yaml> --yaml <model.yaml> [--gpu-id N] [--seed N]

Usage (legacy single-YAML mode):
    python3 _tr_wrapper.py <cfg_name>
    python3 _tr_wrapper.py --config <cfg_name>
"""

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    for task in cfg.get("tr_tasks", []):
        if task.get("task_id") == task_id:
            return task
    return None


def _extract_step_from_ckpt_name(name: str) -> int:
    m = re.search(r"step_(\d+)\.ckpt$", name)
    if m:
        return int(m.group(1))
    m = re.search(r"^(\d+)\.ckpt$", name)
    if m:
        return int(m.group(1))
    return -1


def _package_checkpoints_for_eval(ckpt_dir: str, logger: logging.Logger) -> None:
    if not os.path.isdir(ckpt_dir):
        return
    ckpts = []
    for name in sorted(os.listdir(ckpt_dir)):
        if not name.endswith(".ckpt"):
            continue
        src = os.path.join(ckpt_dir, name)
        if os.path.isfile(src):
            ckpts.append((name, src))
    if not ckpts:
        return
    bundle_root = os.path.join(ckpt_dir, "step_packages")
    os.makedirs(bundle_root, exist_ok=True)
    passthrough_files = ["training_run_manifest.txt", "steps.txt"]
    for name, src_ckpt in ckpts:
        step_id = _extract_step_from_ckpt_name(name)
        if step_id >= 0:
            folder_name = f"step_{step_id}"
        else:
            folder_name = f"step_misc_{os.path.splitext(name)[0]}"
        dst_dir = os.path.join(bundle_root, folder_name)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy2(src_ckpt, os.path.join(dst_dir, name))
        shutil.copy2(src_ckpt, os.path.join(dst_dir, "policy_best.ckpt"))
        for keep in passthrough_files:
            src_keep = os.path.join(ckpt_dir, keep)
            if os.path.isfile(src_keep):
                shutil.copy2(src_keep, os.path.join(dst_dir, keep))
    logger.info("Packaged %d checkpoints to %s", len(ckpts), bundle_root)


def _find_workspace_checkpoint_dir(dp_dir: str, save_name: str, seed: int) -> str:
    target_leaf = f"{save_name}-{seed}"
    outputs_root = os.path.join(dp_dir, "data", "outputs")
    if not os.path.isdir(outputs_root):
        return ""
    found = []
    for root, dir_names, _ in os.walk(outputs_root):
        for dir_name in dir_names:
            if dir_name == target_leaf and os.path.basename(root) == "checkpoints":
                found.append(os.path.join(root, dir_name))
    if not found:
        return ""
    found.sort(key=os.path.getmtime, reverse=True)
    return found[0]


def _ensure_single_task_zarr(
    dp_dir: str,
    task_name: str,
    task_config: str,
    expert_num: int,
    logger: logging.Logger,
) -> str:
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


def _resolve_src_zarr_paths(dp_dir: str, rows: list, logger: logging.Logger) -> list:
    src_paths = []
    for task_name, task_config, expert_num in rows:
        src_rel_path = _ensure_single_task_zarr(
            dp_dir, task_name, task_config, expert_num, logger
        )
        src_abs_path = os.path.join(dp_dir, src_rel_path)
        if not os.path.isdir(src_abs_path):
            raise FileNotFoundError(f"Source zarr not found: {src_rel_path}")
        src_paths.append(
            (task_name, task_config, expert_num, src_rel_path, src_abs_path)
        )
    return src_paths


def _concat_zarrs_from_src_paths(
    src_paths: list,
    combined_abs_path: str,
    combined_rel_path: str,
    logger: logging.Logger,
) -> None:
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
        offset = int(src_ep_ends[-1])
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
    logger.info("Built combined zarr: %s", combined_rel_path)


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
    params = dict(model_defaults)
    for key in task:
        if key not in ("task_id", "data_folder", "data_sources", "_resolved_data"):
            params[key] = task[key]

    if "data_folder" in task:
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
    elif "data_sources" in task:
        task_names = [src["task_name"] for src in task["data_sources"]]
        task_configs = [src["task_config"] for src in task["data_sources"]]
        expert_counts = [src["expert_num"] for src in task["data_sources"]]
    else:
        raise ValueError(f"Task {task_id} must have data_folder or data_sources")

    rows = list(zip(task_names, task_configs, expert_counts))

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

    logger.info("Task ID: %s", task_id)
    logger.info("Tasks: %s", task_names)
    logger.info("Seed: %s", seed)
    logger.info("GPU ID: %s", gpu_id)

    task_slug = "__".join(task_names)
    config_slug = "__".join(task_configs)
    total_episodes = int(sum(expert_counts))
    combined_rel_path = os.path.join(
        "data", f"{task_slug}-{config_slug}-{total_episodes}.zarr"
    )
    combined_abs_path = os.path.join(dp_dir, combined_rel_path)

    src_paths = _resolve_src_zarr_paths(dp_dir, rows, logger)
    _concat_zarrs_from_src_paths(
        src_paths, combined_abs_path, combined_rel_path, logger
    )

    ckpt_dir = os.path.join(dp_dir, "checkpoints", task_id)
    os.makedirs(ckpt_dir, exist_ok=True)
    manifest_path = os.path.join(ckpt_dir, "training_run_manifest.txt")
    with open(manifest_path, "w", encoding="ascii") as mf:
        mf.write(f"task_id={task_id}\n")
        mf.write(f"dp_policy_dir={dp_dir}\n")
        mf.write(f"combined_task_slug={task_slug}\n")
        mf.write(f"combined_config_slug={config_slug}\n")
        mf.write(f"combined_total_episodes={total_episodes}\n")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
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
    logger.info("Launch training: %s", " ".join(cmd))
    save_name = os.path.splitext(os.path.basename(combined_rel_path))[0]
    train_succeeded = False
    try:
        subprocess.run(cmd, check=True, cwd=dp_dir, env=env)
        train_succeeded = True
        workspace_ckpt_dir = _find_workspace_checkpoint_dir(dp_dir, save_name, seed)
        if workspace_ckpt_dir:
            _package_checkpoints_for_eval(workspace_ckpt_dir, logger)
    finally:
        if not train_succeeded and os.path.isfile(manifest_path):
            os.remove(manifest_path)
            logger.info("Removed manifest after failed training: %s", manifest_path)
        if os.path.isdir(combined_abs_path):
            shutil.rmtree(combined_abs_path)
            logger.info("Deleted combined zarr: %s", combined_rel_path)
    return 0


def main(argv: list) -> int:
    parser = argparse.ArgumentParser(description="DP train wrapper.")
    parser.add_argument(
        "--task-id", dest="task_id", type=str, help="Task ID from tr_tasks.yaml."
    )
    parser.add_argument(
        "--yaml",
        dest="yaml_paths",
        action="append",
        type=str,
        help="YAML config path (can specify multiple).",
    )
    parser.add_argument("--gpu-id", dest="gpu_id", type=int, default=0, help="GPU ID.")
    parser.add_argument(
        "--seed", dest="seed", type=int, default=None, help="Random seed."
    )
    parser.add_argument("cfg_name", nargs="?", type=str, help="(legacy) Config name.")
    parser.add_argument(
        "--config", dest="config", type=str, help="(legacy) Config name."
    )
    args = parser.parse_args(argv[1:])

    dp_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(dp_dir)
    logger = _setup_logger(dp_dir)

    try:
        if args.yaml_paths and args.task_id:
            cfg = _merge_yamls(args.yaml_paths)
            seed = args.seed if args.seed is not None else cfg.get("seed", 0)
            gpu_id = args.gpu_id
            return _run_from_merged_config(
                cfg, args.task_id, gpu_id, seed, logger, dp_dir
            )

        cfg_name = args.config or args.cfg_name
        if cfg_name:
            cfg_path = os.path.join(dp_dir, "_tr_cfg", f"{cfg_name}.yaml")
            if not os.path.isfile(cfg_path):
                cfg_path = os.path.join(dp_dir, "_tr_cfg", cfg_name)
            cfg = _load_yaml(cfg_path)
            task_id = cfg.get("tr_tasks", [{}])[0].get("task_id", cfg_name)
            seed = args.seed if args.seed is not None else cfg.get("seed", 0)
            gpu_id = args.gpu_id
            return _run_from_merged_config(cfg, task_id, gpu_id, seed, logger, dp_dir)

        parser.error("Use --task-id + --yaml, or provide cfg_name.")
    except Exception as exc:
        logger.error("Wrapper failed: %s", str(exc))
        logger.error("Stack trace:\n%s", traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
