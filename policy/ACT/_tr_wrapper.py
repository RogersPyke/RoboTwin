#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ACT Train wrapper.

Usage:
    python3 _tr_wrapper.py --task-id <id> --yaml <config.yaml> [--gpu-id N] [--seed N]
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
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def _extract_model_config(
    unified_cfg: Dict[str, Any], model_name: str
) -> Dict[str, Any]:
    global_cfg = unified_cfg.get("global", {})
    model_cfg = unified_cfg.get(model_name, {})
    result = dict(global_cfg)
    for key, value in model_cfg.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _get_task_from_config(
    cfg: Dict[str, Any], task_id: str
) -> Optional[Dict[str, Any]]:
    for task in cfg.get("tr_tasks", []):
        if task.get("task_id") == task_id:
            return task
    return None


def _resolve_data_folder(data_folder: str, robotwin_root: Path) -> Path:
    path = Path(data_folder)
    if path.is_absolute():
        return path
    return robotwin_root / path


def _maybe_add_arg(cmd: list, flag: str, value) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def _extract_step_from_ckpt_name(name: str) -> int:
    m = re.search(r"policy_step_(\d+)\.ckpt$", name)
    if m:
        return int(m.group(1))
    m = re.search(r"policy_epoch_(\d+)_seed_\d+\.ckpt$", name)
    if m:
        return int(m.group(1))
    if name == "policy_best.ckpt":
        return -2
    if name == "policy_last.ckpt":
        return -3
    return -1


def _package_checkpoints_for_eval(ckpt_dir: str, logger: logging.Logger) -> None:
    entries = []
    for name in sorted(os.listdir(ckpt_dir)):
        if not name.endswith(".ckpt"):
            continue
        src_path = os.path.join(ckpt_dir, name)
        if os.path.isfile(src_path):
            entries.append((name, src_path))
    if not entries:
        logger.warning("No checkpoint files found for packaging under %s", ckpt_dir)
        return

    bundle_root = os.path.join(ckpt_dir, "step_packages")
    os.makedirs(bundle_root, exist_ok=True)
    passthrough_files = ["dataset_stats.pkl", "training_run_manifest.txt", "steps.txt"]
    for name, src_ckpt in entries:
        step_id = _extract_step_from_ckpt_name(name)
        if step_id >= 0:
            folder_name = f"step_{step_id}"
        elif step_id == -2:
            folder_name = "step_best"
        elif step_id == -3:
            folder_name = "step_last"
        else:
            folder_name = f"step_misc_{os.path.splitext(name)[0]}"
        dst_dir = os.path.join(bundle_root, folder_name)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy2(src_ckpt, os.path.join(dst_dir, name))
        shutil.copy2(src_ckpt, os.path.join(dst_dir, "policy_best.ckpt"))
        for keep_name in passthrough_files:
            src = os.path.join(ckpt_dir, keep_name)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(dst_dir, keep_name))
    logger.info("Packaged %d checkpoints to %s", len(entries), bundle_root)


def _has_only_symlink_files_recursive(path: str) -> bool:
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


def _run_from_merged_config(
    cfg: Dict[str, Any],
    task_id: str,
    gpu_id: int,
    seed: int,
    logger: logging.Logger,
    act_dir: str,
) -> int:
    task = _get_task_from_config(cfg, task_id)
    if task is None:
        raise ValueError(f"Task not found: {task_id}")

    model_defaults = cfg.get("model_defaults", {})
    params = dict(model_defaults)
    for key in task:
        if key not in ("task_id", "data_folder", "data_sources", "_resolved_data"):
            params[key] = task[key]

    robotwin_root = Path(act_dir).parent.parent

    if "data_folder" in task:
        data_folder = _resolve_data_folder(task["data_folder"], robotwin_root)
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
    elif "data_sources" in task:
        task_names = []
        task_configs = []
        expert_counts = []
        for src in task["data_sources"]:
            task_names.append(src["task_name"])
            task_configs.append(src["task_config"])
            expert_counts.append(src["expert_num"])
    else:
        raise ValueError(f"Task {task_id} must have data_folder or data_sources")

    train_num_epochs = int(params.get("num_epochs", 6000))
    train_save_freq = int(params.get("save_freq", 2000))
    train_batch_size = int(params.get("batch_size", 32))
    train_lr = float(params.get("lr", 1.0e-5))
    train_state_dim = int(params.get("state_dim", 14))
    act_kl_weight = int(params.get("kl_weight", 10))
    act_chunk_size = int(params.get("chunk_size", 50))
    act_hidden_dim = int(params.get("hidden_dim", 512))
    act_dim_feedforward = int(params.get("dim_feedforward", 3200))

    early_stop = cfg.get("early_stop", {})
    early_stop_enabled = bool(early_stop.get("enabled", True))
    early_stop_patience_evals = (
        early_stop.get("patience_evals") if early_stop_enabled else None
    )
    early_stop_rel_tol = early_stop.get("rel_tol") if early_stop_enabled else None
    eval_steps_for_early_stop = (
        early_stop.get("eval_steps") if early_stop_enabled else None
    )
    limits = cfg.get("limits", {})
    max_tr_steps = limits.get("max_tr_steps")
    save_interval = limits.get("save_interval")

    logger.info("Task ID: %s", task_id)
    logger.info("Tasks: %s", task_names)
    logger.info("Configs: %s", task_configs)
    logger.info("Counts: %s", expert_counts)
    logger.info("Seed: %s", seed)
    logger.info("GPU ID: %s", gpu_id)

    combined_task_slug = "__".join(task_names)
    combined_config_slug = "__".join(task_configs)
    combined_total_episodes = int(sum(expert_counts))
    combined_key = (
        f"sim-{combined_task_slug}-{combined_config_slug}-{combined_total_episodes}"
    )
    combined_dataset_dir = f"./processed_data/sim-{combined_task_slug}/{combined_config_slug}-{combined_total_episodes}"

    logger.info("Combined key: %s", combined_key)

    sim_cfg_path = "./SIM_TASK_CONFIGS.json"
    if not os.path.isfile(sim_cfg_path):
        raise FileNotFoundError(
            f"Missing {sim_cfg_path}. Please run process_data.sh first."
        )
    with open(sim_cfg_path, "r") as f:
        sim_task_configs = json.load(f)

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

    if len(task_names) == 1:
        existing_one = sim_task_configs.get(combined_key)
        if existing_one is None:
            raise KeyError(
                f"Missing SIM_TASK_CONFIGS entry for {combined_key}. Run process_data.sh first."
            )
        logger.info(
            "Single-task mode: using existing dataset_dir=%s", combined_dataset_dir
        )
    else:
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

        sim_task_configs[combined_key] = {
            "dataset_dir": combined_dataset_dir,
            "num_episodes": combined_total_episodes,
            "episode_len": episode_len,
            "camera_names": camera_names,
        }
        with open(sim_cfg_path, "w") as f:
            json.dump(sim_task_configs, f, indent=4)

    ckpt_dir = f"./act_ckpt/{task_id}"
    os.makedirs(ckpt_dir, exist_ok=True)
    manifest_path = os.path.join(ckpt_dir, "training_run_manifest.txt")
    with open(manifest_path, "w", encoding="ascii") as mf:
        mf.write(f"task_id={task_id}\n")
        mf.write(f"act_policy_dir={act_dir}\n")
        mf.write(f"combined_task_slug={combined_task_slug}\n")
        mf.write(f"combined_config_slug={combined_config_slug}\n")
        mf.write(f"combined_total_episodes={combined_total_episodes}\n")
    logger.info("Saved training manifest to %s", manifest_path)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    offline_torch_home = os.path.join(act_dir, "torch_offline")
    offline_resnet = os.path.join(
        offline_torch_home, "hub", "checkpoints", "resnet18-f37072fd.pth"
    )
    if os.path.isfile(offline_resnet):
        env["TORCH_HOME"] = offline_torch_home
        logger.info(
            "Offline ResNet18 weights found; TORCH_HOME=%s", offline_torch_home
        )
    cmd = [
        sys.executable,
        "imitate_episodes.py",
        "--task_name",
        combined_key,
        "--ckpt_dir",
        ckpt_dir,
        "--policy_class",
        "ACT",
        "--kl_weight",
        str(act_kl_weight),
        "--chunk_size",
        str(act_chunk_size),
        "--hidden_dim",
        str(act_hidden_dim),
        "--batch_size",
        str(train_batch_size),
        "--dim_feedforward",
        str(act_dim_feedforward),
        "--num_epochs",
        str(train_num_epochs),
        "--lr",
        str(train_lr),
        "--save_freq",
        str(train_save_freq),
        "--state_dim",
        str(train_state_dim),
        "--seed",
        str(seed),
    ]
    _maybe_add_arg(cmd, "--early_stop_rel_tol", early_stop_rel_tol)
    _maybe_add_arg(cmd, "--early_stop_patience_evals", early_stop_patience_evals)
    _maybe_add_arg(cmd, "--eval_steps_for_early_stop", eval_steps_for_early_stop)
    _maybe_add_arg(cmd, "--max_tr_steps", max_tr_steps)
    _maybe_add_arg(cmd, "--save_interval", save_interval)
    logger.info("Launching training: %s", " ".join(cmd))
    sim_task_root_dir = os.path.join("./processed_data", f"sim-{combined_task_slug}")
    try:
        subprocess.run(cmd, check=True, env=env)
        _package_checkpoints_for_eval(ckpt_dir, logger)
    finally:
        if os.path.isdir(sim_task_root_dir) and _has_only_symlink_files_recursive(
            sim_task_root_dir
        ):
            shutil.rmtree(sim_task_root_dir)
            logger.info("Removed symlink-only dataset directory: %s", sim_task_root_dir)
    return 0


def _extract_model_config(
    unified_cfg: Dict[str, Any], model_name: str
) -> Dict[str, Any]:
    """Extract model-specific config from unified config by merging global and model sections."""
    global_cfg = unified_cfg.get("global", {})
    model_cfg = unified_cfg.get(model_name, {})
    result = dict(global_cfg)
    for key, value in model_cfg.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def main(argv: list) -> int:
    parser = argparse.ArgumentParser(description="ACT train wrapper.")
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
    parser.add_argument("--gpu-id", dest="gpu_id", type=int, default=0, help="GPU ID.")
    parser.add_argument(
        "--seed", dest="seed", type=int, default=None, help="Random seed."
    )
    args = parser.parse_args(argv[1:])

    act_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(act_dir)
    logger = _setup_logger(act_dir)

    try:
        unified_cfg = _load_yaml(args.yaml_path)
        cfg = _extract_model_config(unified_cfg, "ACT")
        seed = args.seed if args.seed is not None else cfg.get("seed", 0)
        return _run_from_merged_config(
            cfg, args.task_id, args.gpu_id, seed, logger, act_dir
        )
    except Exception as exc:
        logger.error("Wrapper failed: %s", str(exc))
        logger.error("Stack trace:\n%s", traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
