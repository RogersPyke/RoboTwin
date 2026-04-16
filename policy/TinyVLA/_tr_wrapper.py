#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TinyVLA Train wrapper.

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

import yaml


LOGGER: Optional[logging.Logger] = None


def _setup_logger(tinyvla_dir: str) -> logging.Logger:
    global LOGGER
    logs_dir = os.path.join(tinyvla_dir, "logs")
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
    LOGGER = logger
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


def _build_argv_from_args_dict(
    args_dict: Dict[str, Any], override_seed: Optional[int] = None
) -> List[str]:
    out: List[str] = []
    for k, v in args_dict.items():
        if v is None:
            continue
        flag = str(k)
        if flag.startswith("--"):
            flag = flag[2:]
        if not flag:
            continue
        out.append(f"--{flag}")
        if isinstance(v, bool):
            out.append("True" if v else "False")
        else:
            out.append(str(v))
    if override_seed is not None:
        filtered: List[str] = []
        i = 0
        while i < len(out):
            if out[i] == "--seed":
                i += 2
                continue
            filtered.append(out[i])
            i += 1
        out = filtered
        out.extend(["--seed", str(override_seed)])
    return out


def _extract_step_from_tinyvla_name(name: str) -> int:
    m = re.search(r"checkpoint-(\d+)$", name)
    if m:
        return int(m.group(1))
    m = re.search(r"step_(\d+)$", name)
    if m:
        return int(m.group(1))
    return -1


def _package_checkpoints_for_eval(output_dir: str) -> None:
    if not os.path.isdir(output_dir):
        return
    candidates = []
    for name in sorted(os.listdir(output_dir)):
        path = os.path.join(output_dir, name)
        if name.startswith("checkpoint-") and os.path.isdir(path):
            candidates.append((name, path))
        elif name == "policy_best" and os.path.isdir(path):
            candidates.append((name, path))
    if not candidates:
        return
    bundle_root = os.path.join(output_dir, "step_packages")
    os.makedirs(bundle_root, exist_ok=True)
    passthrough_files = [
        "training_run_manifest.txt",
        "steps.txt",
        "joint_task_spec.json",
    ]
    for name, src_path in candidates:
        step_id = _extract_step_from_tinyvla_name(name)
        if step_id >= 0:
            folder_name = f"step_{step_id}"
        elif name == "policy_best":
            folder_name = "step_best"
        else:
            folder_name = f"step_misc_{name.replace('/', '_')}"
        dst_dir = os.path.join(bundle_root, folder_name)
        os.makedirs(dst_dir, exist_ok=True)
        model_dir = os.path.join(dst_dir, "model")
        shutil.copytree(src_path, model_dir, dirs_exist_ok=True)
        for keep in passthrough_files:
            src_keep = os.path.join(output_dir, keep)
            if os.path.isfile(src_keep):
                shutil.copy2(src_keep, os.path.join(dst_dir, keep))
    if LOGGER is not None:
        LOGGER.info(
            "Packaged %d TinyVLA checkpoints to %s", len(candidates), bundle_root
        )


def _run_from_merged_config(
    cfg: Dict[str, Any],
    task_id: str,
    gpu_id: int,
    seed: int,
    logger: logging.Logger,
    tinyvla_dir: str,
) -> int:
    task = _get_task_from_config(cfg, task_id)
    if task is None:
        raise ValueError(f"Task not found: {task_id}")

    model_defaults = cfg.get("model_defaults", {})
    params = dict(model_defaults)
    for key in task:
        if key not in (
            "task_id",
            "data_folder",
            "data_sources",
            "_resolved_data",
            "task_name",
            "task_config",
            "expert_num",
        ):
            params[key] = task[key]

    if "data_folder" in task:
        task_names = [task.get("task_name", task_id)]
        task_configs = [task.get("task_config", "demo_clean")]
        expert_counts = [task.get("expert_num", 100)]
    elif "data_sources" in task:
        task_names = [src["task_name"] for src in task["data_sources"]]
        task_configs = [src["task_config"] for src in task["data_sources"]]
        expert_counts = [src["expert_num"] for src in task["data_sources"]]
    else:
        raise ValueError(f"Task {task_id} must have data_folder or data_sources")

    combined_task_slug = "__".join(task_names)
    combined_config_slug = "__".join(task_configs)
    combined_total_episodes = int(sum(expert_counts))

    dataset_dirs: List[str] = []
    if "data_folder" in task:
        data_folder = Path(task["data_folder"])
        if not data_folder.is_absolute():
            robotwin_root = Path(tinyvla_dir).parent.parent
            data_folder = robotwin_root / data_folder
        dataset_dirs.append(str(data_folder))
    elif "data_sources" in task:
        robotwin_root = Path(tinyvla_dir).parent.parent
        for src in task["data_sources"]:
            data_folder = Path(src["data_folder"])
            if not data_folder.is_absolute():
                data_folder = robotwin_root / data_folder
            dataset_dirs.append(str(data_folder))

    joint_task_spec = {
        "task_name": combined_task_slug,
        "dataset_dir": dataset_dirs,
        "camera_names": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
        "episode_len": 0,
    }

    output_dir = os.path.join(tinyvla_dir, "tinyvla_ckpt", task_id)

    train_args: Dict[str, Any] = dict(params)
    train_args["task_name"] = combined_task_slug
    train_args["output_dir"] = output_dir
    train_args.setdefault("logging_dir", os.path.join(output_dir, "log"))
    train_args["joint_task_spec"] = json.dumps(joint_task_spec)

    early_stop = cfg.get("early_stop", {})
    early_stop_enabled = bool(early_stop.get("enabled", True))
    if early_stop_enabled:
        if early_stop.get("eval_steps") is not None:
            train_args["eval_steps_for_early_stop"] = early_stop["eval_steps"]
        if early_stop.get("patience_evals") is not None:
            train_args["early_stop_patience_evals"] = early_stop["patience_evals"]
        if early_stop.get("rel_tol") is not None:
            train_args["early_stop_rel_tol"] = early_stop["rel_tol"]
    else:
        train_args["early_stop_patience_evals"] = 0
        train_args["early_stop_rel_tol"] = 0.0

    limits = cfg.get("limits", {})
    if limits.get("max_tr_steps"):
        train_args["max_steps"] = limits["max_tr_steps"]
    if limits.get("save_interval"):
        train_args["save_steps"] = limits["save_interval"]
        train_args["save_strategy"] = "steps"

    deepspeed_cfg = cfg.get("deepspeed", {})
    use_deepspeed = deepspeed_cfg.get("enabled", False)

    if use_deepspeed:
        num_gpus = int(deepspeed_cfg.get("num_gpus", 1))
        master_port = int(deepspeed_cfg.get("master_port", 29604))
        zero2_json = str(deepspeed_cfg.get("zero2_json", "scripts/zero2.json"))
        train_args["deepspeed"] = zero2_json

    logger.info("Task ID: %s", task_id)
    logger.info("Tasks: %s", task_names)
    logger.info("Seed: %s", seed)
    logger.info("GPU ID: %s", gpu_id)
    logger.info("Output dir: %s", output_dir)

    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "training_run_manifest.txt")
    with open(manifest_path, "w", encoding="ascii") as mf:
        mf.write(f"task_id={task_id}\n")
        mf.write(f"tinyvla_policy_dir={tinyvla_dir}\n")
        mf.write(f"combined_task_slug={combined_task_slug}\n")
        mf.write(f"combined_config_slug={combined_config_slug}\n")
        mf.write(f"combined_total_episodes={combined_total_episodes}\n")

    argv_tokens = _build_argv_from_args_dict(train_args, override_seed=seed)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONNOUSERSITE"] = "1"

    if use_deepspeed:
        cmd = [
            "deepspeed",
            "--master_port",
            str(master_port),
            f"--num_gpus={num_gpus}",
            "--num_nodes=1",
            "./train_vla.py",
        ] + argv_tokens
    else:
        cmd = [sys.executable, "./train_vla.py"] + argv_tokens

    logger.info("Launching training: %s", " ".join(cmd))
    train_succeeded = False
    try:
        subprocess.run(cmd, check=True, env=env, cwd=tinyvla_dir)
        train_succeeded = True
        _package_checkpoints_for_eval(output_dir)
    finally:
        if not train_succeeded and os.path.isfile(manifest_path):
            os.remove(manifest_path)
            logger.info("Removed manifest after failed training: %s", manifest_path)
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
    parser = argparse.ArgumentParser(description="TinyVLA train wrapper.")
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

    tinyvla_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(tinyvla_dir)
    logger = _setup_logger(tinyvla_dir)

    try:
        if args.yaml_paths and args.task_id:
            unified_cfg = _merge_yamls(args.yaml_paths)
            cfg = _extract_model_config(unified_cfg, "TinyVLA")
            seed = args.seed if args.seed is not None else cfg.get("seed", 0)
            gpu_id = args.gpu_id
            return _run_from_merged_config(
                cfg, args.task_id, gpu_id, seed, logger, tinyvla_dir
            )

        cfg_name = args.config or args.cfg_name
        if cfg_name:
            cfg_path = os.path.join(tinyvla_dir, "_tr_cfg", f"{cfg_name}.yaml")
            if not os.path.isfile(cfg_path):
                cfg_path = os.path.join(tinyvla_dir, "_tr_cfg", cfg_name)
            cfg = _load_yaml(cfg_path)
            task_id = cfg.get("tr_tasks", [{}])[0].get("task_id", cfg_name)
            seed = args.seed if args.seed is not None else cfg.get("seed", 0)
            gpu_id = args.gpu_id
            return _run_from_merged_config(
                cfg, task_id, gpu_id, seed, logger, tinyvla_dir
            )

        parser.error("Use --task-id + --yaml, or provide cfg_name.")
    except Exception as exc:
        logger.error("Wrapper failed: %s", str(exc))
        logger.error("Stack trace:\n%s", traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
