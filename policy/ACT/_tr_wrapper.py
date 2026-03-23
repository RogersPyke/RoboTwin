#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-task train wrapper: reads _tr_cfg/<name>.yaml, builds combined dataset, trains from scratch.
Usage: python3 _tr_wrapper.py --config <name>
Config name is without extension; file must be _tr_cfg/<name>.yaml.
"""
import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone, timedelta

import yaml


def _setup_logger(act_dir: str) -> logging.Logger:
    """
    @input: [str, act_dir, any non-empty path]
    @output: [logging.Logger, logger instance]
    @scenario: [Provide consistent file+console logging for wrapper runs]
    """
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


def _parse_train_tasks_rows(cfg: dict) -> tuple:
    """
    @input: [dict, raw YAML cfg with TRAIN_TASKS]
    @output: [tuple, (task_names, task_configs, expert_counts) lists]
    @scenario: [Parse TRAIN_TASKS list of [name, cfg, num] for multi-task training]
    """
    if "TRAIN_TASKS" not in cfg:
        raise ValueError("Config must define TRAIN_TASKS as a list of [task_name, task_config, expert_num].")
    rows = cfg["TRAIN_TASKS"]
    if not rows or len(rows) < 2:
        raise ValueError("TRAIN_TASKS must list at least 2 rows [task_name, task_config, expert_num].")
    names, cfgs, nums = [], [], []
    for i, row in enumerate(rows):
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            raise ValueError(f"TRAIN_TASKS[{i}] must be [task_name, task_config, expert_num], got {row!r}")
        names.append(str(row[0]).strip())
        cfgs.append(str(row[1]).strip())
        nums.append(int(row[2]))
    return names, cfgs, nums


def _load_tr_cfg(act_dir: str, name: str) -> tuple:
    """
    @input: [str, act_dir], [str, config name without .yaml]
    @output: [tuple, (dict cfg, str absolute path to yaml file)]
    @scenario: [Load _tr_cfg/<name>.yaml for multi-task training]
    """
    base = name if name.endswith(".yaml") else f"{name}.yaml"
    path = os.path.join(act_dir, "_tr_cfg", base)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No config found: _tr_cfg/{base}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if not cfg:
        raise ValueError("Config file is empty.")
    for k in ("TRAIN_SEED", "TRAIN_GPU_ID"):
        if k not in cfg:
            raise ValueError(f"Missing required key in config: {k}")
    _parse_train_tasks_rows(cfg)
    return cfg, os.path.abspath(path)


def main(argv: list) -> int:
    """
    @input: [List[str], argv with --config <name>]
    @output: [int, 0 on success else non-zero]
    @scenario: [Load _tr_cfg config, link combined dataset episodes, update SIM_TASK_CONFIGS, train from scratch]
    """
    parser = argparse.ArgumentParser(description="Multi-task train wrapper (config under _tr_cfg/*.yaml)")
    parser.add_argument("--config", type=str, required=True, help="Config name (file: _tr_cfg/<name>.yaml)")
    args = parser.parse_args(argv[1:])

    act_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(act_dir)
    logger = _setup_logger(act_dir)

    try:
        cfg, cfg_src_abspath = _load_tr_cfg(act_dir, args.config)
        task_names, task_configs, expert_counts = _parse_train_tasks_rows(cfg)
        global_seed = int(cfg["TRAIN_SEED"])
        gpu_id = str(cfg["TRAIN_GPU_ID"])

        combined_task_slug = "__".join(task_names)
        combined_config_slug = "__".join(task_configs)
        combined_total_episodes = int(sum(expert_counts))
        combined_key = f"sim-{combined_task_slug}-{combined_config_slug}-{combined_total_episodes}"
        combined_dataset_dir = f"./processed_data/sim-{combined_task_slug}/{combined_config_slug}-{combined_total_episodes}"

        logger.info("Tasks: %s", task_names)
        logger.info("Configs: %s", task_configs)
        logger.info("Counts: %s", expert_counts)
        logger.info("Combined key: %s", combined_key)
        logger.info("Global training seed: %s", global_seed)

        sim_cfg_path = "./SIM_TASK_CONFIGS.json"
        if not os.path.isfile(sim_cfg_path):
            raise FileNotFoundError(f"Missing {sim_cfg_path}. Please run process_data.sh first.")
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
                raise KeyError(f"Missing SIM_TASK_CONFIGS entry for {sub_key}. Run process_data.sh for this subtask first.")
            sub_entry = sim_task_configs[sub_key]
            if episode_len is None:
                episode_len = sub_entry["episode_len"]
                camera_names = sub_entry["camera_names"]
            else:
                if sub_entry["episode_len"] != episode_len or sub_entry["camera_names"] != camera_names:
                    raise ValueError("All subtasks must share episode_len and camera_names to be combined.")
            src_dir = sub_entry["dataset_dir"]
            for j in range(expert_counts[i]):
                src_ep = os.path.join(src_dir, f"episode_{j}.hdf5")
                if not os.path.isfile(src_ep):
                    raise FileNotFoundError(f"Missing episode file: {src_ep}")

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
                    raise FileExistsError(f"Destination exists and is not a symlink: {dst_ep}")
                rel_src_ep = os.path.relpath(src_ep, start=os.path.dirname(dst_ep))
                os.symlink(rel_src_ep, dst_ep)

        existing = sim_task_configs.get(combined_key)
        if existing is not None:
            if (
                existing.get("episode_len") != episode_len
                or existing.get("camera_names") != camera_names
                or existing.get("num_episodes") != combined_total_episodes
            ):
                raise ValueError(f"Conflicting existing SIM_TASK_CONFIGS entry for {combined_key}.")
        sim_task_configs[combined_key] = {
            "dataset_dir": combined_dataset_dir,
            "num_episodes": combined_total_episodes,
            "episode_len": episode_len,
            "camera_names": camera_names,
        }
        with open(sim_cfg_path, "w") as f:
            json.dump(sim_task_configs, f, indent=4)

        ckpt_dir = f"./act_ckpt/act-{combined_task_slug}/{combined_config_slug}-{combined_total_episodes}"
        os.makedirs(ckpt_dir, exist_ok=True)
        cfg_basename = os.path.basename(cfg_src_abspath)
        dst_cfg = os.path.join(ckpt_dir, cfg_basename)
        shutil.copy2(cfg_src_abspath, dst_cfg)
        manifest_path = os.path.join(ckpt_dir, "training_run_manifest.txt")
        with open(manifest_path, "w", encoding="ascii") as mf:
            mf.write("training_config_source=%s\n" % cfg_src_abspath)
            mf.write("act_policy_dir=%s\n" % act_dir)
            mf.write("combined_task_slug=%s\n" % combined_task_slug)
            mf.write("combined_config_slug=%s\n" % combined_config_slug)
            mf.write("combined_total_episodes=%s\n" % combined_total_episodes)
            mf.write("copied_yaml=%s\n" % cfg_basename)
        logger.info("Saved training config copy to %s and %s", dst_cfg, manifest_path)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["PYTHONNOUSERSITE"] = "1"
        cmd = [
            "python3",
            "imitate_episodes.py",
            "--task_name", combined_key,
            "--ckpt_dir", ckpt_dir,
            "--policy_class", "ACT",
            "--kl_weight", "10",
            "--chunk_size", "50",
            "--hidden_dim", "512",
            "--batch_size", "8",
            "--dim_feedforward", "3200",
            "--num_epochs", "6000",
            "--lr", "1e-5",
            "--save_freq", "2000",
            "--state_dim", "14",
            "--seed", str(global_seed),
        ]
        logger.info("Launching training: %s", " ".join(cmd))
        subprocess.run(cmd, check=True, env=env)
        return 0
    except Exception as exc:
        logger.error("Wrapper failed: %s", str(exc))
        logger.error("Stack trace:\n%s", traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
