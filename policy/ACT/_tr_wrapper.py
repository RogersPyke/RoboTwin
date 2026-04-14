#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train wrapper: reads _tr_cfg/<name>.yaml, builds combined dataset (1+ tasks), trains from scratch.
Usage: python3 _tr_wrapper.py <cfg_name>
       python3 _tr_wrapper.py --config <cfg_name>   # (legacy)
Config name is without extension; file must be _tr_cfg/<name>.yaml.
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


def _get_cfg_opt(cfg: dict, key: str, default):
    """
    @input: [dict, cfg], [str, key], [Any, default]
    @output: [Any, value or default]
    @scenario: [Fetch optional config keys with backward-compatible defaults]
    """
    if key not in cfg:
        return default
    return cfg.get(key, default)


def _maybe_add_arg(cmd: list, flag: str, value) -> None:
    """
    @input: [list, cmd argv list], [str, flag like '--x'], [Any, value]
    @output: [None]
    @scenario: [Append optional CLI flag when value is not None]
    """
    if value is None:
        return
    cmd.extend([flag, str(value)])


def _extract_step_from_ckpt_name(name: str) -> int:
    """
    @input: [str, checkpoint filename]
    @output: [int, parsed step number or -1]
    @scenario: [Infer training step for checkpoint package folder naming]
    """
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
    """
    @input: [str, ckpt_dir], [logging.Logger, logger]
    @output: [None]
    @scenario: [Pack each checkpoint into an isolated eval-ready directory]
    """
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
    passthrough_files = [
        "dataset_stats.pkl",
        "training_run_manifest.txt",
        "steps.txt",
    ]
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
        for file_name in os.listdir(ckpt_dir):
            if file_name.endswith(".yaml"):
                src_yaml = os.path.join(ckpt_dir, file_name)
                if os.path.isfile(src_yaml):
                    shutil.copy2(src_yaml, os.path.join(dst_dir, file_name))
    logger.info("Packaged %d checkpoints to %s", len(entries), bundle_root)


def _has_only_symlink_files_recursive(path: str) -> bool:
    """
    @input: [str, path]
    @output: [bool, True if all files in tree are symlinks]
    @scenario: [Safety check before recursively removing wrapper-managed sim-* directory]
    """
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
                # Symlinked directories count as symlink entries.
                continue
    return True


def _parse_train_tasks_rows(cfg: dict) -> tuple:
    """
    @input: [dict, raw YAML cfg with TRAIN_TASKS]
    @output: [tuple, (task_names, task_configs, expert_counts) lists]
    @scenario: [Parse TRAIN_TASKS list of [name, cfg, num] for single- or multi-task training]
    """
    if "TRAIN_TASKS" not in cfg:
        raise ValueError("Config must define TRAIN_TASKS as a list of [task_name, task_config, expert_num].")
    rows = cfg["TRAIN_TASKS"]
    if not rows or len(rows) < 1:
        raise ValueError("TRAIN_TASKS must list at least 1 row [task_name, task_config, expert_num].")
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
    @scenario: [Load _tr_cfg/<name>.yaml for training]
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
    @input: [List[str], argv with <cfg_name> or --config <cfg_name>]
    @output: [int, 0 on success else non-zero]
    @scenario: [Load _tr_cfg config, link combined dataset episodes, update SIM_TASK_CONFIGS, train from scratch]
    """
    parser = argparse.ArgumentParser(description="Train wrapper (config under _tr_cfg/*.yaml); 1+ tasks.")
    parser.add_argument(
        "cfg_name",
        nargs="?",
        type=str,
        help="Config name (file: _tr_cfg/<name>.yaml).",
    )
    parser.add_argument(
        "--config",
        dest="config",
        type=str,
        required=False,
        help="(legacy) Config name (file: _tr_cfg/<name>.yaml).",
    )
    parser.add_argument("--gpu-id", dest="gpu_id", type=str, required=False, help="Optional GPU id override.")
    parser.add_argument("--seed", dest="seed", type=int, required=False, help="Optional train seed override.")
    parser.add_argument(
        "--max-tr-steps",
        dest="max_tr_steps",
        type=int,
        required=False,
        help="Optional hard cap for optimizer steps (CLI highest priority).",
    )
    parser.add_argument(
        "--save-interval",
        dest="save_interval",
        type=int,
        required=False,
        help="Optional forced checkpoint interval in optimizer steps (CLI highest priority).",
    )
    args = parser.parse_args(argv[1:])

    act_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(act_dir)
    logger = _setup_logger(act_dir)

    cfg_name = args.config if args.config is not None else args.cfg_name
    if not cfg_name:
        parser.error("Missing cfg_name. Use: python3 _tr_wrapper.py <cfg_name> (or --config <cfg_name>)")

    try:
        cfg, cfg_src_abspath = _load_tr_cfg(act_dir, cfg_name)
        task_names, task_configs, expert_counts = _parse_train_tasks_rows(cfg)
        global_seed = int(cfg["TRAIN_SEED"])
        seed_source = "YAML:TRAIN_SEED"
        gpu_id = str(cfg["TRAIN_GPU_ID"]).strip()
        gpu_source = "YAML:TRAIN_GPU_ID"
        env_seed = os.environ.get("ACT_FLOW_SEED", "").strip()
        env_gpu = os.environ.get("ACT_FLOW_GPU", "").strip()
        if args.seed is not None:
            global_seed = int(args.seed)
            seed_source = "CLI:--seed"
        elif env_seed:
            global_seed = int(env_seed)
            seed_source = "FLOW_ENV:ACT_FLOW_SEED"
        if args.gpu_id is not None:
            gpu_id = str(args.gpu_id).strip()
            gpu_source = "CLI:--gpu-id"
        elif env_gpu:
            gpu_id = env_gpu
            gpu_source = "FLOW_ENV:ACT_FLOW_GPU"

        # Backward-compatible defaults (match historical hard-coded values in this wrapper).
        train_num_epochs = int(_get_cfg_opt(cfg, "TRAIN_NUM_EPOCHS", 6000))
        train_save_freq = int(_get_cfg_opt(cfg, "TRAIN_SAVE_FREQ", 2000))
        train_batch_size = int(_get_cfg_opt(cfg, "TRAIN_BATCH_SIZE", 8))
        train_lr = float(_get_cfg_opt(cfg, "TRAIN_LR", 1.0e-5))
        train_state_dim = int(_get_cfg_opt(cfg, "TRAIN_STATE_DIM", 14))

        act_kl_weight = int(_get_cfg_opt(cfg, "ACT_KL_WEIGHT", 10))
        act_chunk_size = int(_get_cfg_opt(cfg, "ACT_CHUNK_SIZE", 50))
        act_hidden_dim = int(_get_cfg_opt(cfg, "ACT_HIDDEN_DIM", 512))
        act_dim_feedforward = int(_get_cfg_opt(cfg, "ACT_DIM_FEEDFORWARD", 3200))

        # Early stopping: YAML baseline; optional FLOW env from __flow.py overrides YAML.
        early_stop_patience_evals = cfg.get("EARLY_STOP_PATIENCE_EVALS", None)
        early_stop_rel_tol = cfg.get("EARLY_STOP_REL_TOL", None)
        eval_steps_for_early_stop = cfg.get("EVAL_STEPS_FOR_EARLY_STOP", None)
        max_tr_steps = _get_cfg_opt(cfg, "MAX_TR_STEPS", None)
        save_interval = _get_cfg_opt(cfg, "SAVE_INTERVAL", None)
        if "ACT_FLOW_EARLY_STOP_PATIENCE_EVALS" in os.environ:
            early_stop_patience_evals = int(os.environ["ACT_FLOW_EARLY_STOP_PATIENCE_EVALS"].strip())
        if "ACT_FLOW_EARLY_STOP_REL_TOL" in os.environ:
            early_stop_rel_tol = float(os.environ["ACT_FLOW_EARLY_STOP_REL_TOL"].strip())
        if "ACT_FLOW_EVAL_STEPS_FOR_EARLY_STOP" in os.environ:
            eval_steps_for_early_stop = int(os.environ["ACT_FLOW_EVAL_STEPS_FOR_EARLY_STOP"].strip())
        if "ACT_FLOW_MAX_TR_STEPS" in os.environ:
            max_tr_steps = int(os.environ["ACT_FLOW_MAX_TR_STEPS"].strip())
        if "ACT_FLOW_SAVE_INTERVAL" in os.environ:
            save_interval = int(os.environ["ACT_FLOW_SAVE_INTERVAL"].strip())
        if args.max_tr_steps is not None:
            max_tr_steps = int(args.max_tr_steps)
        if args.save_interval is not None:
            save_interval = int(args.save_interval)
        logger.info(
            "Resolved runtime: seed=%s (%s), gpu_id=%s (%s)",
            global_seed,
            seed_source,
            gpu_id,
            gpu_source,
        )

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
        logger.info("Train num_epochs: %s", train_num_epochs)
        logger.info("Train save_freq: %s", train_save_freq)
        logger.info("Train batch_size: %s", train_batch_size)
        logger.info("Train lr: %s", train_lr)
        logger.info("Train state_dim: %s", train_state_dim)
        logger.info("ACT kl_weight: %s", act_kl_weight)
        logger.info("ACT chunk_size: %s", act_chunk_size)
        logger.info("ACT hidden_dim: %s", act_hidden_dim)
        logger.info("ACT dim_feedforward: %s", act_dim_feedforward)
        logger.info("Early-stop patience_evals: %s", early_stop_patience_evals)
        logger.info("Early-stop rel_tol: %s", early_stop_rel_tol)
        logger.info("Early-stop eval_steps_for_early_stop: %s", eval_steps_for_early_stop)
        logger.info("Max training steps: %s", max_tr_steps)
        logger.info("Forced checkpoint save_interval: %s", save_interval)

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

        # Single-task: combined_* paths match process_data layout (same dir as subtask).
        # Symlinking episode_i.hdf5 onto itself would hit FileExistsError; use existing SIM entry only.
        if len(task_names) == 1:
            existing_one = sim_task_configs.get(combined_key)
            if existing_one is None:
                raise KeyError(f"Missing SIM_TASK_CONFIGS entry for {combined_key}. Run process_data.sh first.")
            if (
                existing_one.get("episode_len") != episode_len
                or existing_one.get("camera_names") != camera_names
                or int(existing_one.get("num_episodes", -1)) != combined_total_episodes
            ):
                raise ValueError(
                    f"SIM_TASK_CONFIGS[{combined_key!r}] does not match expected "
                    f"episode_len/camera_names/num_episodes for this run."
                )
            logger.info("Single-task mode: using existing dataset_dir=%s (no symlink merge).", combined_dataset_dir)
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
            "--kl_weight", str(act_kl_weight),
            "--chunk_size", str(act_chunk_size),
            "--hidden_dim", str(act_hidden_dim),
            "--batch_size", str(train_batch_size),
            "--dim_feedforward", str(act_dim_feedforward),
            "--num_epochs", str(train_num_epochs),
            "--lr", str(train_lr),
            "--save_freq", str(train_save_freq),
            "--state_dim", str(train_state_dim),
            "--seed", str(global_seed),
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
            if os.path.isdir(sim_task_root_dir) and _has_only_symlink_files_recursive(sim_task_root_dir):
                shutil.rmtree(sim_task_root_dir)
                logger.info("Removed symlink-only dataset directory: %s", sim_task_root_dir)
            elif os.path.isdir(sim_task_root_dir):
                logger.warning("Skip removing directory with non-symlink files: %s", sim_task_root_dir)
        return 0
    except Exception as exc:
        logger.error("Wrapper failed: %s", str(exc))
        logger.error("Stack trace:\n%s", traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
