#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TinyVLA training wrapper.

Primary mode:
- ACT-aligned YAML with TRAIN_TASKS for joint training
- wrapper-generated output_dir under tinyvla_ckpt/
- top-level early-stop fields injected into train_vla.py

Compatibility mode:
- legacy single-task YAML that still provides VLA_TRAIN_ARGS.task_name and output_dir
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import yaml

from aloha_scripts.constants import TASK_CONFIGS


LOGGER: Optional[logging.Logger] = None
EARLY_STOP_KEY_MAP = {
    "EARLY_STOP_PATIENCE_EVALS": "early_stop_patience_evals",
    "EARLY_STOP_REL_TOL": "early_stop_rel_tol",
    "EVAL_STEPS_FOR_EARLY_STOP": "eval_steps_for_early_stop",
}


def log_exceptions(func):
    """
    @input: [callable, any signature]
    @output: [Any, returns func output]
    @scenario: [Log exceptions with stack trace before re-raising]
    """

    def _wrapped(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            if LOGGER is not None:
                LOGGER.error("Wrapper failed: %s", str(exc))
                LOGGER.error("Stack trace:\n%s", traceback.format_exc())
            raise

    return _wrapped


def _setup_logger(tinyvla_dir: str) -> logging.Logger:
    """
    @input: [str, tinyvla_dir, any non-empty path]
    @output: [logging.Logger, configured logger]
    @scenario: [Consistent file+console logging for wrapper runs]
    """
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
    """
    @input: [str, path, existing yaml file]
    @output: [dict, parsed YAML]
    @scenario: [Load config yaml with basic validation checks]
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict) or not cfg:
        raise ValueError(f"Empty or invalid YAML config: {path}")
    return cfg


def _ensure_required_keys(cfg: Dict[str, Any], keys: List[str]) -> None:
    """
    @input: [dict, cfg], [list[str], keys]
    @output: [None]
    @scenario: [Fail fast when required config keys are missing]
    """
    missing = [k for k in keys if k not in cfg]
    if missing:
        raise KeyError(f"Missing required keys in config: {missing}")


def _normalize_cfg_name(cfg_name: str) -> str:
    """
    @input: [str, cfg_name]
    @output: [str, normalized file basename]
    @scenario: [Support cfg_name with or without .yaml extension]
    """
    return os.path.splitext(os.path.basename(cfg_name))[0]


def _parse_task_rows(cfg: Dict[str, Any], key: str) -> List[Tuple[str, str, int]]:
    """
    @input: [dict, cfg], [str, key]
    @output: [list[tuple[str, str, int]], parsed rows]
    @scenario: [Parse config rows like [task_name, task_config, expert_num]]
    """
    if key not in cfg:
        raise ValueError(f"Config must define {key} as a non-empty list of [task_name, task_config, expert_num].")
    rows = cfg[key]
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{key} must be a non-empty list.")

    parsed_rows: List[Tuple[str, str, int]] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            raise ValueError(f"{key}[{idx}] must be [task_name, task_config, expert_num], got {row!r}")
        task_name = str(row[0]).strip()
        task_config = str(row[1]).strip()
        expert_num = int(row[2])
        if not task_name or not task_config:
            raise ValueError(f"{key}[{idx}] contains an empty task_name/task_config.")
        if expert_num <= 0:
            raise ValueError(f"{key}[{idx}] expert_num must be > 0, got {expert_num}.")
        parsed_rows.append((task_name, task_config, expert_num))
    return parsed_rows


def _build_argv_from_args_dict(args_dict: Dict[str, Any], override_seed: Optional[int]) -> List[str]:
    """
    @input: [dict, args_dict], [Optional[int], override_seed]
    @output: [list[str], argv tokens for train_vla.py]
    @scenario: [Convert structured args dict into CLI tokens]
    """
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


def _extract_output_dir_from_args_dict(args_dict: Dict[str, Any]) -> Optional[str]:
    """
    @input: [dict, args_dict]
    @output: [Optional[str], output_dir if found]
    @scenario: [Enable saving a config snapshot into output_dir]
    """
    for key in ("output_dir", "outputDir", "OUTPUT_DIR"):
        if key in args_dict and args_dict[key] is not None:
            return str(args_dict[key])
    return None


def _build_dataset_dir(tinyvla_dir: str, task_name: str, task_config: str, expert_num: int) -> str:
    """
    @input: [str, tinyvla_dir], [str, task_name], [str, task_config], [int, expert_num]
    @output: [str, absolute dataset directory]
    @scenario: [Map TRAIN_TASKS row to TinyVLA processed-data directory]
    """
    return os.path.join(tinyvla_dir, "data", f"sim-{task_name}", f"{task_config}-{expert_num}")


def _infer_task_weight(task_config: Dict[str, Any]) -> float:
    """
    @input: [dict, task_config from TASK_CONFIGS]
    @output: [float, weight value]
    @scenario: [Keep a stable placeholder weight for wrapper-generated joint specs]
    """
    sample_weights = task_config.get("sample_weights")
    if isinstance(sample_weights, list) and sample_weights:
        try:
            return float(sample_weights[0])
        except (TypeError, ValueError):
            return 1.0
    return 1.0


def _build_joint_train_contract(tinyvla_dir: str, train_rows: List[Tuple[str, str, int]]) -> Dict[str, Any]:
    """
    @input: [str, tinyvla_dir], [list[tuple[str, str, int]], TRAIN_TASKS rows]
    @output: [dict, derived joint-training contract]
    @scenario: [Generate ACT-style joint task slug, output_dir, and task spec]
    """
    task_names = [row[0] for row in train_rows]
    task_configs = [row[1] for row in train_rows]
    expert_counts = [row[2] for row in train_rows]

    combined_task_slug = "__".join(task_names)
    combined_config_slug = "__".join(task_configs)
    combined_total_episodes = int(sum(expert_counts))

    camera_names: Optional[List[str]] = None
    episode_len: Optional[int] = None
    dataset_dirs: List[str] = []
    sample_weights: List[float] = []

    for task_name, task_config_name, expert_num in train_rows:
        if task_name not in TASK_CONFIGS:
            raise KeyError(
                f"Subtask '{task_name}' is missing from aloha_scripts.constants.TASK_CONFIGS. "
                "Register each base task first; the wrapper only synthesizes the joint task."
            )
        base_task_cfg = TASK_CONFIGS[task_name]
        current_camera_names = list(base_task_cfg["camera_names"])
        current_episode_len = int(base_task_cfg["episode_len"])
        if camera_names is None:
            camera_names = current_camera_names
            episode_len = current_episode_len
        else:
            if current_camera_names != camera_names:
                raise ValueError("All TRAIN_TASKS entries must share the same camera_names for joint training.")
            if current_episode_len != episode_len:
                raise ValueError("All TRAIN_TASKS entries must share the same episode_len for joint training.")

        dataset_dir = _build_dataset_dir(tinyvla_dir, task_name, task_config_name, expert_num)
        if not os.path.isdir(dataset_dir):
            raise FileNotFoundError(
                f"Dataset directory does not exist: {dataset_dir}. "
                "Run process_data.py for this task/config/count first."
            )
        for episode_idx in range(expert_num):
            episode_path = os.path.join(dataset_dir, f"episode_{episode_idx}.hdf5")
            if not os.path.isfile(episode_path):
                raise FileNotFoundError(f"Missing dataset episode file: {episode_path}")

        dataset_dirs.append(dataset_dir)
        sample_weights.append(_infer_task_weight(base_task_cfg))

    output_dir = os.path.join(
        tinyvla_dir,
        "tinyvla_ckpt",
        f"tinyvla-{combined_task_slug}",
        f"{combined_config_slug}-{combined_total_episodes}",
    )

    return {
        "combined_task_slug": combined_task_slug,
        "combined_config_slug": combined_config_slug,
        "combined_total_episodes": combined_total_episodes,
        "output_dir": output_dir,
        "joint_task_spec": {
            "task_name": combined_task_slug,
            "dataset_dir": dataset_dirs,
            "camera_names": camera_names,
            "episode_len": episode_len,
            "sample_weights": sample_weights,
            "train_tasks": [
                {
                    "task_name": task_name,
                    "task_config": task_config_name,
                    "expert_num": expert_num,
                }
                for task_name, task_config_name, expert_num in train_rows
            ],
        },
    }


def _inject_top_level_early_stop(cfg: Dict[str, Any], args_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    @input: [dict, cfg], [dict, args_dict]
    @output: [dict, updated args_dict]
    @scenario: [Promote ACT-style top-level early-stop fields into CLI args]
    """
    updated_args = dict(args_dict)
    has_top_level_early_stop = any(key in cfg for key in EARLY_STOP_KEY_MAP)
    if has_top_level_early_stop:
        for arg_key in EARLY_STOP_KEY_MAP.values():
            updated_args.pop(arg_key, None)
        for cfg_key, arg_key in EARLY_STOP_KEY_MAP.items():
            if cfg.get(cfg_key, None) is not None:
                updated_args[arg_key] = cfg[cfg_key]
    return updated_args


def _prepare_train_args(
    tinyvla_dir: str,
    cfg: Dict[str, Any],
) -> Tuple[Dict[str, Any], str, Optional[Dict[str, Any]]]:
    """
    @input: [str, tinyvla_dir], [dict, cfg]
    @output: [tuple, (train_args, output_dir, joint_contract_or_none)]
    @scenario: [Normalize wrapper config into train_vla.py arguments]
    """
    _ensure_required_keys(cfg, ["TRAIN_SEED", "TRAIN_GPU_ID", "VLA_TRAIN_ARGS"])
    vla_train_args = cfg["VLA_TRAIN_ARGS"]
    if not isinstance(vla_train_args, dict) or not vla_train_args:
        raise ValueError("VLA_TRAIN_ARGS must be a non-empty mapping of args_name -> value.")

    train_args = _inject_top_level_early_stop(cfg, vla_train_args)

    if "TRAIN_TASKS" not in cfg:
        output_dir = _extract_output_dir_from_args_dict(train_args)
        if not output_dir:
            raise KeyError("Legacy configs must include VLA_TRAIN_ARGS.output_dir.")
        return dict(train_args), str(output_dir), None

    joint_contract = _build_joint_train_contract(tinyvla_dir, _parse_task_rows(cfg, "TRAIN_TASKS"))
    train_args = dict(train_args)
    train_args["task_name"] = joint_contract["combined_task_slug"]
    train_args["output_dir"] = joint_contract["output_dir"]
    train_args.setdefault("logging_dir", os.path.join(joint_contract["output_dir"], "log"))
    train_args["joint_task_spec"] = json.dumps(
        joint_contract["joint_task_spec"],
        sort_keys=True,
        separators=(",", ":"),
    )

    if LOGGER is not None:
        LOGGER.info("Resolved joint task slug: %s", joint_contract["combined_task_slug"])
        LOGGER.info("Resolved joint config slug: %s", joint_contract["combined_config_slug"])
        LOGGER.info("Resolved joint output_dir: %s", joint_contract["output_dir"])

    return train_args, str(joint_contract["output_dir"]), joint_contract


def _snapshot_training_metadata(
    cfg_path: str,
    output_dir: str,
    tinyvla_dir: str,
    joint_contract: Optional[Dict[str, Any]],
) -> None:
    """
    @input: [str, cfg_path], [str, output_dir], [str, tinyvla_dir], [Optional[dict], joint_contract]
    @output: [None]
    @scenario: [Persist config and wrapper-derived metadata for traceability]
    """
    os.makedirs(output_dir, exist_ok=True)
    dst_cfg = os.path.join(output_dir, os.path.basename(cfg_path))
    shutil.copy2(cfg_path, dst_cfg)

    manifest_path = os.path.join(output_dir, "training_run_manifest.txt")
    with open(manifest_path, "w", encoding="ascii") as manifest:
        manifest.write(f"training_config_source={os.path.abspath(cfg_path)}\n")
        manifest.write(f"tinyvla_policy_dir={tinyvla_dir}\n")
        manifest.write(f"output_dir={output_dir}\n")
        manifest.write(f"copied_yaml={os.path.basename(dst_cfg)}\n")
        if joint_contract is not None:
            manifest.write(f"combined_task_slug={joint_contract['combined_task_slug']}\n")
            manifest.write(f"combined_config_slug={joint_contract['combined_config_slug']}\n")
            manifest.write(f"combined_total_episodes={joint_contract['combined_total_episodes']}\n")

    if joint_contract is not None:
        spec_path = os.path.join(output_dir, "joint_task_spec.json")
        with open(spec_path, "w", encoding="ascii") as f:
            json.dump(joint_contract["joint_task_spec"], f, indent=2)


@log_exceptions
def _run_training(tinyvla_dir: str, cfg_name: str) -> int:
    """
    @input: [str, tinyvla_dir], [str, cfg_name]
    @output: [int, 0 on success else non-zero]
    @scenario: [Load _tr_cfg yaml, build command, launch TinyVLA training]
    """
    cfg_base = _normalize_cfg_name(cfg_name)
    cfg_path = os.path.join(tinyvla_dir, "_tr_cfg", f"{cfg_base}.yaml")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"No config found at: _tr_cfg/{cfg_base}.yaml")

    cfg = _load_yaml(cfg_path)
    train_seed = int(cfg["TRAIN_SEED"])
    train_gpu_id = str(cfg["TRAIN_GPU_ID"])

    train_args, output_dir, joint_contract = _prepare_train_args(tinyvla_dir, cfg)

    deepspeed_cfg = cfg.get("DEEPSPEED", None)
    use_deepspeed = bool(
        deepspeed_cfg and isinstance(deepspeed_cfg, dict) and deepspeed_cfg.get("enabled", False)
    )

    zero2_json = None
    num_gpus = None
    master_port = None
    if use_deepspeed:
        _ensure_required_keys(deepspeed_cfg, ["num_gpus", "master_port", "zero2_json"])
        num_gpus = int(deepspeed_cfg["num_gpus"])
        master_port = int(deepspeed_cfg["master_port"])
        zero2_json = str(deepspeed_cfg["zero2_json"])
        if os.path.isabs(zero2_json):
            zero2_json = os.path.relpath(zero2_json, start=tinyvla_dir)
        train_args = dict(train_args)
        train_args["deepspeed"] = os.path.normpath(zero2_json)

    argv_tokens = _build_argv_from_args_dict(train_args, override_seed=train_seed)
    _snapshot_training_metadata(cfg_path, output_dir, tinyvla_dir, joint_contract)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = train_gpu_id
    env["PYTHONNOUSERSITE"] = "1"

    if use_deepspeed:
        cmd: List[str] = [
            "deepspeed",
            "--master_port",
            str(master_port),
            f"--num_gpus={num_gpus}",
            "--num_nodes=1",
            "./train_vla.py",
        ] + argv_tokens
    else:
        # Use the current interpreter to ensure the subprocess inherits all deps
        # from the active conda environment (e.g., typing_extensions, yaml).
        cmd = [sys.executable, "./train_vla.py"] + argv_tokens

    if LOGGER is not None:
        LOGGER.info("Launching training with config=%s", cfg_base)
        LOGGER.info("CUDA_VISIBLE_DEVICES=%s", train_gpu_id)
        LOGGER.info("Output dir=%s", output_dir)
        LOGGER.info("Command: %s", " ".join(cmd))

    subprocess.run(cmd, check=True, env=env, cwd=tinyvla_dir)
    return 0


def main(argv: List[str]) -> int:
    """
    @input: [List[str], argv]
    @output: [int, 0 on success else non-zero]
    @scenario: [CLI entry for training wrapper]
    """
    parser = argparse.ArgumentParser(description="TinyVLA training wrapper (config under _tr_cfg/*.yaml)")
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
    args = parser.parse_args(argv[1:])

    tinyvla_dir = os.path.dirname(os.path.abspath(__file__))
    _setup_logger(tinyvla_dir)

    cfg = args.config if args.config is not None else args.cfg_name
    if not cfg:
        parser.error("Missing cfg_name. Use: python3 _tr_wrapper.py <cfg_name> (or --config <cfg_name>)")

    try:
        return _run_training(tinyvla_dir, cfg)
    except Exception:
        if LOGGER is not None:
            LOGGER.error("Top-level wrapper exit due to failure.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

