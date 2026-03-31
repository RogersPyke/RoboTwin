#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TinyVLA evaluation wrapper.

Primary mode:
- ACT-aligned YAML with TRAIN_TASKS + EVAL_TASKS
- wrapper-derived joint checkpoint directory under tinyvla_ckpt/
- one config can evaluate the same joint checkpoint on multiple tasks

Compatibility mode:
- legacy single-task YAML with TASK_NAME/TASK_CONFIG/OUTPUT_DIR
"""

import argparse
import logging
import os
import subprocess
import sys
import traceback
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import yaml


LOGGER: Optional[logging.Logger] = None


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
    @input: [str, tinyvla_dir]
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
    @input: [str, path]
    @output: [dict, parsed YAML]
    @scenario: [Load config yaml]
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
    @output: [str, normalized base name]
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


def _build_joint_eval_contract(tinyvla_dir: str, ev_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    @input: [str, tinyvla_dir], [dict, ev_cfg]
    @output: [dict, derived joint checkpoint contract]
    @scenario: [Resolve output_dir/model_path/state_path from TRAIN_TASKS]
    """
    train_rows = _parse_task_rows(ev_cfg, "TRAIN_TASKS")
    task_slug = "__".join([row[0] for row in train_rows])
    config_slug = "__".join([row[1] for row in train_rows])
    total_episodes = int(sum([row[2] for row in train_rows]))
    output_dir = os.path.join(
        tinyvla_dir,
        "tinyvla_ckpt",
        f"tinyvla-{task_slug}",
        f"{config_slug}-{total_episodes}",
    )
    use_policy_best = bool(ev_cfg.get("USE_POLICY_BEST", False))
    model_path = os.path.join(output_dir, "policy_best") if use_policy_best else output_dir
    state_path = os.path.join(output_dir, "dataset_stats.pkl")
    return {
        "task_slug": task_slug,
        "config_slug": config_slug,
        "total_episodes": total_episodes,
        "output_dir": output_dir,
        "model_path": model_path,
        "state_path": state_path,
        "ckpt_setting": config_slug,
        "eval_runs": _parse_task_rows(ev_cfg, "EVAL_TASKS"),
    }


def _build_eval_overrides(
    ev_cfg: Dict[str, Any],
    task_name: str,
    task_config: str,
    expert_num: int,
    ckpt_setting: str,
    model_path: str,
    state_path: str,
) -> List[str]:
    """
    @input: [dict, ev_cfg], [str, task_name], [str, task_config], [int, expert_num]
    @output: [list[str], overrides tokens after --overrides]
    @scenario: [Translate wrapper config into eval_policy CLI overrides]
    """
    overrides: List[str] = []

    def add_pair(key: str, value: Any) -> None:
        if value is None:
            return
        overrides.extend([f"--{key}", str(value)])

    add_pair("task_name", task_name)
    add_pair("task_config", task_config)
    add_pair("ckpt_setting", ckpt_setting)
    add_pair("expert_data_num", expert_num)
    add_pair("seed", ev_cfg.get("EVAL_SEED"))
    add_pair("model_base", ev_cfg.get("MODEL_BASE"))
    add_pair("model_path", model_path)
    add_pair("state_path", state_path)
    add_pair("enable_lore", ev_cfg.get("ENABLE_LORE", False))
    add_pair("instruction_type", ev_cfg.get("INSTRUCTION_TYPE", None))
    # Allow dry-run evaluation to limit rollout count (used by script/eval_policy.py).
    add_pair("test_num", ev_cfg.get("EVAL_TEST_NUM", None))
    add_pair("force_end_reset_to_init", _resolve_eval_force_end_reset_to_init(ev_cfg))
    return overrides


def _to_cli_bool(v: Any) -> str:
    """
    @input: [Any, bool-like value]
    @output: [str, "true" or "false"]
    @scenario: [Normalize bool-like config/env values for CLI overrides]
    """
    if isinstance(v, str):
        return "true" if v.strip().lower() in ("1", "true", "yes", "y", "on") else "false"
    return "true" if bool(v) else "false"


def _resolve_eval_force_end_reset_to_init(ev_cfg: Dict[str, Any]) -> str:
    """
    @input: [dict, eval config]
    @output: [str, "true" or "false"]
    @scenario: [Resolve eval-only reset requirement from eval config]
    """
    return _to_cli_bool(ev_cfg["END_RESET_TO_INIT"])


def _resolve_eval_contract(tinyvla_dir: str, ev_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    @input: [str, tinyvla_dir], [dict, ev_cfg]
    @output: [dict, normalized evaluation contract]
    @scenario: [Support both ACT-style joint-eval config and legacy single-task config]
    """
    if "TRAIN_TASKS" in ev_cfg or "EVAL_TASKS" in ev_cfg:
        _ensure_required_keys(
            ev_cfg,
            ["EVAL_SEED", "EVAL_GPU_ID", "MODEL_BASE", "TRAIN_TASKS", "EVAL_TASKS", "END_RESET_TO_INIT"],
        )
        return _build_joint_eval_contract(tinyvla_dir, ev_cfg)

    _ensure_required_keys(
        ev_cfg,
        ["EVAL_SEED", "EVAL_GPU_ID", "TASK_NAME", "TASK_CONFIG", "MODEL_BASE", "OUTPUT_DIR", "END_RESET_TO_INIT"],
    )
    output_dir = str(ev_cfg["OUTPUT_DIR"])
    use_policy_best = bool(ev_cfg.get("USE_POLICY_BEST", False))
    model_path = os.path.join(output_dir, "policy_best") if use_policy_best else output_dir
    state_path = os.path.join(output_dir, "dataset_stats.pkl")
    return {
        "task_slug": str(ev_cfg["TASK_NAME"]),
        "config_slug": str(ev_cfg.get("CKPT_SETTING", "legacy")),
        "total_episodes": int(ev_cfg.get("EXPERT_DATA_NUM", 0)),
        "output_dir": output_dir,
        "model_path": model_path,
        "state_path": state_path,
        "ckpt_setting": str(ev_cfg.get("CKPT_SETTING", "legacy")),
        "eval_runs": [
            (
                str(ev_cfg["TASK_NAME"]).strip(),
                str(ev_cfg["TASK_CONFIG"]).strip(),
                int(ev_cfg.get("EXPERT_DATA_NUM", 0)),
            )
        ],
    }


@log_exceptions
def _run_eval(tinyvla_dir: str, cfg_name: str) -> int:
    """
    @input: [str, tinyvla_dir], [str, cfg_name]
    @output: [int, 0 on success else non-zero]
    @scenario: [Load _ev_cfg yaml, build overrides, run eval_policy.py]
    """
    cfg_base = _normalize_cfg_name(cfg_name)
    cfg_path = os.path.join(tinyvla_dir, "_ev_cfg", f"{cfg_base}.yaml")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"No config found at: _ev_cfg/{cfg_base}.yaml")

    ev_cfg = _load_yaml(cfg_path)
    contract = _resolve_eval_contract(tinyvla_dir, ev_cfg)

    model_path = contract["model_path"]
    state_path = contract["state_path"]
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    if not os.path.isfile(state_path):
        raise FileNotFoundError(f"State path does not exist: {state_path}")

    repo_root = os.path.abspath(os.path.join(tinyvla_dir, "..", ".."))
    policy_deploy_yml = "policy/TinyVLA/deploy_policy.yml"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(ev_cfg["EVAL_GPU_ID"])
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONWARNINGS"] = "ignore::UserWarning"
    env["ACT_EV_CFG_SNAPSHOT_SRC"] = os.path.abspath(cfg_path)
    # Ensure `from vla import ...` works inside deploy_policy.py.
    # When we run from `repo_root`, Python won't automatically include `policy/TinyVLA/` in sys.path.
    existing_py_path = env.get("PYTHONPATH", "")
    added_py_path = os.path.abspath(tinyvla_dir)
    env["PYTHONPATH"] = added_py_path + (os.pathsep + existing_py_path if existing_py_path else "")

    if LOGGER is not None:
        LOGGER.info("Launching eval with config=%s", cfg_base)
        LOGGER.info("Resolved output_dir=%s", contract["output_dir"])
        LOGGER.info("Resolved model_path=%s", model_path)

    for task_name, task_config, expert_num in contract["eval_runs"]:
        overrides_tokens = _build_eval_overrides(
            ev_cfg=ev_cfg,
            task_name=task_name,
            task_config=task_config,
            expert_num=expert_num,
            ckpt_setting=contract["ckpt_setting"],
            model_path=model_path,
            state_path=state_path,
        )
        cmd: List[str] = [
            sys.executable,
            "script/eval_policy.py",
            "--config",
            policy_deploy_yml,
            "--overrides",
        ] + overrides_tokens
        if LOGGER is not None:
            LOGGER.info(
                "Eval run: task_name=%s task_config=%s expert_num=%s ckpt_setting=%s",
                task_name,
                task_config,
                expert_num,
                contract["ckpt_setting"],
            )
            LOGGER.info("Command: %s", " ".join(cmd))
        subprocess.run(cmd, check=True, env=env, cwd=repo_root)

    return 0


def main(argv: List[str]) -> int:
    """
    @input: [List[str], argv]
    @output: [int, 0 on success else non-zero]
    @scenario: [CLI entry for TinyVLA eval wrapper]
    """
    parser = argparse.ArgumentParser(description="TinyVLA evaluation wrapper (config under _ev_cfg/*.yaml)")
    parser.add_argument(
        "cfg_name",
        nargs="?",
        type=str,
        help="Config name (file: _ev_cfg/<name>.yaml).",
    )
    parser.add_argument(
        "--config",
        dest="config",
        type=str,
        required=False,
        help="(legacy) Config name (file: _ev_cfg/<name>.yaml).",
    )
    args = parser.parse_args(argv[1:])

    tinyvla_dir = os.path.dirname(os.path.abspath(__file__))
    _setup_logger(tinyvla_dir)

    cfg = args.config if args.config is not None else args.cfg_name
    if not cfg:
        parser.error("Missing cfg_name. Use: python3 _ev_wrapper.py <cfg_name> (or --config <cfg_name>)")

    try:
        return _run_eval(tinyvla_dir, cfg)
    except Exception:
        if LOGGER is not None:
            LOGGER.error("Top-level wrapper exit due to failure.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

