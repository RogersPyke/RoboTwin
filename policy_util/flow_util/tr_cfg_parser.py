#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training Configuration Parser.

@input: [str, model name (ACT/DP/TinyVLA)], [str, optional config path]
@output: [dict, merged and validated configuration]
@scenario: [Load shared config + model-specific config, merge with inheritance, validate structure]

Usage:
    from policy_util.flow_util.tr_cfg_parser import load_tr_config

    cfg = load_tr_config("ACT")
    cfg = load_tr_config("DP", config_path="custom/path/tr_tasks.yaml")
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


ROBOTWIN_ROOT = Path(__file__).resolve().parent.parent.parent
POLICY_ROOT = ROBOTWIN_ROOT / "policy"
SHARED_CFG_PATH = ROBOTWIN_ROOT / "policy_util" / "config" / "tr.yaml"

MODEL_CFG_PATHS = {
    "ACT": POLICY_ROOT / "ACT" / "_tr_cfg" / "tr_tasks.yaml",
    "DP": POLICY_ROOT / "DP" / "_tr_cfg" / "tr_tasks.yaml",
    "TinyVLA": POLICY_ROOT / "TinyVLA" / "_tr_cfg" / "tr_tasks.yaml",
}


def _load_yaml_file(path: Path) -> Dict[str, Any]:
    """
    @input: [Path, yaml file path]
    @output: [dict, parsed yaml content]
    @scenario: [Load and parse yaml file with error handling]
    """
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        content = yaml.safe_load(f)
    if not isinstance(content, dict):
        raise ValueError(f"Config file must be a dict: {path}")
    return content


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    @input: [dict, base config], [dict, override config]
    @output: [dict, merged config]
    @scenario: [Recursively merge override into base, override takes precedence]
    """
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _resolve_data_folder(data_folder: str) -> Path:
    """
    @input: [str, relative or absolute path to data folder]
    @output: [Path, resolved absolute path]
    @scenario: [Resolve data folder path relative to RoboTwin root]
    """
    path = Path(data_folder)
    if path.is_absolute():
        return path
    return ROBOTWIN_ROOT / path


def _validate_task_entry(task: Dict[str, Any], task_idx: int) -> None:
    """
    @input: [dict, task entry], [int, task index for error messages]
    @output: [None]
    @scenario: [Validate required fields in a tr_tasks entry]
    """
    if "task_id" not in task:
        raise ValueError(f"tr_tasks[{task_idx}] missing required field: task_id")

    has_single = "data_folder" in task
    has_joint = "data_sources" in task

    if has_single and has_joint:
        raise ValueError(
            f"tr_tasks[{task_idx}] cannot have both data_folder and data_sources"
        )

    if not has_single and not has_joint:
        raise ValueError(
            f"tr_tasks[{task_idx}] must have either data_folder or data_sources"
        )

    if has_single:
        required = ["task_name", "task_config", "expert_num", "data_folder"]
        for field in required:
            if field not in task:
                raise ValueError(
                    f"tr_tasks[{task_idx}] missing required field: {field}"
                )

    if has_joint:
        sources = task["data_sources"]
        if not isinstance(sources, list) or len(sources) == 0:
            raise ValueError(
                f"tr_tasks[{task_idx}].data_sources must be a non-empty list"
            )
        for src_idx, src in enumerate(sources):
            required = ["task_name", "task_config", "expert_num", "data_folder"]
            for field in required:
                if field not in src:
                    raise ValueError(
                        f"tr_tasks[{task_idx}].data_sources[{src_idx}] missing: {field}"
                    )


def _expand_task_config_alias(
    task_config: str,
    aliases: Dict[str, List[str]],
) -> List[str]:
    """
    @input: [str, task_config name], [dict, alias mapping]
    @output: [list[str], list of acceptable task_config names]
    @scenario: [Expand task_config to include aliases if defined]
    """
    if task_config in aliases:
        return aliases[task_config]
    return [task_config]


def _resolve_task_data_folders(
    task: Dict[str, Any],
    aliases: Dict[str, List[str]],
) -> List[Tuple[str, str, int, Path]]:
    """
    @input: [dict, task entry], [dict, task_config aliases]
    @output: [list[tuple], list of (task_name, task_config, expert_num, data_folder_path)]
    @scenario: [Resolve all data folders for a task (single or joint)]
    """
    results: List[Tuple[str, str, int, Path]] = []

    if "data_folder" in task:
        task_name = task["task_name"]
        task_config = task["task_config"]
        expert_num = task["expert_num"]
        data_folder = _resolve_data_folder(task["data_folder"])
        results.append((task_name, task_config, expert_num, data_folder))

    elif "data_sources" in task:
        for src in task["data_sources"]:
            task_name = src["task_name"]
            task_config = src["task_config"]
            expert_num = src["expert_num"]
            data_folder = _resolve_data_folder(src["data_folder"])
            results.append((task_name, task_config, expert_num, data_folder))

    return results


def load_tr_config(
    model: str,
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    @input: [str, model name (ACT/DP/TinyVLA)], [str, optional custom config path]
    @output: [dict, merged and validated configuration]
    @scenario: [Load shared config + model config, merge, validate, resolve paths]
    """
    model_lower = model.lower()
    if model_lower == "tinyvla":
        model_normalized = "TinyVLA"
    else:
        model_normalized = model.upper()
    if model_normalized not in MODEL_CFG_PATHS:
        raise ValueError(
            f"Unknown model: {model}. Must be one of: {list(MODEL_CFG_PATHS.keys())}"
        )
    model = model_normalized

    shared_cfg = _load_yaml_file(SHARED_CFG_PATH)
    using_unified_shared = "global" in shared_cfg and model in shared_cfg

    if config_path:
        model_cfg_path = Path(config_path)
        model_cfg = _load_yaml_file(model_cfg_path)
        if "global" in model_cfg and model in model_cfg:
            merged = _deep_merge(model_cfg.get("global", {}), model_cfg.get(model, {}))
        else:
            if using_unified_shared:
                merged = _deep_merge(shared_cfg.get("global", {}), shared_cfg.get(model, {}))
                merged = _deep_merge(merged, model_cfg)
            else:
                merged = _deep_merge(shared_cfg, model_cfg)
    else:
        if using_unified_shared:
            model_cfg_path = SHARED_CFG_PATH
            merged = _deep_merge(shared_cfg.get("global", {}), shared_cfg.get(model, {}))
        else:
            model_cfg_path = MODEL_CFG_PATHS[model]
            model_cfg = _load_yaml_file(model_cfg_path)
            merged = _deep_merge(shared_cfg, model_cfg)

    if "gpu_parallel" not in merged:
        raise ValueError(f"Model config must define gpu_parallel: {model}")
    if "model_defaults" not in merged:
        raise ValueError(f"Model config must define model_defaults: {model}")
    if "tr_tasks" not in merged:
        raise ValueError(f"Model config must define tr_tasks: {model}")

    tr_tasks = merged["tr_tasks"]
    if not isinstance(tr_tasks, list):
        raise ValueError("tr_tasks must be a list")

    aliases = merged.get("task_config_aliases", {})

    for idx, task in enumerate(tr_tasks):
        _validate_task_entry(task, idx)
        resolved = _resolve_task_data_folders(task, aliases)
        task["_resolved_data"] = resolved

    merged["_model"] = model
    merged["_config_path"] = str(model_cfg_path)
    merged["_robotwin_root"] = str(ROBOTWIN_ROOT)

    return merged


def get_task_override_params(
    task: Dict[str, Any],
    model_defaults: Dict[str, Any],
) -> Dict[str, Any]:
    """
    @input: [dict, task entry], [dict, model defaults]
    @output: [dict, merged params with task overrides applied]
    @scenario: [Get effective params for a task: defaults + task-specific overrides]
    """
    result = dict(model_defaults)

    override_keys = set(task.keys()) - {
        "task_id",
        "task_name",
        "task_config",
        "expert_num",
        "data_folder",
        "data_sources",
        "_resolved_data",
    }

    for key in override_keys:
        result[key] = task[key]

    return result


if __name__ == "__main__":
    import json

    for model in ["ACT", "DP", "TinyVLA"]:
        print(f"\n{'=' * 60}")
        print(f"Model: {model}")
        print(f"{'=' * 60}")
        cfg = load_tr_config(model)
        print(f"GPU parallel: {cfg['gpu_parallel']}")
        print(f"Seed: {cfg.get('seed')}")
        print(f"Tasks: {len(cfg['tr_tasks'])}")
        for task in cfg["tr_tasks"]:
            print(
                f"  - {task['task_id']}: {len(task['_resolved_data'])} data source(s)"
            )
