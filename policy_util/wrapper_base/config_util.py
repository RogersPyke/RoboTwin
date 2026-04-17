#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Common configuration utilities for ACT/DP/TinyVLA wrappers.

@input: [str, YAML path], [dict, config dict], [str, task_id]
@output: [dict, merged config], [Optional[dict], task config]
@scenario: [Load, merge, and resolve training configurations]

Dependencies:
- PyYAML

Usage:
    from wrapper_base.config_util import (
        load_unified_config,
        merge_global_and_model_config,
        resolve_task_config,
        resolve_data_folder,
        maybe_add_arg,
    )
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def load_unified_config(yaml_path: str) -> Dict[str, Any]:
    """
    @input: [str, path to YAML file]
    @output: [dict, parsed YAML content]
    @scenario: [Load and validate unified training config from YAML file]

    Raises ValueError if the file does not contain a valid dict.
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML config: {yaml_path}")
    return cfg


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    @input: [dict, base config], [dict, override config]
    @output: [dict, merged config with override taking precedence]
    @scenario: [Recursively merge two dicts, override values take precedence]

    Nested dicts are merged recursively; other values are replaced.
    """
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def merge_global_and_model_config(
    unified_cfg: Dict[str, Any], model_name: str
) -> Dict[str, Any]:
    """
    @input: [dict, unified config with 'global' and model sections], [str, model name like 'ACT', 'DP', 'TinyVLA']
    @output: [dict, merged config with global defaults overridden by model-specific settings]
    @scenario: [Extract model-specific config by merging global defaults with model overrides]

    The unified config should have a 'global' section and optional model-specific sections.
    """
    global_cfg = unified_cfg.get("global", {})
    model_cfg = unified_cfg.get(model_name, {})
    result = dict(global_cfg)
    for key, value in model_cfg.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def resolve_task_config(
    cfg: Dict[str, Any], task_id: str, task_list_key: str = "tr_tasks"
) -> Optional[Dict[str, Any]]:
    """
    @input: [dict, config with task list], [str, task_id to find], [str, key for task list]
    @output: [Optional[dict], task config if found, else None]
    @scenario: [Find a specific task config by task_id in the task list]
    """
    for task in cfg.get(task_list_key, []):
        if task.get("task_id") == task_id:
            return task
    return None


def resolve_data_folder(data_folder: str, root: Path) -> Path:
    """
    @input: [str, data folder path (relative or absolute)], [Path, root for relative paths]
    @output: [Path, resolved absolute path]
    @scenario: [Resolve data folder path, handling both relative and absolute paths]
    """
    path = Path(data_folder)
    if path.is_absolute():
        return path
    return root / path


def maybe_add_arg(cmd: List[str], flag: str, value: Any) -> None:
    """
    @input: [list, command list to modify], [str, flag like '--gpu-id'], [Any, value to add]
    @output: [None, modifies cmd in place]
    @scenario: [Append flag and value to command list if value is not None]
    """
    if value is not None:
        cmd.extend([flag, str(value)])


def build_argv_from_args_dict(
    args_dict: Dict[str, Any], override_seed: Optional[int] = None
) -> List[str]:
    """
    @input: [dict, argument name -> value mapping], [Optional[int], seed override]
    @output: [list, CLI argument list like ['--arg1', 'val1', '--arg2', 'val2']]
    @scenario: [Convert args dict to CLI argv format, optionally overriding seed]

    Used primarily by TinyVLA wrapper for HuggingFace-style argument passing.
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
