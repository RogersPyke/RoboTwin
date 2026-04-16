#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified flow scheduler utility for ACT/DP/TinyVLA training pipelines.

This module provides common functionality for:
1. Loading unified YAML configuration
2. Extracting model-specific settings (GPU parallel, tasks, etc.)
3. Building training commands for _tr_wrapper.py
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


def get_unified_config_path() -> Path:
    """Get the default unified config path."""
    policy_util_root = Path(__file__).resolve().parent.parent
    return policy_util_root / "config" / "tr.yaml"


def load_unified_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load the unified training configuration."""
    if config_path is None:
        config_path = get_unified_config_path()
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML config: {config_path}")
    return cfg


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def get_model_config(unified_cfg: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """
    Get model-specific configuration by merging global and model sections.

    @input: unified_cfg - Full unified config dict
    @input: model_name - Model name (ACT, DP, TinyVLA)
    @output: Merged config for the model
    """
    global_cfg = unified_cfg.get("global", {})
    model_cfg = unified_cfg.get(model_name, {})
    return deep_merge(global_cfg, model_cfg)


def get_gpu_parallel(model_cfg: Dict[str, Any]) -> List[int]:
    """
    Get GPU parallel configuration.

    @input: model_cfg - Model-specific config
    @output: List of GPU IDs for parallel slots
    """
    return model_cfg.get("gpu_parallel", [0])


def get_tr_tasks(model_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Get training tasks list.

    @input: model_cfg - Model-specific config
    @output: List of task configurations
    """
    return model_cfg.get("tr_tasks", [])


def get_early_stop_config(model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Get early stop configuration."""
    return model_cfg.get("early_stop", {})


def get_limits_config(model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Get limits configuration."""
    return model_cfg.get("limits", {})


def get_seed(model_cfg: Dict[str, Any]) -> int:
    """Get seed value."""
    return model_cfg.get("seed", 0)


def build_tr_wrapper_cmd(
    task_id: str,
    yaml_path: Path,
    gpu_id: int,
    seed: int,
    wrapper_script: str = "_tr_wrapper.py",
) -> List[str]:
    """
    Build command for _tr_wrapper.py in multi-YAML mode.

    @input: task_id - Task ID from tr_tasks
    @input: yaml_path - Path to unified YAML config
    @input: gpu_id - GPU ID for this slot
    @input: seed - Random seed
    @input: wrapper_script - Wrapper script name
    @output: Command list for subprocess
    """
    return [
        "python3",
        wrapper_script,
        "--task-id",
        task_id,
        "--yaml",
        str(yaml_path),
        "--gpu-id",
        str(gpu_id),
        "--seed",
        str(seed),
    ]


def extract_task_data_from_tr_tasks(
    tr_tasks: List[Dict[str, Any]],
) -> Tuple[List[str], List[str], List[int]]:
    """
    Extract task names, configs, and expert nums from tr_tasks.

    Handles both single-task (data_folder) and multi-task (data_sources) cases.

    @input: tr_tasks - List of task configurations
    @output: (task_names, task_configs, expert_nums)
    """
    all_task_names: List[str] = []
    all_task_configs: List[str] = []
    all_expert_nums: List[int] = []

    for task in tr_tasks:
        if "data_folder" in task:
            task_name = task.get("task_name", task.get("task_id", "unknown"))
            task_config = task.get("task_config", "demo_clean")
            expert_num = task.get("expert_num", 100)
            all_task_names.append(task_name)
            all_task_configs.append(task_config)
            all_expert_nums.append(expert_num)
        elif "data_sources" in task:
            for src in task["data_sources"]:
                all_task_names.append(src.get("task_name", "unknown"))
                all_task_configs.append(src.get("task_config", "demo_clean"))
                all_expert_nums.append(src.get("expert_num", 100))

    return all_task_names, all_task_configs, all_expert_nums


def get_process_data_tasks(tr_tasks: List[Dict[str, Any]]) -> List[str]:
    """
    Get unique task names for process_data step.

    @input: tr_tasks - List of task configurations
    @output: List of unique task names for data processing
    """
    task_names, _, _ = extract_task_data_from_tr_tasks(tr_tasks)
    unique_tasks = []
    seen = set()
    for name in task_names:
        if name not in seen:
            seen.add(name)
            unique_tasks.append(name)
    return unique_tasks
