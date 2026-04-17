#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model-specific checkpoint handling utilities for ACT/DP/TinyVLA.

@input: [str, checkpoint path/name], [str, model type], [Logger, optional]
@output: [int, step number], [None, packages checkpoints in place]
@scenario: [Extract step from checkpoint names and package for evaluation]

Dependencies:
- Python standard library only

Usage:
    from wrapper_base.checkpoint_util import (
        extract_step_from_ckpt_name,
        package_checkpoints_for_eval,
        find_latest_checkpoint,
    )
"""

import logging
import os
import re
import shutil
from enum import Enum
from typing import List, Optional, Tuple


class ModelType(str, Enum):
    ACT = "ACT"
    DP = "DP"
    TinyVLA = "TinyVLA"


# Special step IDs for named checkpoints
STEP_BEST = -2
STEP_LAST = -3


def extract_step_from_ckpt_name(name: str, model_type: str) -> int:
    """
    @input: [str, checkpoint filename or dirname], [str, model type: 'ACT', 'DP', 'TinyVLA']
    @output: [int, step number, or -1/-2/-3 for special cases]
    @scenario: [Extract training step from checkpoint name pattern]

    ACT patterns:
        - policy_step_N.ckpt -> N
        - policy_epoch_N_seed_M.ckpt -> N
        - policy_best.ckpt -> -2
        - policy_last.ckpt -> -3

    DP patterns:
        - step_N.ckpt -> N
        - N.ckpt -> N

    TinyVLA patterns:
        - checkpoint-N -> N (directory)
        - step_N -> N (directory)
    """
    model_type = model_type.upper()

    if model_type == "ACT":
        m = re.search(r"policy_step_(\d+)\.ckpt$", name)
        if m:
            return int(m.group(1))
        m = re.search(r"policy_epoch_(\d+)_seed_\d+\.ckpt$", name)
        if m:
            return int(m.group(1))
        if name == "policy_best.ckpt":
            return STEP_BEST
        if name == "policy_last.ckpt":
            return STEP_LAST
        return -1

    elif model_type == "DP":
        m = re.search(r"step_(\d+)\.ckpt$", name)
        if m:
            return int(m.group(1))
        m = re.search(r"^(\d+)\.ckpt$", name)
        if m:
            return int(m.group(1))
        return -1

    elif model_type == "TinyVLA":
        m = re.search(r"checkpoint-(\d+)$", name)
        if m:
            return int(m.group(1))
        m = re.search(r"step_(\d+)$", name)
        if m:
            return int(m.group(1))
        return -1

    else:
        return -1


def _get_passthrough_files(model_type: str) -> List[str]:
    """Return list of files to copy alongside checkpoint packages."""
    model_type = model_type.upper()
    if model_type == "ACT":
        return ["dataset_stats.pkl", "training_run_manifest.txt", "steps.txt"]
    elif model_type == "DP":
        return ["training_run_manifest.txt", "steps.txt"]
    elif model_type == "TinyVLA":
        return ["training_run_manifest.txt", "steps.txt", "joint_task_spec.json"]
    return []


def package_checkpoints_for_eval(
    ckpt_dir: str,
    model_type: str,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    @input: [str, checkpoint directory path], [str, model type], [Logger, optional]
    @output: [None, creates step_packages subdirectory with organized checkpoints]
    @scenario: [Package all checkpoints for evaluation, organizing by step number]

    Creates a 'step_packages' subdirectory with:
        - step_N/ containing checkpoint and passthrough files
        - step_best/ for ACT/TinyVLA best checkpoints
        - step_last/ for ACT last checkpoints
    """
    if not os.path.isdir(ckpt_dir):
        return

    model_type = model_type.upper()
    passthrough_files = _get_passthrough_files(model_type)
    candidates = _find_checkpoint_candidates(ckpt_dir, model_type)

    if not candidates:
        if logger:
            logger.warning("No checkpoint files found for packaging under %s", ckpt_dir)
        return

    bundle_root = os.path.join(ckpt_dir, "step_packages")
    os.makedirs(bundle_root, exist_ok=True)

    for name, src_path in candidates:
        step_id = extract_step_from_ckpt_name(name, model_type)
        folder_name = _get_folder_name(step_id, name, model_type)
        dst_dir = os.path.join(bundle_root, folder_name)
        os.makedirs(dst_dir, exist_ok=True)

        if model_type == "TinyVLA":
            # TinyVLA uses directories, copy tree
            model_dir = os.path.join(dst_dir, "model")
            shutil.copytree(src_path, model_dir, dirs_exist_ok=True)
        else:
            # ACT/DP use files, copy file
            shutil.copy2(src_path, os.path.join(dst_dir, name))
            # Also create policy_best.ckpt symlink/copy for compatibility
            shutil.copy2(src_path, os.path.join(dst_dir, "policy_best.ckpt"))

        # Copy passthrough files
        for keep in passthrough_files:
            src_keep = os.path.join(ckpt_dir, keep)
            if os.path.exists(src_keep):
                if os.path.isfile(src_keep):
                    shutil.copy2(src_keep, os.path.join(dst_dir, keep))
                elif os.path.isdir(src_keep):
                    shutil.copytree(src_keep, os.path.join(dst_dir, keep), dirs_exist_ok=True)

    if logger:
        logger.info("Packaged %d checkpoints to %s", len(candidates), bundle_root)


def _find_checkpoint_candidates(
    ckpt_dir: str, model_type: str
) -> List[Tuple[str, str]]:
    """Find all checkpoint files/directories in the given directory."""
    candidates = []
    model_type = model_type.upper()

    for name in sorted(os.listdir(ckpt_dir)):
        path = os.path.join(ckpt_dir, name)

        if model_type == "TinyVLA":
            # TinyVLA uses directories
            if name.startswith("checkpoint-") and os.path.isdir(path):
                candidates.append((name, path))
            elif name == "policy_best" and os.path.isdir(path):
                candidates.append((name, path))
        else:
            # ACT/DP use .ckpt files
            if name.endswith(".ckpt") and os.path.isfile(path):
                candidates.append((name, path))

    return candidates


def _get_folder_name(step_id: int, original_name: str, model_type: str) -> str:
    """Generate folder name for a checkpoint based on its step ID."""
    if step_id >= 0:
        return f"step_{step_id}"
    elif step_id == STEP_BEST:
        return "step_best"
    elif step_id == STEP_LAST:
        return "step_last"
    else:
        # Unknown format, use original name
        if model_type.upper() == "TinyVLA":
            return f"step_misc_{original_name.replace('/', '_')}"
        else:
            return f"step_misc_{os.path.splitext(original_name)[0]}"


def find_latest_checkpoint(
    output_dir: str, model_type: str, prefer_best: bool = True
) -> Optional[str]:
    """
    @input: [str, output directory containing checkpoints], [str, model type], [bool, prefer best over latest]
    @output: [Optional[str], path to the best/latest checkpoint, or None if not found]
    @scenario: [Find the most suitable checkpoint for evaluation]

    If prefer_best is True, returns 'policy_best' if it exists, otherwise the highest step.
    """
    if not os.path.isdir(output_dir):
        return None

    model_type = model_type.upper()

    # Check for policy_best first
    if prefer_best:
        if model_type == "TinyVLA":
            best_path = os.path.join(output_dir, "policy_best")
            if os.path.isdir(best_path):
                return best_path
        else:
            best_path = os.path.join(output_dir, "policy_best.ckpt")
            if os.path.isfile(best_path):
                return best_path

    # Find the checkpoint with the highest step
    candidates = _find_checkpoint_candidates(output_dir, model_type)
    if not candidates:
        return None

    best_step = -1
    best_path = None
    for name, path in candidates:
        step = extract_step_from_ckpt_name(name, model_type)
        if step > best_step:
            best_step = step
            best_path = path

    return best_path


def find_workspace_checkpoint_dir(
    base_dir: str, save_name: str, seed: int, outputs_subdir: str = "data/outputs"
) -> Optional[str]:
    """
    @input: [str, base directory], [str, save name], [int, seed], [str, outputs subdirectory]
    @output: [Optional[str], path to checkpoint directory, or None]
    @scenario: [Find checkpoint directory in workspace outputs (used by DP)]

    Looks for directories matching pattern: outputs_subdir/.../checkpoints/{save_name}-{seed}
    """
    target_leaf = f"{save_name}-{seed}"
    outputs_root = os.path.join(base_dir, outputs_subdir)

    if not os.path.isdir(outputs_root):
        return None

    found = []
    for root, dir_names, _ in os.walk(outputs_root):
        for dir_name in dir_names:
            if dir_name == target_leaf and os.path.basename(root) == "checkpoints":
                found.append(os.path.join(root, dir_name))

    if not found:
        return None

    # Return most recently modified
    found.sort(key=os.path.getmtime, reverse=True)
    return found[0]
