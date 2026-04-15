"""
Checkpoint resolution utility for DP and ACT models.

@input: [str, ckpt_dir], [str or int or None, checkpoint_num]
@output: [str, resolved checkpoint file path]
@scenario: [Resolve checkpoint path with best > max_step > specified logic]

Logic:
1. If checkpoint_num == "best": use best checkpoint if exists
2. If checkpoint_num is int: use that specific step checkpoint
3. If checkpoint_num is None or not found: prefer best, else max step

DP checkpoints: {step}.ckpt, best_val.ckpt
ACT checkpoints: policy_epoch_{step}_seed_{seed}.ckpt, policy_best.ckpt, policy_last.ckpt
"""

import os
import re
from pathlib import Path
from typing import Optional, Union


def resolve_dp_checkpoint(
    ckpt_dir: str,
    checkpoint_num: Optional[Union[str, int]] = None,
) -> str:
    """
    Resolve DP checkpoint path.

    @input: [str, ckpt_dir], [str or int or None, checkpoint_num]
    @output: [str, absolute checkpoint file path]
    @scenario: [Find best or max step checkpoint for DP model]
    """
    ckpt_path = Path(ckpt_dir)
    if not ckpt_path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    best_ckpt = ckpt_path / "best_val.ckpt"

    if checkpoint_num is not None:
        checkpoint_str = str(checkpoint_num)
        if checkpoint_str.lower() == "best":
            if best_ckpt.is_file():
                return str(best_ckpt.resolve())
            raise FileNotFoundError(f"Best checkpoint not found: {best_ckpt}")
        target_ckpt = ckpt_path / f"{checkpoint_str}.ckpt"
        if target_ckpt.is_file():
            return str(target_ckpt.resolve())
        raise FileNotFoundError(f"Checkpoint not found: {target_ckpt}")

    if best_ckpt.is_file():
        return str(best_ckpt.resolve())

    step_ckpts = []
    for f in ckpt_path.iterdir():
        if f.is_file() and f.suffix == ".ckpt":
            m = re.match(r"^(\d+)\.ckpt$", f.name)
            if m:
                step_ckpts.append((int(m.group(1)), f))

    if not step_ckpts:
        raise FileNotFoundError(f"No step checkpoints found in: {ckpt_dir}")

    step_ckpts.sort(key=lambda x: x[0], reverse=True)
    return str(step_ckpts[0][1].resolve())


def resolve_act_checkpoint(
    ckpt_dir: str,
    checkpoint_num: Optional[Union[str, int]] = None,
    seed: int = 0,
) -> str:
    """
    Resolve ACT checkpoint path.

    @input: [str, ckpt_dir], [str or int or None, checkpoint_num], [int, seed]
    @output: [str, absolute checkpoint file path]
    @scenario: [Find best or max step checkpoint for ACT model]
    """
    ckpt_path = Path(ckpt_dir)
    if not ckpt_path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    best_ckpt = ckpt_path / "policy_best.ckpt"
    last_ckpt = ckpt_path / "policy_last.ckpt"

    if checkpoint_num is not None:
        checkpoint_str = str(checkpoint_num)
        if checkpoint_str.lower() == "best":
            if best_ckpt.is_file():
                return str(best_ckpt.resolve())
            raise FileNotFoundError(f"Best checkpoint not found: {best_ckpt}")
        target_ckpt = ckpt_path / f"policy_epoch_{checkpoint_str}_seed_{seed}.ckpt"
        if target_ckpt.is_file():
            return str(target_ckpt.resolve())
        raise FileNotFoundError(f"Checkpoint not found: {target_ckpt}")

    if best_ckpt.is_file():
        return str(best_ckpt.resolve())

    step_ckpts = []
    for f in ckpt_path.iterdir():
        if f.is_file() and f.suffix == ".ckpt":
            m = re.match(r"^policy_epoch_(\d+)_seed_(\d+)\.ckpt$", f.name)
            if m:
                step_ckpts.append((int(m.group(1)), int(m.group(2)), f))

    if not step_ckpts:
        if last_ckpt.is_file():
            return str(last_ckpt.resolve())
        raise FileNotFoundError(f"No checkpoints found in: {ckpt_dir}")

    step_ckpts.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return str(step_ckpts[0][2].resolve())


def get_checkpoint_info(ckpt_dir: str, policy_type: str = "DP") -> dict:
    """
    Get checkpoint info for debugging/logging.

    @input: [str, ckpt_dir], [str, policy_type "DP" or "ACT"]
    @output: [dict, checkpoint info with available steps and best status]
    @scenario: [Inspect checkpoint directory contents]
    """
    ckpt_path = Path(ckpt_dir)
    info = {
        "dir": str(ckpt_dir),
        "exists": ckpt_path.is_dir(),
        "has_best": False,
        "has_last": False,
        "step_ckpts": [],
    }

    if not ckpt_path.is_dir():
        return info

    if policy_type == "DP":
        info["has_best"] = (ckpt_path / "best_val.ckpt").is_file()
        for f in ckpt_path.iterdir():
            if f.is_file() and f.suffix == ".ckpt":
                m = re.match(r"^(\d+)\.ckpt$", f.name)
                if m:
                    info["step_ckpts"].append(int(m.group(1)))
    else:
        info["has_best"] = (ckpt_path / "policy_best.ckpt").is_file()
        info["has_last"] = (ckpt_path / "policy_last.ckpt").is_file()
        for f in ckpt_path.iterdir():
            if f.is_file() and f.suffix == ".ckpt":
                m = re.match(r"^policy_epoch_(\d+)_seed_(\d+)\.ckpt$", f.name)
                if m:
                    info["step_ckpts"].append(int(m.group(1)))

    info["step_ckpts"].sort(reverse=True)
    return info
