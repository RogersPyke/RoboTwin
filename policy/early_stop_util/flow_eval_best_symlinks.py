"""
Eval flow helpers: ensure checkpoint paths used by deploy/wrappers resolve to best checkpoints.

Symlinks are recreated each run (remove existing link or file at the same name, then ln -s).
Paths match policy/DP/_ev_wrapper.py, policy/ACT/_ev_wrapper.py, policy/TinyVLA/_ev_wrapper.py.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import yaml


def _load_ev_cfg(policy_dir: Path, stem: str) -> Dict[str, Any]:
    base = stem if stem.endswith(".yaml") else f"{stem}.yaml"
    path = policy_dir / "_ev_cfg" / base
    if not path.is_file():
        raise FileNotFoundError(f"Missing eval config: {path}")
    with open(path, encoding="utf-8", errors="replace") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict) or not cfg:
        raise ValueError(f"Invalid or empty YAML: {path}")
    return cfg


def _parse_task_rows(cfg: Dict[str, Any], key: str) -> List[Tuple[str, str, int]]:
    rows = cfg.get(key)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{key} must be a non-empty list.")
    out: List[Tuple[str, str, int]] = []
    for i, row in enumerate(rows):
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            raise ValueError(f"{key}[{i}] must be [task_name, task_config, expert_num], got {row!r}")
        out.append((str(row[0]).strip(), str(row[1]).strip(), int(row[2])))
    return out


def _resolve_seed(env_key: str, flow_seed: int | None, yaml_seed: int) -> int:
    v = os.environ.get(env_key, "").strip()
    if v:
        return int(v)
    if flow_seed is not None:
        return int(flow_seed)
    return int(yaml_seed)


def _remove_then_symlink(link_path: Path, target_relative: str, tag: str) -> None:
    """
    @input: link_path [Path, file to create], target_relative [str, relative to link_path.parent]
    @output: None
    @scenario: Replace any existing file or symlink at link_path with a new symlink.
    """
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    link_path.symlink_to(target_relative, target_is_directory=False)
    print(f"{tag} {link_path} -> {target_relative}", flush=True)


def prepare_dp_flow_best_symlinks(
    policy_dp_dir: Path,
    repo_root: Path,
    stems: Sequence[str],
    flow_seed: int | None,
) -> None:
    """
    DP deploy loads policy/DP/checkpoints/<slug>/{checkpoint_num}.ckpt.
    Training early-stop saves best_val.ckpt. Symlink numbered ckpt -> best_val.ckpt.
    """
    tag = "[flow][best_symlink][DP]"
    for stem in stems:
        try:
            cfg = _load_ev_cfg(policy_dp_dir, stem)
            train_rows = _parse_task_rows(cfg, "TRAIN_TASKS")
            train_task_slug = "__".join([r[0] for r in train_rows])
            train_config_slug = "__".join([r[1] for r in train_rows])
            train_total = int(sum(r[2] for r in train_rows))
            checkpoint_expert_data_num = int(cfg.get("CHECKPOINT_EXPERT_DATA_NUM", train_total))
            v_ced = os.environ.get("DP_FLOW_CHECKPOINT_EXPERT_DATA_NUM", "").strip()
            if v_ced:
                checkpoint_expert_data_num = int(v_ced)
            checkpoint_num = int(cfg.get("CHECKPOINT_NUM", 600))
            v_ck = os.environ.get("DP_FLOW_CHECKPOINT_NUM", "").strip()
            if v_ck:
                checkpoint_num = int(v_ck)
            seed = _resolve_seed("DP_FLOW_SEED", flow_seed, int(cfg["EVAL_SEED"]))
            rel_dir = (
                Path("policy/DP/checkpoints")
                / f"{train_task_slug}-{train_config_slug}-{checkpoint_expert_data_num}-{seed}"
            )
            ckpt_dir = (repo_root / rel_dir).resolve()
            best = ckpt_dir / "best_val.ckpt"
            link = ckpt_dir / f"{checkpoint_num}.ckpt"
            if not best.is_file():
                print(
                    f"{tag} skip stem={stem}: missing {best}",
                    flush=True,
                )
                continue
            _remove_then_symlink(link, "best_val.ckpt", tag)
        except Exception as exc:
            print(f"{tag} stem={stem} failed: {exc}", flush=True)
            raise


def prepare_act_flow_best_symlinks(
    policy_act_dir: Path,
    repo_root: Path,
    stems: Sequence[str],
    flow_seed: int | None,
) -> None:
    """
    ACT deploy loads policy_last.ckpt. Symlink policy_last.ckpt -> policy_best.ckpt.
    """
    tag = "[flow][best_symlink][ACT]"
    for stem in stems:
        try:
            cfg = _load_ev_cfg(policy_act_dir, stem)
            rows = _parse_task_rows(cfg, "TRAIN_TASKS")
            names = [r[0] for r in rows]
            cfgs = [r[1] for r in rows]
            nums = [r[2] for r in rows]
            combined_task_slug = "__".join(names)
            ckpt_setting = "__".join(cfgs)
            expert_data_num = int(sum(nums))
            _resolve_seed("ACT_FLOW_SEED", flow_seed, int(cfg["EVAL_SEED"]))
            rel = (
                Path("policy/ACT/act_ckpt")
                / f"act-{combined_task_slug}"
                / f"{ckpt_setting}-{expert_data_num}"
            )
            ckpt_dir = (repo_root / rel).resolve()
            best = ckpt_dir / "policy_best.ckpt"
            last = ckpt_dir / "policy_last.ckpt"
            if not best.is_file():
                print(f"{tag} skip stem={stem}: missing {best}", flush=True)
                continue
            _remove_then_symlink(last, "policy_best.ckpt", tag)
        except Exception as exc:
            print(f"{tag} stem={stem} failed: {exc}", flush=True)
            raise


def prepare_tinyvla_flow_best_symlinks(
    policy_tvla_dir: Path,
    repo_root: Path,
    stems: Sequence[str],
    flow_seed: int | None,
) -> None:
    """
    TinyVLA eval uses HF files under output_dir when USE_POLICY_BEST is false.
    Symlink each top-level file from policy_best/ into output_dir so from_pretrained sees best weights.
    """
    tag = "[flow][best_symlink][TinyVLA]"
    for stem in stems:
        try:
            cfg = _load_ev_cfg(policy_tvla_dir, stem)
            if "TRAIN_TASKS" not in cfg:
                print(f"{tag} skip stem={stem}: no TRAIN_TASKS (legacy config)", flush=True)
                continue
            train_rows = _parse_task_rows(cfg, "TRAIN_TASKS")
            task_slug = "__".join([r[0] for r in train_rows])
            config_slug = "__".join([r[1] for r in train_rows])
            total_episodes = int(sum(r[2] for r in train_rows))
            _resolve_seed("TVLA_FLOW_SEED", flow_seed, int(cfg["EVAL_SEED"]))
            rel_out = (
                Path("policy/TinyVLA/tinyvla_ckpt")
                / f"tinyvla-{task_slug}"
                / f"{config_slug}-{total_episodes}"
            )
            output_dir = (repo_root / rel_out).resolve()
            best_dir = output_dir / "policy_best"
            if not best_dir.is_dir():
                print(f"{tag} skip stem={stem}: missing dir {best_dir}", flush=True)
                continue
            for src in sorted(best_dir.iterdir()):
                if not src.is_file():
                    continue
                dest = output_dir / src.name
                _remove_then_symlink(dest, f"policy_best/{src.name}", tag)
        except Exception as exc:
            print(f"{tag} stem={stem} failed: {exc}", flush=True)
            raise
