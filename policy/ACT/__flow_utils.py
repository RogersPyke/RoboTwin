# -*- coding: utf-8 -*-
"""
Helpers for ACT serial train/eval flow (__flow.py).
All comments and log messages in ENGLISH (ASCII).
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, List, Tuple

import yaml

LOGGER = logging.getLogger("act_flow")


def repo_paths(act_dir: Path) -> Tuple[Path, Path, Path]:
    """
    @input: act_dir Path to policy/ACT
    @output: (rw_root RoboTwin, data_root, repo_root thesis-draft-dev)
    @scenario: Resolve RoboTwin and monorepo roots like legacy __flow.sh
    """
    act_dir = act_dir.resolve()
    rw_root = act_dir.parent.parent
    repo_root = act_dir.parent.parent.parent.parent
    data_root = rw_root / "data"
    return rw_root, data_root, repo_root


def discover_pos_neg_pairs(data_root: Path) -> List[Tuple[str, str]]:
    """
    @input: data_root with subdirs un* and matching positive names
    @output: sorted list of (pos, neg) e.g. (hanging_mug, unhanging_mug)
    @scenario: Same pairing rule as __flow.sh (un* -> strip un prefix for pos)
    """
    if not data_root.is_dir():
        return []
    pairs: List[Tuple[str, str]] = []
    for p in sorted(data_root.iterdir()):
        if not p.is_dir():
            continue
        neg = p.name
        if not neg.startswith("un"):
            continue
        pos = neg[2:]
        if (data_root / pos).is_dir():
            pairs.append((pos, neg))
    pairs.sort(key=lambda x: (x[0], x[1]))
    return pairs


def parse_pairs_env(raw: str) -> List[Tuple[str, str]]:
    """
    @input: newline-separated lines, each "pos_task,neg_task" (optional # comments)
    @output: list of (pos, neg) in line order
    @scenario: PAIRS from __flow.sh; no structured literal parsing
    """
    text = (raw or "").strip()
    if not text:
        return []
    pairs: List[Tuple[str, str]] = []
    for i, line in enumerate(text.splitlines()):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if "," not in line:
            raise ValueError(f"PAIRS line {i + 1}: need pos,neg (comma): {line!r}")
        pos, neg = line.split(",", 1)
        pos, neg = pos.strip(), neg.strip()
        if not pos or not neg:
            raise ValueError(f"PAIRS line {i + 1}: empty task name: {line!r}")
        pairs.append((pos, neg))
    return pairs


def snapshot_eval_leaves(eval_result_root: Path) -> List[str]:
    """
    @input: eval_result root directory
    @output: sorted list of dir paths at depth 5 under root (same as find -mindepth 5 -maxdepth 5)
    @scenario: Diff before/after eval to find new dirs for optional rm after upload
    """
    if not eval_result_root.is_dir():
        return []
    root = eval_result_root.resolve()
    out: List[str] = []
    for dirpath, _, _ in os.walk(root):
        rel = Path(dirpath).relative_to(root)
        if len(rel.parts) == 5:
            out.append(dirpath)
    return sorted(out)


def load_train_tasks_rows(act_dir: Path, cfg_name: str, *, from_eval_cfg: bool) -> List[List]:
    """
    @input: cfg_name without .yaml; from_eval_cfg True -> _ev_cfg, else _tr_cfg
    @output: TRAIN_TASKS rows from YAML
    @scenario: Eval prep must use _ev_cfg so process_data matches checkpoint layout used by _ev_wrapper
    """
    sub = "_ev_cfg" if from_eval_cfg else "_tr_cfg"
    path = act_dir / sub / f"{cfg_name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not cfg:
        raise ValueError(f"empty yaml: {path}")
    rows = cfg.get("TRAIN_TASKS") or []
    if not rows:
        raise ValueError(f"TRAIN_TASKS empty: {path}")
    for i, row in enumerate(rows):
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            raise ValueError(f"TRAIN_TASKS[{i}] must be [task, config, n]: {row!r}")
    return [list(row) for row in rows]


def run_process_data(
    act_dir: Path,
    cfg_name: str,
    *,
    from_eval_cfg: bool,
    record: Callable[[str, str, str], None],
    tag: str,
) -> bool:
    """
    @input: cfg_name; from_eval_cfg selects _ev_cfg vs _tr_cfg for TRAIN_TASKS
    @output: True on success
    @scenario: Run process_data.sh once per TRAIN_TASKS row before train or eval
    """
    try:
        rows = load_train_tasks_rows(act_dir, cfg_name, from_eval_cfg=from_eval_cfg)
    except Exception as exc:
        LOGGER.error("process_data load failed: %s", exc)
        record("FAIL", "process_data", tag)
        return False
    for row in rows:
        cmd = ["bash", "process_data.sh", str(row[0]), str(row[1]), str(row[2])]
        try:
            subprocess.run(cmd, check=True, cwd=str(act_dir))
        except subprocess.CalledProcessError:
            record("FAIL", "process_data", tag)
            return False
    record("OK", "process_data", tag)
    return True


def clear_processed(act_dir: Path, record: Callable[[str, str, str], None]) -> bool:
    """
    @input: ACT directory
    @output: True if processed_data cleared and SIM_TASK_CONFIGS.json reset
    @scenario: Start each pos/neg pair from clean processed_data
    """
    pd = act_dir / "processed_data"
    try:
        if pd.is_dir():
            import shutil

            shutil.rmtree(pd)
        pd.mkdir(parents=True, exist_ok=True)
        stc = act_dir / "SIM_TASK_CONFIGS.json"
        stc.write_text("{}\n", encoding="utf-8")
    except OSError as exc:
        LOGGER.error("clear_processed: %s", exc)
        record("FAIL", "clear_processed", str(pd))
        return False
    record("OK", "clear_processed", "reset_SIM_TASK_CONFIGS.json")
    return True


def upload_ckpt(ms_up: Path, act_dir: Path, ns: str, repo: str, record: Callable[[str, str, str], None], tag: str) -> bool:
    """Upload act_ckpt tree via ms_up.py."""
    ckpt = act_dir / "act_ckpt"
    ckpt.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ms_up),
        str(ckpt),
        "--namespace",
        ns,
        "--repo-name",
        repo,
        "--repo-type",
        "model",
        "--batch-size",
        "0",
        "--max-retries",
        "5",
    ]
    try:
        subprocess.run(cmd, check=True)
        record("OK", "upload_ckpt", tag)
        return True
    except subprocess.CalledProcessError:
        record("FAIL", "upload_ckpt", tag)
        return False


def upload_eval(ms_up: Path, eval_root: Path, ns: str, repo: str, record: Callable[[str, str, str], None], tag: str) -> bool:
    """Upload eval_result tree via ms_up.py."""
    eval_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ms_up),
        str(eval_root),
        "--namespace",
        ns,
        "--repo-name",
        repo,
        "--repo-type",
        "dataset",
        "--batch-size",
        "0",
        "--max-retries",
        "5",
    ]
    try:
        subprocess.run(cmd, check=True)
        record("OK", "upload_eval", tag)
        return True
    except subprocess.CalledProcessError:
        record("FAIL", "upload_eval", tag)
        return False
