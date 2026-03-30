#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Serial DP train/eval flow + ModelScope upload (ms_up.py at monorepo root).
Requires: export MS_TOKEN=<sdk_token>

Single-task: _tr_cfg/flow_single_<task>.yaml + _train.sh; eval: _ev_cfg/flow_single_<task>.yaml + _eval.sh
Joint: _tr_cfg/flow_joint_<pos>.yaml + _ev_cfg/flow_joint_<pos>.yaml

DP_FLOW_GPU overrides YAML TRAIN_GPU_ID / EVAL_GPU_ID in wrappers when set.
DP_FLOW_DRY_RUN=1 or --dry-run: prechecks + YAML parse only.

Checkpoint dir (train): policy/DP/checkpoints/<task_slug>-<cfg_slug>-<n_eps>-<seed>/
Eval result: eval_result/<task>/DP/<task_config>/<ckpt_setting>/<timestamp>/ (eval_policy.py)

Usage:
  ./__flow.sh [--dry-run ...]
  python3 __flow.py --pairs "$(printf '%s\\n' "${PAIRS[@]}")"
"""
from __future__ import annotations

import argparse
import atexit
import logging
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

from __flow_utils import (
    clear_dp_data,
    parse_pairs_env,
    repo_paths,
    run_process_data,
    snapshot_eval_leaves,
    upload_ckpt,
    upload_eval,
)

N_EPISODES_SINGLE = 100
N_EPISODES_JOINT = 200
TCFG = "demo_clean"
FLOW_TRAIN_SEED = 0

MS_CKPT_REPO = "robotwin-dp-ckpt"
MS_EVAL_REPO = "robotwin-dp-eval-result"
MS_NS = "rogerspyke"

LOGGER = logging.getLogger("dp_flow")


class FlowRecorder:
    """Append status lines for end-of-run summary."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._printed = False

    def record(self, st: str, name: str, extra: str = "") -> None:
        line = f"{st}|{name}|{extra}\n"
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line)
        LOGGER.info("[flow] %s %s %s", st, name, extra)

    def print_summary(self) -> None:
        if self._printed:
            return
        self._printed = True
        print("========== __flow (DP) run summary ==========", file=sys.stderr)
        if self.path.is_file():
            text = self.path.read_text(encoding="utf-8")
            print(text, end="" if text.endswith("\n") else "\n", file=sys.stderr)
            oks = sum(1 for ln in text.splitlines() if ln.startswith("OK|"))
            fails = sum(1 for ln in text.splitlines() if ln.startswith("FAIL|"))
            print(f"---------- totals: OK={oks} FAIL={fails} ----------", file=sys.stderr)
        else:
            print("(no status file)", file=sys.stderr)


def _setup_logging(dp_dir: Path) -> None:
    log_dir = dp_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="[__flow] [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )


def _yaml_ok(path: Path) -> bool:
    try:
        with open(path, encoding="utf-8") as f:
            yaml.safe_load(f)
        return True
    except Exception:
        return False


def _precheck(
    dp_dir: Path,
    data_root: Path,
    repo_root: Path,
    record: FlowRecorder,
) -> Path:
    if not os.environ.get("MS_TOKEN", "").strip():
        print("[ERROR] MS_TOKEN is empty or unset.", file=sys.stderr)
        record.record("FAIL", "precheck", "MS_TOKEN_missing")
        sys.exit(1)
    ms_up = repo_root / "ms_up.py"
    if not ms_up.is_file():
        print(f"[ERROR] ms_up.py not found: {ms_up}", file=sys.stderr)
        record.record("FAIL", "precheck", "ms_up_missing")
        sys.exit(1)
    if not data_root.is_dir():
        print(f"[ERROR] data root not found: {data_root}", file=sys.stderr)
        record.record("FAIL", "precheck", "data_root_missing")
        sys.exit(1)
    if not (dp_dir / "process_data.sh").is_file():
        print(f"[ERROR] process_data.sh missing under {dp_dir}", file=sys.stderr)
        record.record("FAIL", "precheck", "process_data_sh_missing")
        sys.exit(1)
    return ms_up


def _run_train(dp_dir: Path, cfg_name: str, env_gpu: str) -> bool:
    env = os.environ.copy()
    env["DP_FLOW_GPU"] = env_gpu
    env["PYTHONNOUSERSITE"] = "1"
    try:
        subprocess.run(
            ["bash", "_train.sh", cfg_name],
            check=True,
            cwd=str(dp_dir),
            env=env,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def _run_eval(dp_dir: Path, cfg_name: str, env_gpu: str) -> bool:
    env = os.environ.copy()
    env["DP_FLOW_GPU"] = env_gpu
    env["PYTHONNOUSERSITE"] = "1"
    try:
        subprocess.run(
            ["bash", "_eval.sh", cfg_name],
            check=True,
            cwd=str(dp_dir),
            env=env,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def _ckpt_folder_single(task: str) -> str:
    return f"{task}-{TCFG}-{N_EPISODES_SINGLE}-{FLOW_TRAIN_SEED}"


def _ckpt_folder_joint(pos: str, neg: str) -> str:
    return f"{pos}__{neg}-{TCFG}__{TCFG}-{N_EPISODES_JOINT}-{FLOW_TRAIN_SEED}"


def _rm_ckpt_folder(dp_dir: Path, folder: str, record: FlowRecorder) -> None:
    full = dp_dir / "checkpoints" / folder
    if not full.is_dir():
        return
    try:
        shutil.rmtree(full)
        record.record("OK", "rm_ckpt", str(full))
    except OSError:
        record.record("FAIL", "rm_ckpt", str(full))


def _rm_eval_paths(paths: List[str], record: FlowRecorder) -> None:
    for p in paths:
        if not p or not os.path.isdir(p):
            continue
        try:
            shutil.rmtree(p)
            record.record("OK", "rm_eval", p)
        except OSError:
            record.record("FAIL", "rm_eval", p)


def _diff_new_dirs(before: List[str], after: List[str]) -> List[str]:
    b, a = set(before), set(after)
    return sorted(a - b)


def run_train_single(
    dp_dir: Path,
    task: str,
    tag: str,
    gpu_id: str,
    record: FlowRecorder,
    ms_up: Path,
    ckpt_uploaded_ok: Dict[str, int],
) -> None:
    cfg = f"flow_single_{task}"
    ckpt_key = _ckpt_folder_single(task)
    ckpt_uploaded_ok[ckpt_key] = 0
    if not (dp_dir / "_tr_cfg" / f"{cfg}.yaml").is_file():
        print(f"[ERROR] missing _tr_cfg/{cfg}.yaml", file=sys.stderr)
        record.record("FAIL", "train", tag)
        return
    if not run_process_data(dp_dir, cfg, from_eval_cfg=False, record=record.record, tag=f"pre_train_{tag}"):
        record.record("FAIL", "train", tag)
        return
    if _run_train(dp_dir, cfg, gpu_id):
        record.record("OK", "train", tag)
        if upload_ckpt(ms_up, dp_dir, MS_NS, MS_CKPT_REPO, record.record, tag):
            ckpt_uploaded_ok[ckpt_key] = 1
    else:
        record.record("FAIL", "train", tag)


def run_eval_single(
    dp_dir: Path,
    task: str,
    tag: str,
    gpu_id: str,
    record: FlowRecorder,
    ms_up: Path,
    rw_root: Path,
    ckpt_uploaded_ok: Dict[str, int],
) -> None:
    evcfg = f"flow_single_{task}"
    ckpt_key = _ckpt_folder_single(task)
    eval_root = rw_root / "eval_result"
    if not (dp_dir / "_ev_cfg" / f"{evcfg}.yaml").is_file():
        print(f"[ERROR] missing _ev_cfg/{evcfg}.yaml", file=sys.stderr)
        record.record("FAIL", "eval", tag)
        return
    if not run_process_data(dp_dir, evcfg, from_eval_cfg=True, record=record.record, tag=f"pre_eval_{tag}"):
        record.record("FAIL", "eval", tag)
        return
    snap_b = snapshot_eval_leaves(eval_root)
    if _run_eval(dp_dir, evcfg, gpu_id):
        record.record("OK", "eval", tag)
        snap_a = snapshot_eval_leaves(eval_root)
        new_dirs = _diff_new_dirs(snap_b, snap_a)
        if upload_eval(ms_up, eval_root, MS_NS, MS_EVAL_REPO, record.record, tag):
            _rm_eval_paths(new_dirs, record)
            if ckpt_uploaded_ok.get(ckpt_key, 0) == 1:
                _rm_ckpt_folder(dp_dir, ckpt_key, record)
    else:
        record.record("FAIL", "eval", tag)


def run_train_joint(
    dp_dir: Path,
    stem: str,
    pos: str,
    neg: str,
    gpu_id: str,
    record: FlowRecorder,
    ms_up: Path,
    ckpt_uploaded_ok: Dict[str, int],
) -> None:
    tag = f"joint_{pos}__{neg}"
    ckpt_key = _ckpt_folder_joint(pos, neg)
    ckpt_uploaded_ok[ckpt_key] = 0
    if not run_process_data(dp_dir, stem, from_eval_cfg=False, record=record.record, tag=f"pre_train_joint_{tag}"):
        record.record("FAIL", "train_joint", tag)
        return
    if _run_train(dp_dir, stem, gpu_id):
        record.record("OK", "train_joint", tag)
        if upload_ckpt(ms_up, dp_dir, MS_NS, MS_CKPT_REPO, record.record, tag):
            ckpt_uploaded_ok[ckpt_key] = 1
    else:
        record.record("FAIL", "train_joint", tag)


def run_eval_joint(
    dp_dir: Path,
    stem: str,
    pos: str,
    neg: str,
    gpu_id: str,
    record: FlowRecorder,
    ms_up: Path,
    rw_root: Path,
    ckpt_uploaded_ok: Dict[str, int],
) -> None:
    tag = f"joint_{pos}__{neg}"
    ckpt_key = _ckpt_folder_joint(pos, neg)
    eval_root = rw_root / "eval_result"
    if not run_process_data(dp_dir, stem, from_eval_cfg=True, record=record.record, tag=f"pre_eval_joint_{tag}"):
        record.record("FAIL", "eval_joint", tag)
        return
    snap_b = snapshot_eval_leaves(eval_root)
    if _run_eval(dp_dir, stem, gpu_id):
        record.record("OK", "eval_joint", tag)
        snap_a = snapshot_eval_leaves(eval_root)
        new_dirs = _diff_new_dirs(snap_b, snap_a)
        if upload_eval(ms_up, eval_root, MS_NS, MS_EVAL_REPO, record.record, tag):
            _rm_eval_paths(new_dirs, record)
            if ckpt_uploaded_ok.get(ckpt_key, 0) == 1:
                _rm_ckpt_folder(dp_dir, ckpt_key, record)
    else:
        record.record("FAIL", "eval_joint", tag)


def dry_run(dp_dir: Path, pairs: List[Tuple[str, str]], record: FlowRecorder) -> None:
    for pos, neg in pairs:
        stem = f"flow_joint_{pos}"
        need = [
            dp_dir / "_tr_cfg" / f"flow_single_{pos}.yaml",
            dp_dir / "_tr_cfg" / f"flow_single_{neg}.yaml",
            dp_dir / "_tr_cfg" / f"{stem}.yaml",
            dp_dir / "_ev_cfg" / f"flow_single_{pos}.yaml",
            dp_dir / "_ev_cfg" / f"flow_single_{neg}.yaml",
            dp_dir / "_ev_cfg" / f"{stem}.yaml",
        ]
        for f in need:
            if not f.is_file():
                print(f"[DRY-RUN] FAIL missing file: {f}", file=sys.stderr)
                record.record("FAIL", "dry_run", f"missing_{f.name}")
                sys.exit(1)
            if not _yaml_ok(f):
                print(f"[DRY-RUN] FAIL invalid YAML: {f}", file=sys.stderr)
                record.record("FAIL", "dry_run", f"bad_yaml_{f.name}")
                sys.exit(1)
        print(
            f"[DRY-RUN] OK pair {pos}/{neg} (6 stages + uploads when not dry-run)",
            file=sys.stderr,
        )
    for script in ("__flow.py", "__flow_utils.py"):
        src = dp_dir / script
        try:
            compile(src.read_text(encoding="utf-8"), str(src), "exec")
        except SyntaxError as exc:
            print(f"[DRY-RUN] FAIL syntax {script}: {exc}", file=sys.stderr)
            record.record("FAIL", "dry_run", f"syntax_{script}")
            sys.exit(1)
    record.record("OK", "dry_run", "all_pairs_ok")
    print("[DRY-RUN] success. No train/eval/upload executed.", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description="DP serial train/eval flow.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prechecks and YAML parse only (also if DP_FLOW_DRY_RUN=1).",
    )
    parser.add_argument(
        "--pairs",
        required=True,
        metavar="TEXT",
        help="Required. One pos_task,neg_task per line (same format as __flow.sh PAIRS).",
    )
    args = parser.parse_args()
    dry = args.dry_run or os.environ.get("DP_FLOW_DRY_RUN", "").strip() == "1"

    dp_dir = Path(__file__).resolve().parent
    os.chdir(dp_dir)
    os.environ["PYTHONNOUSERSITE"] = "1"

    _setup_logging(dp_dir)
    rw_root, data_root, repo_root = repo_paths(dp_dir)
    status_path = dp_dir / "logs" / f"__flow_status_{os.getpid()}.txt"
    recorder = FlowRecorder(status_path)
    atexit.register(recorder.print_summary)

    def _sig(_sig=None, _frame=None) -> None:
        recorder.print_summary()
        sys.exit(130)

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    gpu_id = os.environ.get("DP_FLOW_GPU", "").strip() or "0"
    ms_up = _precheck(dp_dir, data_root, repo_root, recorder)

    pairs_text = args.pairs.strip()
    if not pairs_text:
        print("[ERROR] --pairs is empty.", file=sys.stderr)
        recorder.record("FAIL", "precheck", "empty_pairs")
        return 1
    try:
        pairs = parse_pairs_env(pairs_text)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        recorder.record("FAIL", "precheck", "bad_pairs")
        return 1
    if not pairs:
        print("[ERROR] --pairs has no valid lines.", file=sys.stderr)
        recorder.record("FAIL", "precheck", "empty_pairs")
        return 1

    if dry:
        print(
            f"[DRY-RUN] DP_DIR={dp_dir} RW_ROOT={rw_root} pair_count={len(pairs)}",
            file=sys.stderr,
        )
        dry_run(dp_dir, pairs, recorder)
        return 0

    ckpt_uploaded_ok: Dict[str, int] = {}

    for pos, neg in pairs:
        stem = f"flow_joint_{pos}"
        need_tr = [
            dp_dir / "_tr_cfg" / f"{stem}.yaml",
            dp_dir / "_tr_cfg" / f"flow_single_{pos}.yaml",
            dp_dir / "_tr_cfg" / f"flow_single_{neg}.yaml",
        ]
        need_ev = [
            dp_dir / "_ev_cfg" / f"{stem}.yaml",
            dp_dir / "_ev_cfg" / f"flow_single_{pos}.yaml",
            dp_dir / "_ev_cfg" / f"flow_single_{neg}.yaml",
        ]
        if not all(f.is_file() for f in need_tr + need_ev):
            print(
                f"[WARN] skip pair {pos}/{neg}: missing joint or flow_single _tr_cfg/_ev_cfg",
                file=sys.stderr,
            )
            recorder.record("FAIL", "skip_pair", f"{pos}__{neg}")
            continue

        print(f"========== DP FLOW {pos} / {neg} ==========", file=sys.stderr)
        if not clear_dp_data(dp_dir, recorder.record):
            print(f"[ERROR] failed to clear DP data for pair {pos}/{neg}", file=sys.stderr)
            recorder.record("FAIL", "clear_dp_data", f"{pos}__{neg}")
            continue

        run_train_single(dp_dir, pos, f"pos_{pos}__{neg}", gpu_id, recorder, ms_up, ckpt_uploaded_ok)
        run_eval_single(dp_dir, pos, f"pos_{pos}__{neg}", gpu_id, recorder, ms_up, rw_root, ckpt_uploaded_ok)
        run_train_single(dp_dir, neg, f"neg_{pos}__{neg}", gpu_id, recorder, ms_up, ckpt_uploaded_ok)
        run_eval_single(dp_dir, neg, f"neg_{pos}__{neg}", gpu_id, recorder, ms_up, rw_root, ckpt_uploaded_ok)
        run_train_joint(dp_dir, stem, pos, neg, gpu_id, recorder, ms_up, ckpt_uploaded_ok)
        run_eval_joint(dp_dir, stem, pos, neg, gpu_id, recorder, ms_up, rw_root, ckpt_uploaded_ok)

    recorder.record("OK", "flow", "finished_all_pairs")
    print("[OK] DP __flow.py finished (see summary on exit).", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
