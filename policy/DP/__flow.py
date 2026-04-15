#!/usr/bin/env python3

"""
Eval-only flow scheduler for DP model.

Design notes (minimal override contract):
1) Keep YAML as baseline defaults for each stem config.
2) Allow flow-level overrides for practical batch scheduling.
3) Keep optional CLI as highest-priority emergency override.

Runtime precedence:
CLI args > FLOW env injected by this file > YAML config defaults.

This file only injects flow-level env values and process orchestration:
- DP_FLOW_GPU      from PARALLEL slot gpu id
- DP_FLOW_SEED     from FLOW_SEED (optional)
- DP_FLOW_TEST_NUM from FLOW_TEST_NUM (optional)
- DP_FLOW_CHECKPOINT_NUM from FLOW_CHECKPOINT_NUM (optional)
- DP_FLOW_EVAL_HEAD_CAMERA_TYPE from FLOW_EVAL_HEAD_CAMERA_TYPE (optional)
- DP_FLOW_END_RESET_TO_INIT from FLOW_END_RESET_TO_INIT (optional)
"""

import atexit
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO

import yaml

BASE_DIR = Path(__file__).resolve().parent

_POLICY_UTIL_ROOT = Path(__file__).resolve().parent.parent.parent / "policy_util"
if str(_POLICY_UTIL_ROOT) not in sys.path:
    sys.path.insert(0, str(_POLICY_UTIL_ROOT))
from log_util.flow_log import (
    bash_lc_cmd,
    dump_log_tail_to_stderr,
    ensure_logs_dir,
    inject_flow_child_env,
    open_flow_text_log,
    rename_log_with_pid,
    safe_filename_part,
    utc8_now_str,
)


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML config: {path}")
    return cfg


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _merge_yamls(yaml_paths: List[str]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for path in yaml_paths:
        if os.path.isfile(path):
            cfg = _load_yaml(path)
            merged = _deep_merge(merged, cfg)
    return merged


def _get_ev_task_ids(cfg: Dict[str, Any]) -> List[str]:
    ev_tasks = cfg.get("ev_tasks", [])
    return [task.get("task_id") for task in ev_tasks if task.get("task_id")]


def _resolve_checkpoint_dir(task: Dict[str, Any]) -> str:
    if "ckpt_dir" in task and task["ckpt_dir"]:
        return task["ckpt_dir"]
    train_tasks = task.get("train_tasks", [])
    if not train_tasks:
        raise ValueError("Either ckpt_dir or train_tasks is required")
    names = [str(row[0]).strip() for row in train_tasks]
    cfgs = [str(row[1]).strip() for row in train_tasks]
    nums = [int(row[2]) for row in train_tasks]
    train_task_slug = "__".join(names)
    train_config_slug = "__".join(cfgs)
    train_total = int(sum(nums))
    return f"policy/DP/checkpoints/{train_task_slug}-{train_config_slug}-{train_total}"


def _prepare_dp_best_symlink(ckpt_dir: str, checkpoint_num: int) -> None:
    ckpt_path = Path(ckpt_dir)
    if not ckpt_path.is_dir():
        return
    best = ckpt_path / "best_val.ckpt"
    link = ckpt_path / f"{checkpoint_num}.ckpt"
    if not best.is_file():
        return
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to("best_val.ckpt")
    print(f"[flow][best_symlink][DP] {link} -> best_val.ckpt", flush=True)


# Optional flow-level overrides. Keep None to use YAML defaults.
PARALLEL: Optional[List[int]] = None
FLOW_SEED: Optional[int] = None
FLOW_TEST_NUM: Optional[int] = None
FLOW_CHECKPOINT_NUM: Optional[int] = None
FLOW_EVAL_HEAD_CAMERA_TYPE: Optional[str] = None
FLOW_END_RESET_TO_INIT: Optional[bool] = None

# Config paths (relative to repo root).
SHARED_CFG_PATH = "policy/config/_ev_cfg_shared.yaml"
MODEL_CFG_PATH = "policy/DP/_ev_cfg/ev_tasks.yaml"


@dataclass
class Job:
    slot: int
    task_id: str
    queue_idx: int
    process: subprocess.Popen
    log_path: Path
    log_file: TextIO = field(repr=False)


ACTIVE_JOBS: Dict[int, Job] = {}


def kill_process_group(pid: int) -> None:
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.time() + 5.0
    while time.time() < deadline:
        try:
            os.killpg(pgid, 0)
            time.sleep(0.1)
        except ProcessLookupError:
            return
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return


def cleanup_all_jobs() -> None:
    for pid, job in list(ACTIVE_JOBS.items()):
        if job.process.poll() is None:
            kill_process_group(pid)
        try:
            job.log_file.close()
        except Exception:
            pass
    ACTIVE_JOBS.clear()


def on_signal(signum: int, _frame) -> None:
    cleanup_all_jobs()
    raise SystemExit(128 + signum)


def start_slot_job(
    slot: int,
    task_id: str,
    gpu_id: int,
    env: dict,
    queue_idx: int,
    cfg: Dict[str, Any],
) -> Job:
    slot_env = env.copy()
    slot_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    slot_env["DP_FLOW_GPU"] = str(gpu_id)
    slot_env["DP_FLOW_SLOT"] = str(slot)
    slot_env["DP_FLOW_STEM"] = task_id
    slot_env["DP_FLOW_PHASE"] = "eval"

    model_defaults = cfg.get("model_defaults", {})
    task_cfg = None
    for t in cfg.get("ev_tasks", []):
        if t.get("task_id") == task_id:
            task_cfg = t
            break

    seed = (
        task_cfg.get("seed", model_defaults.get("seed", 0))
        if task_cfg
        else model_defaults.get("seed", 0)
    )
    test_num = (
        task_cfg.get("test_num", model_defaults.get("test_num", 50))
        if task_cfg
        else model_defaults.get("test_num", 50)
    )
    end_reset = (
        task_cfg.get("end_reset_to_init", model_defaults.get("end_reset_to_init", True))
        if task_cfg
        else model_defaults.get("end_reset_to_init", True)
    )
    checkpoint_num = (
        task_cfg.get("checkpoint_num", model_defaults.get("checkpoint_num", 600))
        if task_cfg
        else model_defaults.get("checkpoint_num", 600)
    )
    head_camera_type = (
        task_cfg.get(
            "eval_head_camera_type", model_defaults.get("eval_head_camera_type", "D435")
        )
        if task_cfg
        else model_defaults.get("eval_head_camera_type", "D435")
    )

    if FLOW_SEED is not None:
        slot_env["DP_FLOW_SEED"] = str(FLOW_SEED)
    else:
        slot_env["DP_FLOW_SEED"] = str(seed)
    if FLOW_TEST_NUM is not None:
        slot_env["DP_FLOW_TEST_NUM"] = str(FLOW_TEST_NUM)
    else:
        slot_env["DP_FLOW_TEST_NUM"] = str(test_num)
    if FLOW_CHECKPOINT_NUM is not None:
        slot_env["DP_FLOW_CHECKPOINT_NUM"] = str(FLOW_CHECKPOINT_NUM)
    else:
        slot_env["DP_FLOW_CHECKPOINT_NUM"] = str(checkpoint_num)
    if FLOW_EVAL_HEAD_CAMERA_TYPE is not None:
        slot_env["DP_FLOW_EVAL_HEAD_CAMERA_TYPE"] = str(FLOW_EVAL_HEAD_CAMERA_TYPE)
    else:
        slot_env["DP_FLOW_EVAL_HEAD_CAMERA_TYPE"] = str(head_camera_type)
    if FLOW_END_RESET_TO_INIT is not None:
        slot_env["DP_FLOW_END_RESET_TO_INIT"] = (
            "true" if FLOW_END_RESET_TO_INIT else "false"
        )
    else:
        slot_env["DP_FLOW_END_RESET_TO_INIT"] = "true" if end_reset else "false"

    if task_cfg:
        ckpt_dir = _resolve_checkpoint_dir(task_cfg)
        _prepare_dp_best_symlink(ckpt_dir, checkpoint_num)

    logs = ensure_logs_dir(BASE_DIR)
    ts = utc8_now_str()
    safe_stem = safe_filename_part(task_id)
    tmp_path = logs / (
        f"flow_eval_{safe_stem}_slot{slot}_gpu{gpu_id}_q{queue_idx}_{ts}_tmp.log"
    )
    final_path = logs / (
        f"flow_eval_{safe_stem}_slot{slot}_gpu{gpu_id}_q{queue_idx}_{ts}_pid{{pid}}.log"
    )

    lf = open_flow_text_log(tmp_path)
    lf.write(
        f"# flow_meta kind=eval task_id={task_id} slot={slot} gpu={gpu_id} "
        f"queue_idx={queue_idx} ts_utc8={ts}\n"
    )
    lf.flush()

    script = f"set -euo pipefail; bash _eval.sh {task_id!r}"

    print(
        f"[flow][slot={slot}][eval][task_id={task_id}][gpu={gpu_id}] "
        f"log={tmp_path} starting",
        flush=True,
    )
    process = subprocess.Popen(
        bash_lc_cmd(script),
        cwd=str(BASE_DIR),
        env=slot_env,
        stdout=lf,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    pid = process.pid
    lf.write(f"# child_pid={pid}\n")
    lf.flush()
    done_final = final_path.parent / final_path.name.format(pid=pid)
    rename_log_with_pid(tmp_path, done_final)

    print(
        f"[flow][slot={slot}][eval][task_id={task_id}][gpu={gpu_id}] "
        f"log={done_final} pid={pid}",
        flush=True,
    )
    return Job(
        slot=slot,
        task_id=task_id,
        queue_idx=queue_idx,
        process=process,
        log_path=done_final,
        log_file=lf,
    )


def main() -> int:
    repo_root = BASE_DIR.parent.parent
    shared_cfg = repo_root / SHARED_CFG_PATH
    model_cfg = repo_root / MODEL_CFG_PATH

    cfg = _merge_yamls([str(shared_cfg), str(model_cfg)])

    gpu_parallel = PARALLEL if PARALLEL is not None else cfg.get("gpu_parallel", [0])
    if not gpu_parallel:
        print("[flow] gpu_parallel is empty", file=sys.stderr)
        return 1

    task_ids = _get_ev_task_ids(cfg)
    if not task_ids:
        print("[flow] No ev_tasks found in config", file=sys.stderr)
        return 1

    os.environ.update(inject_flow_child_env(os.environ.copy()))
    print(
        f"[flow][eval-only] gpu_parallel={gpu_parallel} "
        f"FLOW_SEED={FLOW_SEED} FLOW_TEST_NUM={FLOW_TEST_NUM} "
        f"FLOW_CHECKPOINT_NUM={FLOW_CHECKPOINT_NUM} "
        f"FLOW_EVAL_HEAD_CAMERA_TYPE={FLOW_EVAL_HEAD_CAMERA_TYPE} "
        f"FLOW_END_RESET_TO_INIT={FLOW_END_RESET_TO_INIT} "
        f"tasks={len(task_ids)}",
        flush=True,
    )

    next_idx = 0
    total = len(task_ids)
    slot_count = len(gpu_parallel)

    def try_fill_slots() -> None:
        nonlocal next_idx
        while next_idx < total and len(ACTIVE_JOBS) < slot_count:
            slot = None
            for s in range(slot_count):
                used = any(j.slot == s for j in ACTIVE_JOBS.values())
                if not used:
                    slot = s
                    break
            if slot is None:
                break
            task_id = task_ids[next_idx]
            gpu_id = gpu_parallel[slot]
            job = start_slot_job(
                slot, task_id, gpu_id, os.environ.copy(), next_idx, cfg
            )
            ACTIVE_JOBS[job.process.pid] = job
            next_idx += 1

    try_fill_slots()

    while ACTIVE_JOBS:
        failed = False
        for pid, job in list(ACTIVE_JOBS.items()):
            code = job.process.poll()
            if code is None:
                continue
            del ACTIVE_JOBS[pid]
            try:
                job.log_file.close()
            except Exception:
                pass
            if code != 0:
                failed = True
                hdr = f"[flow][slot={job.slot}][eval][task_id={job.task_id}]"
                print(
                    f"{hdr} failed code={code} log={job.log_path}",
                    file=sys.stderr,
                    flush=True,
                )
                dump_log_tail_to_stderr(job.log_path, hdr)
                break
            try_fill_slots()
        if failed:
            cleanup_all_jobs()
            print("[flow] job failed, aborted", file=sys.stderr)
            return 1
        time.sleep(0.15)

    return 0


if __name__ == "__main__":
    atexit.register(cleanup_all_jobs)
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, on_signal)
    os.chdir(BASE_DIR)
    raise SystemExit(main())
