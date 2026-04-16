#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Training Task Scheduler.

@input: [dict, training config from tr_cfg_parser]
@output: [int, exit code (0 for success)]
@scenario: [Execute training tasks in parallel using GPU slots]

Design:
- GPU slots defined by gpu_parallel config
- Tasks executed in order, parallel within available slots
- Passes shared YAML + model YAML paths to wrapper
- Wrapper merges YAMLs with later overriding earlier
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

BASE_DIR = Path(__file__).resolve().parent.parent.parent

_POLICY_UTIL_ROOT = BASE_DIR / "policy_util"
if str(_POLICY_UTIL_ROOT) not in sys.path:
    sys.path.insert(0, str(_POLICY_UTIL_ROOT))

from log_util.flow_log import (
    dump_log_tail_to_stderr,
    ensure_logs_dir,
    inject_flow_child_env,
    open_flow_text_log,
    rename_log_with_pid,
    safe_filename_part,
    utc8_now_str,
)

SHARED_YAML = BASE_DIR / "policy_util" / "config" / "tr.yaml"


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


def _get_model_dir(model: str) -> Path:
    return BASE_DIR / "policy" / model


def _build_train_command(
    model: str,
    task_id: str,
    shared_yaml: Path,
    gpu_id: int,
    seed: int,
) -> List[str]:
    model_dir = _get_model_dir(model)
    cmd = [
        sys.executable,
        str(model_dir / "_tr_wrapper.py"),
        "--task-id",
        task_id,
        "--yaml",
        str(shared_yaml),
        "--gpu-id",
        str(gpu_id),
        "--seed",
        str(seed),
    ]
    return cmd


def start_slot_job(
    slot: int,
    task_id: str,
    gpu_id: int,
    env: dict,
    queue_idx: int,
    cfg: Dict[str, Any],
) -> Job:
    model = cfg["_model"]
    model_dir = _get_model_dir(model)

    slot_env = env.copy()
    slot_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    logs = ensure_logs_dir(model_dir)
    ts = utc8_now_str()
    safe_task_id = safe_filename_part(task_id)
    tmp_path = (
        logs
        / f"flow_train_{safe_task_id}_slot{slot}_gpu{gpu_id}_q{queue_idx}_{ts}_tmp.log"
    )
    final_path = (
        logs
        / f"flow_train_{safe_task_id}_slot{slot}_gpu{gpu_id}_q{queue_idx}_{ts}_pid{{pid}}.log"
    )

    lf = open_flow_text_log(tmp_path)
    lf.write(
        f"# flow_meta kind=train task_id={task_id} slot={slot} gpu={gpu_id} "
        f"queue_idx={queue_idx} ts_utc8={ts}\n"
    )
    lf.flush()

    seed = cfg.get("seed", 0)
    cmd = _build_train_command(model, task_id, SHARED_YAML, gpu_id, seed)

    print(
        f"[flow][slot={slot}][train][task={task_id}][gpu={gpu_id}] "
        f"log={tmp_path} starting",
        flush=True,
    )

    process = subprocess.Popen(
        cmd,
        cwd=str(model_dir),
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
        f"[flow][slot={slot}][train][task={task_id}][gpu={gpu_id}] "
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


def run_scheduler(cfg: Dict[str, Any]) -> int:
    model = cfg["_model"]
    gpu_parallel = cfg["gpu_parallel"]
    tr_tasks = cfg["tr_tasks"]

    if not gpu_parallel:
        print(f"[flow] gpu_parallel is empty for model {model}", file=sys.stderr)
        return 1

    env = inject_flow_child_env(os.environ.copy())

    print(
        f"[flow] model={model} gpu_parallel={gpu_parallel} "
        f"tasks={len(tr_tasks)} seed={cfg.get('seed')}",
        flush=True,
    )

    next_idx = 0
    total = len(tr_tasks)
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
            task = tr_tasks[next_idx]
            task_id = task["task_id"]
            gpu_id = gpu_parallel[slot]
            job = start_slot_job(slot, task_id, gpu_id, env, next_idx, cfg)
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
                hdr = f"[flow][slot={job.slot}][train][task={job.task_id}]"
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
