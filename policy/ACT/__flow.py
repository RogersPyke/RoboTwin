#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flow scheduler for ACT train/eval pipeline.

Uses unified configuration from policy_util/config/tr.yaml.

Usage:
    cd policy/ACT
    bash __flow.sh

The flow will:
1. Process data for all tasks defined in unified config
2. Train models in parallel using GPU slots from gpu_parallel config
"""

import atexit
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, TextIO

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
from flow_util.unified_flow import (
    build_tr_wrapper_cmd,
    get_gpu_parallel,
    get_model_config,
    get_process_data_tasks,
    get_seed,
    get_tr_tasks,
    load_unified_config,
)

MODEL_NAME = "ACT"
UNIFIED_CFG_PATH = _POLICY_UTIL_ROOT / "config" / "tr.yaml"


@dataclass
class Job:
    slot: int
    phase: str
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


def run_process_data_steps(env: dict, task_data: List[str], gpu_tag: str) -> int:
    """Sequential process_data; one log file per task (full console capture)."""
    logs = ensure_logs_dir(BASE_DIR)
    for q_idx, task_name in enumerate(task_data):
        ts = utc8_now_str()
        safe_task = safe_filename_part(task_name)
        tmp_path = (
            logs
            / f"flow_process_data_{safe_task}_slot0_gpu{gpu_tag}_q{q_idx}_{ts}_tmp.log"
        )
        final_path = (
            logs
            / f"flow_process_data_{safe_task}_slot0_gpu{gpu_tag}_q{q_idx}_{ts}_pid{{pid}}.log"
        )
        lf = open_flow_text_log(tmp_path)
        lf.write(
            f"# flow_meta kind=process_data task_name={task_name} slot=0 gpu={gpu_tag} "
            f"queue_idx={q_idx} ts_utc8={ts}\n"
        )
        lf.flush()
        cmd = ["bash", "process_data.sh", task_name, "demo_clean", "100"]
        if shutil.which("stdbuf"):
            cmd = ["stdbuf", "-oL", "-eL"] + cmd
        print(
            f"[flow][slot=0][process_data][task={task_name}][gpu={gpu_tag}] "
            f"log={tmp_path} starting",
            flush=True,
        )
        p = subprocess.Popen(
            cmd,
            cwd=str(BASE_DIR),
            env=env,
            stdout=lf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        pid = p.pid
        lf.write(f"# child_pid={pid}\n")
        lf.flush()
        done_final = final_path.parent / final_path.name.format(pid=pid)
        rename_log_with_pid(tmp_path, done_final)
        code = p.wait()
        try:
            lf.close()
        except Exception:
            pass
        if code != 0:
            print(
                f"[flow][process_data][task={task_name}] failed code={code} log={done_final}",
                file=sys.stderr,
                flush=True,
            )
            dump_log_tail_to_stderr(
                done_final,
                f"[flow][process_data][task={task_name}]",
            )
            return code
        print(
            f"[flow][process_data][task={task_name}] done log={done_final}",
            flush=True,
        )
    return 0


def start_slot_job(
    slot: int,
    phase: str,
    task_id: str,
    gpu_id: int,
    env: dict,
    queue_idx: int,
    seed: int,
    yaml_path: Path,
) -> Job:
    slot_env = env.copy()
    slot_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    slot_env["ACT_FLOW_GPU"] = str(gpu_id)
    slot_env["ACT_FLOW_SLOT"] = str(slot)
    slot_env["ACT_FLOW_TASK_ID"] = task_id
    slot_env["ACT_FLOW_PHASE"] = phase

    logs = ensure_logs_dir(BASE_DIR)
    ts = utc8_now_str()
    safe_task_id = safe_filename_part(task_id)
    tmp_path = logs / (
        f"flow_{phase}_{safe_task_id}_slot{slot}_gpu{gpu_id}_q{queue_idx}_{ts}_tmp.log"
    )
    final_path = logs / (
        f"flow_{phase}_{safe_task_id}_slot{slot}_gpu{gpu_id}_q{queue_idx}_{ts}_pid{{pid}}.log"
    )

    lf = open_flow_text_log(tmp_path)
    lf.write(
        f"# flow_meta kind={phase} task_id={task_id} slot={slot} gpu={gpu_id} "
        f"queue_idx={queue_idx} ts_utc8={ts}\n"
    )
    lf.flush()

    cmd = build_tr_wrapper_cmd(
        task_id=task_id,
        yaml_path=yaml_path,
        gpu_id=gpu_id,
        seed=seed,
        wrapper_script="_tr_wrapper.py",
    )

    print(
        f"[flow][slot={slot}][{phase}][task_id={task_id}][gpu={gpu_id}] "
        f"log={tmp_path} starting",
        flush=True,
    )
    process = subprocess.Popen(
        cmd,
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
        f"[flow][slot={slot}][{phase}][task_id={task_id}][gpu={gpu_id}] "
        f"log={done_final} pid={pid}",
        flush=True,
    )
    return Job(
        slot=slot,
        phase=phase,
        task_id=task_id,
        queue_idx=queue_idx,
        process=process,
        log_path=done_final,
        log_file=lf,
    )


def main() -> int:
    unified_cfg = load_unified_config(UNIFIED_CFG_PATH)
    model_cfg = get_model_config(unified_cfg, MODEL_NAME)
    parallel = get_gpu_parallel(model_cfg)
    tr_tasks = get_tr_tasks(model_cfg)
    seed = get_seed(model_cfg)

    if not parallel:
        print("[flow] gpu_parallel is empty", file=sys.stderr)
        return 1

    if not tr_tasks:
        print("[flow] tr_tasks is empty", file=sys.stderr)
        return 1

    task_data = get_process_data_tasks(tr_tasks)
    gpu_tag = str(parallel[0])

    env = inject_flow_child_env(os.environ.copy())
    print(
        f"[flow] main MODEL={MODEL_NAME} PARALLEL={parallel} SEED={seed} "
        f"TASK_DATA={len(task_data)} TR_TASKS={len(tr_tasks)} "
        f"CONFIG={UNIFIED_CFG_PATH}",
        flush=True,
    )

    pd_code = run_process_data_steps(env, task_data, gpu_tag)
    if pd_code != 0:
        return pd_code

    next_idx = 0
    total = len(tr_tasks)
    slot_count = len(parallel)

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
            task_id = task.get("task_id", f"task_{next_idx}")
            gpu_id = parallel[slot]
            job = start_slot_job(
                slot, "train", task_id, gpu_id, env, next_idx, seed, UNIFIED_CFG_PATH
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
                hdr = f"[flow][slot={job.slot}][{job.phase}][task_id={job.task_id}]"
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
