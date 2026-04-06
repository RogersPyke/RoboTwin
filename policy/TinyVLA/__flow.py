#!/usr/bin/env python3

"""
Flow scheduler for TinyVLA train/eval pipeline.

Design notes (minimal override contract):
1) Keep YAML as baseline defaults for each stem config.
2) Allow flow-level overrides for practical batch scheduling.
3) Keep optional CLI as highest-priority emergency override.

Runtime precedence:
CLI args > FLOW env injected by this file > YAML config defaults.

This file only injects flow-level env values and process orchestration:
- TVLA_FLOW_GPU      from PARALLEL slot gpu id
- TVLA_FLOW_SEED     from FLOW_SEED (optional)
- TVLA_FLOW_TEST_NUM from FLOW_TEST_NUM (optional)
- TVLA_FLOW_EVAL_STEPS_FOR_EARLY_STOP, TVLA_FLOW_EARLY_STOP_PATIENCE_EVALS,
  TVLA_FLOW_EARLY_STOP_REL_TOL from matching FLOW constants (optional, None skips)

process_data: python3 process_data.py <task> <task_config> <expert_num> (cwd=TinyVLA).
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

_POLICY_ROOT = Path(__file__).resolve().parent.parent
if str(_POLICY_ROOT) not in sys.path:
    sys.path.insert(0, str(_POLICY_ROOT))
from flow_log_util.flow_log import (
    bash_lc_cmd,
    dump_log_tail_to_stderr,
    ensure_logs_dir,
    inject_flow_child_env,
    open_flow_text_log,
    rename_log_with_pid,
    safe_filename_part,
    utc8_now_str,
)


CATEGORY = "train"

# Optional flow-level overrides. Keep None to use YAML defaults.
TASK_CONFIG = "demo_clean"
EXPERT_NUM = "100"
PARALLEL = [2, 3]
FLOW_SEED = 0
FLOW_TEST_NUM = 50
EVAL_STEPS_FOR_EARLY_STOP = 1000
EARLY_STOP_PATIENCE_EVALS = 30
EARLY_STOP_REL_TOL = 1e-2

TASK_DATA = [
    "move_pillbottle_pad",
    "unmove_pillbottle_pad",
    "stack_bowls_three",
    "unstack_bowls_three",
    "stack_blocks_three",
    "unstack_blocks_three",
    "hanging_mug",
    "unhanging_mug",
]

TASK_SEQ = [(CATEGORY, stem) for stem in [
    "flow_single_move_pillbottle_pad",
    "flow_single_unmove_pillbottle_pad",
    "flow_joint_move_pillbottle_pad",
    "flow_single_stack_bowls_three",
    "flow_single_unstack_bowls_three",
    "flow_joint_stack_bowls_three",
    "flow_single_stack_blocks_three",
    "flow_single_unstack_blocks_three",
    "flow_joint_stack_blocks_three",
    "flow_single_hanging_mug",
    "flow_single_unhanging_mug",
    "flow_joint_hanging_mug",
]]


@dataclass
class Job:
    slot: int
    phase: str
    stem: str
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


def run_process_data_steps(env: dict) -> int:
    """Sequential process_data.py; one log file per task (full console capture)."""
    gpu_tag = str(PARALLEL[0]) if PARALLEL else "none"
    py = sys.executable
    logs = ensure_logs_dir(BASE_DIR)
    for q_idx, task_name in enumerate(TASK_DATA):
        ts = utc8_now_str()
        safe_task = safe_filename_part(task_name)
        tmp_path = logs / f"flow_process_data_{safe_task}_slot0_gpu{gpu_tag}_q{q_idx}_{ts}_tmp.log"
        final_path = logs / f"flow_process_data_{safe_task}_slot0_gpu{gpu_tag}_q{q_idx}_{ts}_pid{{pid}}.log"
        lf = open_flow_text_log(tmp_path)
        lf.write(
            f"# flow_meta kind=process_data task_name={task_name} slot=0 gpu={gpu_tag} "
            f"queue_idx={q_idx} ts_utc8={ts}\n"
        )
        lf.flush()
        cmd = [py, "process_data.py", task_name, TASK_CONFIG, EXPERT_NUM]
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
    stem: str,
    gpu_id: int,
    env: dict,
    queue_idx: int,
) -> Job:
    slot_env = env.copy()
    slot_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    slot_env["TVLA_FLOW_GPU"] = str(gpu_id)
    slot_env["TVLA_FLOW_SLOT"] = str(slot)
    slot_env["TVLA_FLOW_STEM"] = stem
    slot_env["TVLA_FLOW_PHASE"] = phase
    if FLOW_SEED is not None:
        slot_env["TVLA_FLOW_SEED"] = str(FLOW_SEED)
    if FLOW_TEST_NUM is not None:
        slot_env["TVLA_FLOW_TEST_NUM"] = str(FLOW_TEST_NUM)
    if EVAL_STEPS_FOR_EARLY_STOP is not None:
        slot_env["TVLA_FLOW_EVAL_STEPS_FOR_EARLY_STOP"] = str(EVAL_STEPS_FOR_EARLY_STOP)
    if EARLY_STOP_PATIENCE_EVALS is not None:
        slot_env["TVLA_FLOW_EARLY_STOP_PATIENCE_EVALS"] = str(EARLY_STOP_PATIENCE_EVALS)
    if EARLY_STOP_REL_TOL is not None:
        slot_env["TVLA_FLOW_EARLY_STOP_REL_TOL"] = str(EARLY_STOP_REL_TOL)

    logs = ensure_logs_dir(BASE_DIR)
    ts = utc8_now_str()
    safe_stem = safe_filename_part(stem)
    tmp_path = logs / (
        f"flow_{phase}_{safe_stem}_slot{slot}_gpu{gpu_id}_q{queue_idx}_{ts}_tmp.log"
    )
    final_path = logs / (
        f"flow_{phase}_{safe_stem}_slot{slot}_gpu{gpu_id}_q{queue_idx}_{ts}_pid{{pid}}.log"
    )

    lf = open_flow_text_log(tmp_path)
    lf.write(
        f"# flow_meta kind={phase} stem={stem} slot={slot} gpu={gpu_id} "
        f"queue_idx={queue_idx} ts_utc8={ts}\n"
    )
    lf.flush()

    if phase == "train":
        script = f"set -euo pipefail; bash _train.sh {stem!r}"
    else:
        script = f"set -euo pipefail; bash _eval.sh {stem!r}"

    print(
        f"[flow][slot={slot}][{phase}][stem={stem}][gpu={gpu_id}] "
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
        f"[flow][slot={slot}][{phase}][stem={stem}][gpu={gpu_id}] "
        f"log={done_final} pid={pid} (source=FLOW.PARALLEL)",
        flush=True,
    )
    return Job(
        slot=slot,
        phase=phase,
        stem=stem,
        queue_idx=queue_idx,
        process=process,
        log_path=done_final,
        log_file=lf,
    )


def main() -> int:
    if not PARALLEL:
        print("[flow] PARALLEL is empty", file=sys.stderr)
        return 1

    env = inject_flow_child_env(os.environ.copy())
    print(
        f"[flow] main TASK_CONFIG={TASK_CONFIG} EXPERT_NUM={EXPERT_NUM} "
        f"PARALLEL={PARALLEL} FLOW_SEED={FLOW_SEED} FLOW_TEST_NUM={FLOW_TEST_NUM} "
        f"TASK_DATA={len(TASK_DATA)} TASK_SEQ={len(TASK_SEQ)} "
        f"(source=FLOW constants)",
        flush=True,
    )

    pd_code = run_process_data_steps(env)
    if pd_code != 0:
        return pd_code

    next_idx = 0
    total = len(TASK_SEQ)
    slot_count = len(PARALLEL)

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
            phase, stem = TASK_SEQ[next_idx]
            gpu_id = PARALLEL[slot]
            job = start_slot_job(slot, phase, stem, gpu_id, env, next_idx)
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
                hdr = (
                    f"[flow][slot={job.slot}][{job.phase}][stem={job.stem}]"
                )
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
