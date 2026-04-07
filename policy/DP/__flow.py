#!/usr/bin/env python3

"""
Eval-only flow scheduler (ev branch): parallel bash _eval.sh <stem> per EVAL_STEMS.

Runtime precedence for eval wrappers:
CLI args > FLOW env injected by this file > YAML in _ev_cfg/.

Injected env:
- DP_FLOW_GPU, DP_FLOW_SLOT, DP_FLOW_STEM, DP_FLOW_PHASE (=eval)
- DP_FLOW_SEED, DP_FLOW_TEST_NUM (optional; from FLOW_* constants below)
"""

import atexit
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, TextIO

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
from early_stop_util.flow_eval_best_symlinks import prepare_dp_flow_best_symlinks

PARALLEL = [1, 1]
FLOW_SEED = 0
FLOW_TEST_NUM = 50

EVAL_STEMS = [
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
]


@dataclass
class Job:
    slot: int
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


def start_slot_job(
    slot: int,
    stem: str,
    gpu_id: int,
    env: dict,
    queue_idx: int,
) -> Job:
    slot_env = env.copy()
    slot_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    slot_env["DP_FLOW_GPU"] = str(gpu_id)
    slot_env["DP_FLOW_SLOT"] = str(slot)
    slot_env["DP_FLOW_STEM"] = stem
    slot_env["DP_FLOW_PHASE"] = "eval"
    if FLOW_SEED is not None:
        slot_env["DP_FLOW_SEED"] = str(FLOW_SEED)
    if FLOW_TEST_NUM is not None:
        slot_env["DP_FLOW_TEST_NUM"] = str(FLOW_TEST_NUM)

    logs = ensure_logs_dir(BASE_DIR)
    ts = utc8_now_str()
    safe_stem = safe_filename_part(stem)
    tmp_path = logs / (
        f"flow_eval_{safe_stem}_slot{slot}_gpu{gpu_id}_q{queue_idx}_{ts}_tmp.log"
    )
    final_path = logs / (
        f"flow_eval_{safe_stem}_slot{slot}_gpu{gpu_id}_q{queue_idx}_{ts}_pid{{pid}}.log"
    )

    lf = open_flow_text_log(tmp_path)
    lf.write(
        f"# flow_meta kind=eval stem={stem} slot={slot} gpu={gpu_id} "
        f"queue_idx={queue_idx} ts_utc8={ts}\n"
    )
    lf.flush()

    script = f"set -euo pipefail; bash _eval.sh {stem!r}"

    print(
        f"[flow][slot={slot}][eval][stem={stem}][gpu={gpu_id}] "
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
        f"[flow][slot={slot}][eval][stem={stem}][gpu={gpu_id}] "
        f"log={done_final} pid={pid} (source=FLOW.PARALLEL)",
        flush=True,
    )
    return Job(
        slot=slot,
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
        f"[flow][eval-only] PARALLEL={PARALLEL} FLOW_SEED={FLOW_SEED} "
        f"FLOW_TEST_NUM={FLOW_TEST_NUM} EVAL_STEMS={len(EVAL_STEMS)}",
        flush=True,
    )
    try:
        prepare_dp_flow_best_symlinks(BASE_DIR, BASE_DIR.parent.parent, EVAL_STEMS, FLOW_SEED)
    except Exception as exc:
        print(f"[flow][best_symlink] failed: {exc}", file=sys.stderr, flush=True)
        return 1

    next_idx = 0
    total = len(EVAL_STEMS)
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
            stem = EVAL_STEMS[next_idx]
            gpu_id = PARALLEL[slot]
            job = start_slot_job(slot, stem, gpu_id, env, next_idx)
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
                hdr = f"[flow][slot={job.slot}][eval][stem={job.stem}]"
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
