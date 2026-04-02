#!/usr/bin/env python3

"""
Flow scheduler for DP train/eval pipeline.

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
"""

import atexit
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

# Optional flow-level overrides. Keep None to use YAML defaults.
TASK_CONFIG = "demo_clean"
EXPERT_NUM = "100"
PARALLEL = [1, 1, 1]
FLOW_SEED = None
FLOW_TEST_NUM = None

TASKS = [
    "move_pillbottle_pad",
    "unmove_pillbottle_pad",
    "stack_bowls_three",
    "unstack_bowls_three",
    "stack_blocks_three",
    "unstack_blocks_three",
    "hanging_mug",
    "unhanging_mug",
]

STEMS = [
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
    process: subprocess.Popen


ACTIVE_JOBS: Dict[int, Job] = {}
BASE_DIR = Path(__file__).resolve().parent


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
    ACTIVE_JOBS.clear()


def on_signal(signum: int, _frame) -> None:
    cleanup_all_jobs()
    raise SystemExit(128 + signum)


def run_foreground(cmd: List[str], env: dict) -> None:
    subprocess.run(cmd, cwd=BASE_DIR, env=env, check=True)


def start_slot_job(slot: int, stem: str, gpu_id: int, env: dict) -> Job:
    slot_env = env.copy()
    # Inject flow context for wrappers; wrappers resolve with:
    # CLI > FLOW env > YAML.
    slot_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    slot_env["DP_FLOW_GPU"] = str(gpu_id)
    slot_env["DP_FLOW_SLOT"] = str(slot)
    slot_env["DP_FLOW_STEM"] = stem
    if FLOW_SEED is not None:
        slot_env["DP_FLOW_SEED"] = str(FLOW_SEED)
    if FLOW_TEST_NUM is not None:
        slot_env["DP_FLOW_TEST_NUM"] = str(FLOW_TEST_NUM)
    print(
        f"[flow] child slot={slot} stem={stem} gpu={gpu_id} "
        f"seed={FLOW_SEED} test_num={FLOW_TEST_NUM} "
        f"(source=FLOW.PARALLEL)",
        flush=True,
    )
    script = (
        f"set -euo pipefail; "
        f"bash _train.sh {stem!r}; "
        f"bash _eval.sh {stem!r}"
    )
    process = subprocess.Popen(
        ["bash", "-lc", script],
        cwd=BASE_DIR,
        env=slot_env,
        start_new_session=True,
    )
    return Job(slot=slot, stem=stem, process=process)


def main() -> int:
    if not PARALLEL:
        print("[flow] PARALLEL is empty", file=sys.stderr)
        return 1

    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    print(
        f"[flow] main TASK_CONFIG={TASK_CONFIG} EXPERT_NUM={EXPERT_NUM} "
        f"PARALLEL={PARALLEL} FLOW_SEED={FLOW_SEED} FLOW_TEST_NUM={FLOW_TEST_NUM} "
        f"TASKS={len(TASKS)} STEMS={len(STEMS)} "
        f"(source=FLOW constants)",
        flush=True,
    )

    for task in TASKS:
        run_foreground(["bash", "process_data.sh", task, TASK_CONFIG, EXPERT_NUM], env)

    next_idx = 0
    total = len(STEMS)
    slot_count = len(PARALLEL)

    for slot in range(slot_count):
        if next_idx >= total:
            break
        stem = STEMS[next_idx]
        job = start_slot_job(slot, stem, PARALLEL[slot], env)
        ACTIVE_JOBS[job.process.pid] = job
        next_idx += 1

    while ACTIVE_JOBS:
        failed = False
        for pid, job in list(ACTIVE_JOBS.items()):
            code = job.process.poll()
            if code is None:
                continue
            del ACTIVE_JOBS[pid]
            if code != 0:
                failed = True
                break
            if next_idx < total:
                stem = STEMS[next_idx]
                new_job = start_slot_job(job.slot, stem, PARALLEL[job.slot], env)
                ACTIVE_JOBS[new_job.process.pid] = new_job
                next_idx += 1
        if failed:
            cleanup_all_jobs()
            print("[flow] job failed, aborted", file=sys.stderr)
            return 1
        time.sleep(0.5)

    return 0


if __name__ == "__main__":
    atexit.register(cleanup_all_jobs)
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    os.chdir(BASE_DIR)
    raise SystemExit(main())
