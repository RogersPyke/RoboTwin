#!/usr/bin/env python3
"""
Batch data collection launcher: Cartesian product of task list x config list,
run via collect_data.sh in parallel. Config is declared at the beginning of this file.

Dependencies: Python 3.6+, standard library only (subprocess, logging, multiprocessing).
Usage: invoked by collect_data_flow.sh with no CLI arguments.

Call-chain note (from this script as caller):
  This script only invokes: collect_data.sh -> collect_data.py -> task env (load_robot/set_planner).
  (1) "Robot has no attribute left_planner": Downstream Robot.set_planner() sets left_planner only
      when left and right curobo config paths are equal; otherwise it uses subprocess+pipe and does
      not set left_planner. Later update_world_pcd() always accesses left_planner, so that path
      triggers AttributeError. Caller cannot fix this without changing robot.py.
  (2) Process groups and shutdown: Parent must not exit until all children are terminated;
      otherwise subprocesses become orphans. This script uses process groups: main process is
      the group leader; each Pool worker becomes its own process group leader. Subprocesses
      (Popen) are started without start_new_session so they stay in the worker's group. On
      SIGINT/SIGTERM the main process first terminates the pool (workers get SIGTERM); each
      worker's SIGTERM handler kills its entire process group (worker + subprocesses), then
      main joins the pool and exits.

Features:
  - Per-process logging with immediate flush
  - Each worker gets its own log file
  - Parent process has separate log file
  - Safe process termination: SIGTERM first, wait 5s, then SIGKILL
  - GPU memory check before starting each task
  - Console: only process status (START/OK/FAIL), not subprocess output
  - Log file: all subprocess output captured
  - Stall timeout: every COLLECT_DATA_JOB_TIMEOUT_SEC (default 1800s) compare SHA256 of the
    worker log file with the previous slice; SIGTERM only if two consecutive hashes match
    (log unchanged). Override: COLLECT_DATA_JOB_TIMEOUT_SEC
"""

import hashlib
import logging
import multiprocessing as mp
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from log_utils import (
    setup_parent_process_logging,
    ImmediateFlushFileHandler,
    ColoredFormatter,
    flush_log,
    LOG_DIR,
    UTC8,
    RED,
    GREEN,
    BLUE,
    YELLOW,
    RESET,
    SUCCESS_LEVEL,
)

# ---------------------------------------------------------------------------
# Flow config (declare/edit here)
# ---------------------------------------------------------------------------
TASK_TO_COLL = [
    "hanging_mug_pert",
    "unhanging_mug_pert",
    "stack_blocks_three_pert",
    "unstack_blocks_three_pert",
    "stack_bowls_three_pert",
    "unstack_bowls_three_pert",
    "move_pillbottle_pad_pert",
    "unmove_pillbottle_pad_pert",
]
CFG_TO_COLL = ["demo_clean_pert"]
GPU_PARALLEL = [0, 1]
END_RESET_TO_INIT = True

# Process management constants
SIGTERM_TIMEOUT_SEC = 5.0  # Wait time before SIGKILL
MIN_GPU_FREE_MEMORY_MB = 1024  # Minimum free GPU memory required (MB)
# Interval between stall checks (same as each proc.wait timeout slice). Default 30 minutes.
# Termination: only if the worker log file SHA256 matches the stack top twice in a row.
# Override: export COLLECT_DATA_JOB_TIMEOUT_SEC=3600
COLLECT_DATA_JOB_TIMEOUT_SEC = int(
    os.environ.get("COLLECT_DATA_JOB_TIMEOUT_SEC", str(30 * 60))
)


def _join_pool_workers_with_timeout(pool, timeout_sec: float) -> None:
    """
    Wait for Pool worker processes to exit, up to timeout_sec.
    @input: pool (multiprocessing.pool.Pool), timeout_sec (float, seconds > 0)
    @output: None; returns when all workers are dead or timeout elapses
    @scenario: stdlib Pool.join() has no timeout; poll Process.is_alive on pool._pool.
    """
    workers = getattr(pool, "_pool", None)
    if not workers:
        pool.join()
        return
    deadline = time.monotonic() + float(timeout_sec)
    while time.monotonic() < deadline:
        if not any(p.is_alive() for p in workers):
            return
        time.sleep(0.05)


def _worker_ignore_sigint():
    """Ignore SIGINT in pool workers so only the main process handles Ctrl+C."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _worker_process_group_and_sigterm():
    """
    Make this worker the leader of its own process group and install SIGTERM handler
    that kills the whole group (worker + any Popen children). Ensures no orphan subprocesses
    when the main process requests shutdown. Strategy: SIGTERM first, wait, then SIGKILL.
    """
    try:
        os.setpgid(0, 0)
    except OSError:
        pass

    _pgid = os.getpgrp()

    def _kill_process_group(_signum, _frame):
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        try:
            os.killpg(_pgid, signal.SIGTERM)
        except OSError:
            pass
        time.sleep(SIGTERM_TIMEOUT_SEC)
        try:
            os.killpg(_pgid, signal.SIGKILL)
        except OSError:
            pass

    signal.signal(signal.SIGTERM, _kill_process_group)


def _worker_init():
    """
    Pool worker initializer: ignore SIGINT, create own process group, install SIGTERM
    handler to kill entire group (worker + Popen children) so no orphans on shutdown.
    """
    _worker_ignore_sigint()
    _worker_process_group_and_sigterm()


def _main_install_shutdown_handler(
    shutdown_event: threading.Event, log: logging.Logger
):
    """
    Install SIGTERM handler so that on external kill (e.g. systemd, kill <pid>), the main
    process sets shutdown_event and then the main loop will terminate the pool and exit.
    Does not run cleanup inside the handler (not signal-safe); only sets the flag.
    """

    def _handler(_signum, _frame):
        shutdown_event.set()

    signal.signal(signal.SIGTERM, _handler)


def load_config():
    """Load TASK_TO_COLL, CFG_TO_COLL, GPU_PARALLEL from file-level config."""
    return list(TASK_TO_COLL), list(CFG_TO_COLL), list(GPU_PARALLEL)


def check_gpu_memory(gpu_id: int, log: logging.Logger) -> Tuple[bool, int, int]:
    """
    Check if GPU has enough free memory.
    @input: gpu_id (int), log (logging.Logger)
    @output: (has_enough_memory, free_mb, total_mb)
    @scenario: Called before starting a task to ensure GPU is not OOM
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free,memory.total",
                "--format=csv,noheader,nounits",
                f"--id={gpu_id}",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            log.warning("[GPU_CHECK] nvidia-smi failed: %s", result.stderr.strip())
            return True, 0, 0

        parts = result.stdout.strip().split(",")
        if len(parts) != 2:
            log.warning(
                "[GPU_CHECK] Unexpected nvidia-smi output: %s", result.stdout.strip()
            )
            return True, 0, 0

        free_mb = int(parts[0].strip())
        total_mb = int(parts[1].strip())
        has_enough = free_mb >= MIN_GPU_FREE_MEMORY_MB

        return has_enough, free_mb, total_mb
    except Exception as e:
        log.warning("[GPU_CHECK] Exception: %s", e)
        return True, 0, 0


# Repo root (parent of script/); collect_data.sh and logs live here.
SCRIPT_DIR = Path(__file__).resolve().parent.parent


def _timestamp_utc8():
    """Return current timestamp string YYYYMMDDHHMMSS in UTC+8."""
    return datetime.now(UTC8).strftime("%Y%m%d%H%M%S")


def _run_tag(task: str, cfg: str, use_color: bool = True) -> str:
    """Return a consistent prefix for log lines: [task][cfg]. use_color adds ANSI for console."""
    if use_color:
        return f"{BLUE}[{task}][{cfg}]{RESET}"
    return f"[{task}][{cfg}]"


def _worker_log_file_sha256(log_file_path: Path) -> str:
    """
    Full-file SHA256 hex digest of the worker log file (subprocess output sink).

    @input: log_file_path (Path), UTF-8 text log; may not exist yet
    @output: str, 64-char hex digest; empty/missing file hashes empty content
    @scenario: Stable fingerprint for stall detection between timeout slices
    """
    if not log_file_path.exists():
        return hashlib.sha256(b"").hexdigest()
    h = hashlib.sha256()
    with open(log_file_path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _read_stdout_to_file(proc: subprocess.Popen, log_file_path: Path) -> None:
    """
    Read subprocess stdout and write to log file line by line.
    @input: proc (subprocess.Popen with stdout=PIPE), log_file_path (Path)
    @output: None
    @scenario: Captures all subprocess output to file without printing to console
    """
    with open(log_file_path, "a", encoding="utf-8") as f:
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            f.write(line)
            f.flush()


def run_collect_data_sh(
    task: str,
    cfg: str,
    gpu_id: int,
    script_dir: Path,
    worker_log: logging.Logger,
    worker_log_file: Path,
) -> bool:
    """
    Run collect_data.sh for one (task, config) pair on the given GPU.
    @input: task/cfg (str), gpu_id (int), script_dir (Path), worker_log (Logger), worker_log_file (Path)
    @output: True on success, False on failure
    @scenario: Console shows only START/OK/FAIL; log file captures all subprocess output
    """
    tag = _run_tag(task, cfg)

    # Check GPU memory before starting
    has_memory, free_mb, total_mb = check_gpu_memory(gpu_id, worker_log)
    worker_log.info(
        "%s GPU%s memory: %d MB free / %d MB total", tag, gpu_id, free_mb, total_mb
    )
    if not has_memory:
        worker_log.error(
            "%s GPU%s OOM risk: only %d MB free (need %d MB)",
            tag,
            gpu_id,
            free_mb,
            MIN_GPU_FREE_MEMORY_MB,
        )
        flush_log(worker_log)
        return False
    flush_log(worker_log)

    cmd = ["bash", str(script_dir / "collect_data.sh"), task, cfg, str(gpu_id)]
    worker_log.info(
        "%s START GPU%s (stall_check_interval_sec=%d; terminate only if worker log "
        "SHA256 unchanged for two consecutive intervals; COLLECT_DATA_JOB_TIMEOUT_SEC)",
        tag,
        gpu_id,
        COLLECT_DATA_JOB_TIMEOUT_SEC,
    )
    flush_log(worker_log)

    try:
        # Always capture subprocess output to file, not console
        proc = subprocess.Popen(
            cmd,
            cwd=str(script_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Read stdout in background thread, write to worker's log file
        reader = threading.Thread(
            target=_read_stdout_to_file,
            args=(proc, worker_log_file),
            daemon=True,
        )
        reader.start()

        # Stack of SHA256 digests of worker_log_file after each stall check slice.
        stall_log_hash_stack: List[str] = []

        while True:
            try:
                proc.wait(timeout=COLLECT_DATA_JOB_TIMEOUT_SEC)
                break
            except subprocess.TimeoutExpired:
                digest = _worker_log_file_sha256(worker_log_file)
                if stall_log_hash_stack and digest == stall_log_hash_stack[-1]:
                    worker_log.error(
                        "%s STALL_TIMEOUT: worker log SHA256 unchanged for two consecutive "
                        "%ds intervals; digest=%s; SIGTERM child",
                        tag,
                        COLLECT_DATA_JOB_TIMEOUT_SEC,
                        digest,
                    )
                    flush_log(worker_log)
                    proc.terminate()
                    proc.wait()
                    raise subprocess.TimeoutExpired(
                        cmd, COLLECT_DATA_JOB_TIMEOUT_SEC
                    ) from None
                stall_log_hash_stack.append(digest)
                worker_log.info(
                    "%s stall check: log still changing or first interval "
                    "(stack_depth=%d digest_prefix=%s)",
                    tag,
                    len(stall_log_hash_stack),
                    digest[:16],
                )
                flush_log(worker_log)

        reader.join(timeout=5.0)

        if proc.returncode != 0:
            worker_log.error("%s FAILED (exit code %d)", tag, proc.returncode)
            flush_log(worker_log)
            return False

        worker_log.success("%s OK", tag)
        flush_log(worker_log)
        return True

    except subprocess.TimeoutExpired:
        worker_log.error(
            "%s TIMEOUT (stall: two identical worker-log SHA256 checks at %ds); "
            "child was SIGTERM'd — data collection may be partial",
            tag,
            COLLECT_DATA_JOB_TIMEOUT_SEC,
        )
        flush_log(worker_log)
        return False
    except Exception as e:
        worker_log.exception("%s ERR: %s", tag, e)
        flush_log(worker_log)
        return False


def worker(jobs: list, gpu_id: int, script_dir: Path, worker_id: int) -> int:
    """
    Run a list of (task, cfg) jobs on one GPU. Returns number of failures.
    @input: jobs list of (task, cfg), gpu_id (int), script_dir (Path), worker_id (int)
    @output: int, count of failed runs
    @scenario: Console shows only process status; log file captures all output
    """
    _worker_init()

    # Setup per-worker logging
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"collect_data_flow_worker{worker_id}_{_timestamp_utc8()}.log"

    worker_log = logging.getLogger(f"worker_{worker_id}")
    worker_log.setLevel(logging.DEBUG)
    worker_log.handlers = []

    fmt = "%(asctime)s [%(name)s] %(levelname)s %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    # File handler: all DEBUG+ messages
    fh = ImmediateFlushFileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    worker_log.addHandler(fh)

    # Console handler: only worker-level status (INFO+), no subprocess output
    # Use a custom handler that only shows worker status messages
    class WorkerStatusHandler(logging.StreamHandler):
        """Console handler that only shows START/OK/FAIL/TIMEOUT/ERR messages."""

        STATUS_KEYWORDS = ("START", "OK", "FAIL", "TIMEOUT", "ERR", "GPU", "Worker")

        def emit(self, record):
            msg = record.getMessage()
            if any(kw in msg for kw in self.STATUS_KEYWORDS):
                super().emit(record)
            self.flush()

    ch = WorkerStatusHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(ColoredFormatter(fmt=fmt, datefmt=datefmt))
    worker_log.addHandler(ch)

    worker_log.info("Worker %d started, GPU %d, log: %s", worker_id, gpu_id, log_file)
    flush_log(worker_log)

    failed = 0
    for task, cfg in jobs:
        if not run_collect_data_sh(task, cfg, gpu_id, script_dir, worker_log, log_file):
            failed += 1
    return failed


def main() -> int:
    """
    Load file-level config, build job list with strict order, assign to workers by len(GPU_PARALLEL).
    Order: (1) TASK order = TASK_TO_COLL order; (2) for each CFG, collect all TASKs then next CFG
    (i.e. outer loop CFG, inner loop TASK); (3) GPU parallel preserves this order, processing
    multiple jobs simultaneously. Exit 0 if all OK, 1 if any failed.
    Console shows only process status (START/OK/FAIL); log files capture all subprocess output.
    """
    task_to_coll, cfg_to_coll, gpu_parallel = load_config()
    os.environ["END_RESET_TO_INIT"] = "true" if END_RESET_TO_INIT else "false"

    # Setup parent process logging
    log = setup_parent_process_logging()
    log.info(
        "[main] TASK_TO_COLL=%s CFG_TO_COLL=%s GPU_PARALLEL=%s",
        task_to_coll,
        cfg_to_coll,
        gpu_parallel,
    )
    flush_log(log)

    # Order: for each CFG, all TASKs (CFG outer, TASK inner); pair (task, cfg) for run_collect_data_sh.
    product = [(task, cfg) for cfg in cfg_to_coll for task in task_to_coll]
    if not product:
        log.warning("[main] Empty Cartesian product; nothing to run.")
        return 0

    n_workers = len(gpu_parallel)
    worker_jobs = [[] for _ in range(n_workers)]
    for i, pair in enumerate(product):
        worker_jobs[i % n_workers].append(pair)

    args_list = [
        (worker_jobs[i], gpu_parallel[i], SCRIPT_DIR, i) for i in range(n_workers)
    ]
    # Process group: main is group leader so we control shutdown; workers get own group in initializer.
    try:
        os.setpgid(0, 0)
    except OSError:
        pass
    shutdown_event = threading.Event()
    _main_install_shutdown_handler(shutdown_event, log)

    def _terminate_pool_and_exit(exit_code: int) -> None:
        """
        Safely terminate all child processes with graceful shutdown.
        Strategy: SIGTERM -> wait 5s -> SIGKILL for remaining.
        No return; exits process.
        """
        log.warning("[main] Shutdown requested; terminating all workers...")
        flush_log(log)

        # Step 1: Send SIGTERM to pool (graceful shutdown)
        pool.terminate()

        # Step 2: Wait for workers to exit gracefully (real timeout; not Pool.join(timeout=...))
        _join_pool_workers_with_timeout(pool, SIGTERM_TIMEOUT_SEC)

        # Step 3: Force kill any remaining children
        remaining_children = list(mp.active_children())
        if remaining_children:
            log.warning(
                "[main] %d worker(s) did not exit gracefully, sending SIGKILL",
                len(remaining_children),
            )
            flush_log(log)
            for child in remaining_children:
                # Workers are process-group leaders (_worker_init); kill whole group (bash/sim).
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except OSError:
                    try:
                        os.kill(child.pid, signal.SIGKILL)
                    except OSError:
                        pass
            # Wait for SIGKILL to take effect
            time.sleep(0.5)

        log.warning("[main] All workers terminated. Exiting.")
        flush_log(log)
        os._exit(exit_code)

    pool = mp.Pool(processes=n_workers, initializer=_worker_init)
    try:
        results_obj = pool.starmap_async(worker, args_list)
        results = None
        while True:
            if shutdown_event.is_set():
                _terminate_pool_and_exit(143)
            try:
                results = results_obj.get(timeout=1.0)
                break
            except mp.TimeoutError:
                continue
    except KeyboardInterrupt:
        _terminate_pool_and_exit(130)
    finally:
        pool.close()
        pool.join()

    total_failed = sum(results)
    if total_failed > 0:
        log.error("[main] Done. %s task(s) failed.", total_failed)
        flush_log(log)
        return 1
    log.success("[main] Done. All tasks completed successfully.")
    flush_log(log)
    return 0

    n_workers = len(gpu_parallel)
    worker_jobs = [[] for _ in range(n_workers)]
    for i, pair in enumerate(product):
        worker_jobs[i % n_workers].append(pair)

    args_list = [
        (worker_jobs[i], gpu_parallel[i], SCRIPT_DIR, subprocess_print, i)
        for i in range(n_workers)
    ]
    # Process group: main is group leader so we control shutdown; workers get own group in initializer.
    try:
        os.setpgid(0, 0)
    except OSError:
        pass
    shutdown_event = threading.Event()
    _main_install_shutdown_handler(shutdown_event, log)

    def _terminate_pool_and_exit(exit_code: int) -> None:
        """
        Safely terminate all child processes with graceful shutdown.
        Strategy: SIGTERM -> wait 5s -> SIGKILL for remaining.
        No return; exits process.
        """
        log.warning("[main] Shutdown requested; terminating all workers...")
        flush_log(log)

        # Step 1: Send SIGTERM to pool (graceful shutdown)
        pool.terminate()

        # Step 2: Wait for workers to exit gracefully (real timeout; not Pool.join(timeout=...))
        _join_pool_workers_with_timeout(pool, SIGTERM_TIMEOUT_SEC)

        # Step 3: Force kill any remaining children
        remaining_children = list(mp.active_children())
        if remaining_children:
            log.warning(
                "[main] %d worker(s) did not exit gracefully, sending SIGKILL",
                len(remaining_children),
            )
            flush_log(log)
            for child in remaining_children:
                # Workers are process-group leaders (_worker_init); kill whole group (bash/sim).
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except OSError:
                    try:
                        os.kill(child.pid, signal.SIGKILL)
                    except OSError:
                        pass
            # Wait for SIGKILL to take effect
            time.sleep(0.5)

        log.warning("[main] All workers terminated. Exiting.")
        flush_log(log)
        os._exit(exit_code)

    pool = mp.Pool(processes=n_workers, initializer=_worker_init)
    try:
        results_obj = pool.starmap_async(worker, args_list)
        results = None
        while True:
            if shutdown_event.is_set():
                _terminate_pool_and_exit(143)
            try:
                results = results_obj.get(timeout=1.0)
                break
            except mp.TimeoutError:
                continue
    except KeyboardInterrupt:
        _terminate_pool_and_exit(130)
    finally:
        pool.close()
        pool.join()

    total_failed = sum(results)
    if total_failed > 0:
        log.error("[main] Done. %s task(s) failed.", total_failed)
        flush_log(log)
        return 1
    log.success("[main] Done. All tasks completed successfully.")
    flush_log(log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
