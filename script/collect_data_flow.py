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
"""

import logging
import multiprocessing as mp
import os
import re
import queue
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional

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
# When False, raw subprocess stdout/stderr is not written to the terminal; full stream goes to per-job log files only.
SUBPROCESS_PRINT = False
END_RESET_TO_INIT = True


def _worker_ignore_sigint():
    """Ignore SIGINT in pool workers so only the main process handles Ctrl+C."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _worker_init():
    """
    Pool worker initializer: ignore SIGINT, create own process group, install SIGTERM
    handler to kill entire group (worker + Popen children) so no orphans on shutdown.
    """
    _worker_ignore_sigint()
    _worker_process_group_and_sigterm()


def _worker_process_group_and_sigterm():
    """
    Make this worker the leader of its own process group and install SIGTERM handler
    that kills the whole group (worker + any Popen children). Ensures no orphan subprocesses
    when the main process requests shutdown.
    """
    try:
        os.setpgid(0, 0)
    except OSError:
        pass

    def _kill_process_group(_signum, _frame):
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        pgid = os.getpgrp()
        try:
            os.killpg(pgid, signal.SIGTERM)
        except OSError:
            pass
        # Escalate to SIGKILL to avoid residual GPU-holding children.
        time.sleep(0.2)
        try:
            os.killpg(pgid, signal.SIGKILL)
        except OSError:
            pass

    signal.signal(signal.SIGTERM, _kill_process_group)


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


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _pool_worker_pids(pool) -> list:
    workers = getattr(pool, "_pool", None) or []
    pids = []
    for worker in workers:
        if worker is None:
            continue
        pid = getattr(worker, "pid", None)
        if isinstance(pid, int) and pid > 0:
            pids.append(pid)
    return pids


def _kill_worker_groups(pool, log: logging.Logger, grace_sec: float = 2.0) -> None:
    """
    Forcefully reclaim worker process groups.
    Workers are group leaders (pgid == worker pid), so killpg(pid, sig) can terminate
    worker + collect_data.sh + python descendants that still hold GPU memory.
    """
    worker_pids = _pool_worker_pids(pool)
    if not worker_pids:
        return

    log.warning("[main] Worker groups to reclaim: %s", worker_pids)
    for pid in worker_pids:
        try:
            os.killpg(pid, signal.SIGTERM)
        except OSError:
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass

    deadline = time.time() + max(0.1, grace_sec)
    while time.time() < deadline:
        if not any(_pid_alive(pid) for pid in worker_pids):
            return
        time.sleep(0.1)

    survivors = [pid for pid in worker_pids if _pid_alive(pid)]
    if survivors:
        log.warning("[main] Escalate SIGKILL for worker groups: %s", survivors)
    for pid in survivors:
        try:
            os.killpg(pid, signal.SIGKILL)
        except OSError:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass


# Repo root (parent of script/); collect_data.sh and logs live here.
SCRIPT_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = SCRIPT_DIR / "logs"
UTC8 = timezone(timedelta(hours=8))

# ANSI
RED = "\033[31m"
GREEN = "\033[32m"
BLUE = "\033[34m"
YELLOW = "\033[33m"
RESET = "\033[0m"


# Custom level for success messages (green)
SUCCESS_LEVEL = 25
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")


def success(self, msg, *args, **kwargs):
    if self.isEnabledFor(SUCCESS_LEVEL):
        self._log(SUCCESS_LEVEL, msg, args, **kwargs)


logging.Logger.success = success


class FlushFileHandler(logging.FileHandler):
    """File handler that flushes after each record so log files have no write-back delay."""

    def emit(self, record):
        super().emit(record)
        self.flush()


class FlushStreamHandler(logging.StreamHandler):
    """Stream handler that flushes after each record (immediate console feedback)."""

    def emit(self, record):
        super().emit(record)
        self.flush()


class ColoredFormatter(logging.Formatter):
    """Format log records with [stage] and color for level (WARNING/ERROR red, SUCCESS green, info blue)."""

    LEVEL_COLORS = {
        logging.DEBUG: BLUE,
        logging.INFO: RESET,
        SUCCESS_LEVEL: GREEN,
        logging.WARNING: RED,
        logging.ERROR: RED,
    }

    def __init__(self, fmt=None, datefmt=None, use_color=True):
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.use_color = use_color

    def format(self, record):
        if self.use_color and record.levelno in self.LEVEL_COLORS:
            color = self.LEVEL_COLORS[record.levelno]
            record.msg = f"{color}{record.msg}{RESET}"
        return super().format(record)


def _timestamp_utc8():
    """Return current timestamp string YYYYMMDDHHMMSS in UTC+8."""
    return datetime.now(UTC8).strftime("%Y%m%d%H%M%S")


def _sanitize_log_token(token: str) -> str:
    """Convert arbitrary token to a filesystem-safe ASCII fragment."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", token.strip())
    return cleaned or "unknown"


def _subprocess_log_path(task: str, cfg: str, gpu_id: int) -> Path:
    """Build per-subprocess log path under LOG_DIR."""
    task_safe = _sanitize_log_token(task)
    cfg_safe = _sanitize_log_token(cfg)
    return LOG_DIR / (
        f"collect_data_subproc_{task_safe}_{cfg_safe}_gpu{gpu_id}_pid{os.getpid()}_{_timestamp_utc8()}.log"
    )


def setup_logging():
    """
    Configure root logger: file in LOG_DIR named collect_data_flow_<timestamp>.log,
    and console with colors. Idempotent for repeated calls in same process.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"collect_data_flow_{_timestamp_utc8()}.log"
    fmt = "%(asctime)s [%(name)s] [%(levelname)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    root = logging.getLogger()
    if root.handlers:
        return str(log_file)
    root.setLevel(logging.DEBUG)

    fh = FlushFileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    root.addHandler(fh)

    ch = FlushStreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(ColoredFormatter(fmt=fmt, datefmt=datefmt))
    root.addHandler(ch)

    return str(log_file)


def _wrapper_progress_message(line: str, state: dict) -> Optional[str]:
    """
    Map one line of subprocess stdout to a single wrapper progress message, or None.
    State: in_data_collection (bool), data_collection_episode_index (int).
    Used only when SUBPROCESS_PRINT is False.
    """
    line_stripped = line.strip()
    if not line_stripped:
        return None
    # Strip ANSI for matching
    line_plain = re.sub(r"\033\[[\d;]*m", "", line_stripped)

    # Seed phase: "simulate data episode X success! (seed = Y)" or "fail! (seed = Y)"
    m = re.search(
        r"simulate data episode (\d+) (success|fail)! \(seed = (\d+)\)", line_plain
    )
    if m:
        idx, result, seed = m.group(1), m.group(2), m.group(3)
        return f"Seed test #{idx} (seed={seed}), result: {result}"

    # "Exist seed file, Start from: X / Y"
    m = re.search(r"Exist seed file, Start from: (\d+) / (\d+)", line_plain)
    if m:
        return f"Resuming from seed {m.group(1)}, {m.group(2)} succeeded so far"

    # "[Start Data Collection]"
    if "[Start Data Collection]" in line_plain:
        state["in_data_collection"] = True
        state["data_collection_episode_index"] = 0
        return "Data collection started"

    # Data collection: "Task name: ..." (per episode) -> saving video N
    if state.get("in_data_collection") and "Task name:" in line_plain:
        n = state["data_collection_episode_index"]
        state["data_collection_episode_index"] = n + 1
        return f"Saving video #{n} (current episode)"

    # "Folder ... deleted successfully" -> saved video N (episode index is the one we just finished)
    if "deleted successfully" in line_plain and state.get("in_data_collection"):
        n = max(0, state.get("data_collection_episode_index", 1) - 1)
        return f"Saved video #{n}"

    # "Complete simulation, failed X times / Y tries"
    m = re.search(r"Complete simulation, failed .+? (\d+) tries", line_plain)
    if m:
        return f"Seed phase done ({m.group(1)} tries)"

    return None


def _run_tag(task: str, cfg: str, use_color: bool = True) -> str:
    """Return a consistent prefix for log lines: [task][cfg]. use_color adds ANSI for console."""
    if use_color:
        return f"{BLUE}[{task}][{cfg}]{RESET}"
    return f"[{task}][{cfg}]"


def _log_file_signature(path: Path):
    """Return (mtime_ns, size) for idle detection, or None if stat fails."""
    try:
        st = path.stat()
        mt = getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))
        return (mt, st.st_size)
    except OSError:
        return None


def _read_stdout_and_log_wrapper(
    proc: subprocess.Popen,
    state: dict,
    log: logging.Logger,
    task: str,
    cfg: str,
    subprocess_print: bool,
    output_queue: "queue.Queue[str]",
) -> None:
    """Read subprocess stdout, optionally mirror to console, and emit wrapper progress."""
    tag = _run_tag(task, cfg)
    try:
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            output_queue.put(line)
            if subprocess_print:
                sys.stdout.write(line)
                sys.stdout.flush()
            else:
                msg = _wrapper_progress_message(line, state)
                if msg:
                    log.info("%s wrapper: %s", tag, msg)
    finally:
        output_queue.put(None)


def run_collect_data_sh(
    task: str, cfg: str, gpu_id: int, script_dir: Path, subprocess_print: bool
) -> bool:
    """
    Run collect_data.sh for one (task, config) pair on the given GPU.
    Input: task/cfg (str), gpu_id (int), script_dir (Path), subprocess_print (bool).
    Output: True on success, False on failure. Logs to module logger.
    When subprocess_print is False, raw child stdout/stderr is not mirrored to the terminal; every line
    is appended to the per-job subprocess log file (flushed). Wrapper progress lines still go through
    the main logger (console + collect_data_flow log file).

    Idle timeout: env COLLECT_DATA_IDLE_TIMEOUT_SEC (default 3600). While the child runs, the parent
    polls subproc_log_file mtime/size; any change resets the idle clock. <=0 disables idle kill.
    """
    log = logging.getLogger("run_collect")
    tag = _run_tag(task, cfg)
    cmd = ["bash", str(script_dir / "collect_data.sh"), task, cfg, str(gpu_id)]
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    subproc_log_file = _subprocess_log_path(task, cfg, gpu_id)
    idle_timeout_sec = float(os.environ.get("COLLECT_DATA_IDLE_TIMEOUT_SEC", "3600"))
    log.info("%s START GPU%s", tag, gpu_id)
    log.info("%s Subprocess log: %s", tag, subproc_log_file)
    log.info(
        "%s idle_timeout_sec=%s (COLLECT_DATA_IDLE_TIMEOUT_SEC, log-file mtime/size)",
        tag,
        idle_timeout_sec,
    )
    try:
        proc_env = os.environ.copy()
        proc_env.setdefault("PYTHONUNBUFFERED", "1")
        with subproc_log_file.open(
            "w", encoding="utf-8", buffering=1
        ) as subproc_log_fp:
            # Do not use start_new_session: keep subprocess in worker's process group
            # so that worker's SIGTERM handler (killpg) terminates this child too.
            proc = subprocess.Popen(
                cmd,
                cwd=str(script_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=proc_env,
            )
            state = {"in_data_collection": False, "data_collection_episode_index": 0}
            output_queue = queue.Queue()
            reader = threading.Thread(
                target=_read_stdout_and_log_wrapper,
                args=(proc, state, log, task, cfg, subprocess_print, output_queue),
                daemon=True,
            )
            reader.start()
            try:
                stream_done = False
                last_sig = _log_file_signature(subproc_log_file)
                last_change = time.time()
                while True:
                    if not stream_done:
                        try:
                            line = output_queue.get(timeout=0.2)
                            if line is None:
                                stream_done = True
                            else:
                                subproc_log_fp.write(line)
                                subproc_log_fp.flush()
                        except queue.Empty:
                            pass
                    if stream_done and proc.poll() is not None:
                        break
                    if idle_timeout_sec > 0 and proc.poll() is None:
                        sig = _log_file_signature(subproc_log_file)
                        if sig is not None:
                            if last_sig is None or sig != last_sig:
                                last_sig = sig
                                last_change = time.time()
                            elif time.time() - last_change > idle_timeout_sec:
                                raise subprocess.TimeoutExpired(
                                    cmd=cmd, timeout=idle_timeout_sec
                                )
            except subprocess.TimeoutExpired:
                proc.terminate()
                proc.wait()
                raise
            reader.join(timeout=5.0)
            result = type(
                "Result",
                (),
                {"returncode": proc.returncode, "stdout": None, "stderr": None},
            )()

        if result.returncode != 0:
            log.error("%s FAILED. See subprocess log: %s", tag, subproc_log_file)
            return False
        log.success("%s OK", tag)
        return True
    except subprocess.TimeoutExpired:
        log.error(
            "%s IDLE_TIMEOUT (no subprocess log change for %ss): %s",
            tag,
            idle_timeout_sec,
            subproc_log_file,
        )
        return False
    except Exception as e:
        log.exception("%s ERR: %s", tag, e)
        return False


def worker(jobs: list, gpu_id: int, script_dir: Path, subprocess_print: bool) -> int:
    """
    Run a list of (task, cfg) jobs on one GPU. Returns number of failures.
    Input: jobs list of (task, cfg), gpu_id (int), script_dir (Path), subprocess_print (bool).
    Output: int, count of failed runs.
    """
    failed = 0
    for task, cfg in jobs:
        if not run_collect_data_sh(task, cfg, gpu_id, script_dir, subprocess_print):
            failed += 1
    return failed


def main() -> int:
    """
    Load file-level config, build job list with strict order, assign to workers by len(GPU_PARALLEL).
    Order: (1) TASK order = TASK_TO_COLL order; (2) for each CFG, collect all TASKs then next CFG
    (i.e. outer loop CFG, inner loop TASK); (3) GPU parallel preserves this order, processing
    multiple jobs simultaneously. Exit 0 if all OK, 1 if any failed.
    When SUBPROCESS_PRINT is False, raw subprocess output is only written to per-job log files; the
    terminal still shows high-level wrapper lines and main-process messages. Log handlers flush after
    each record to avoid delayed writes.
    Each run_collect line is prefixed with [task][cfg] so parallel workers' output can be distinguished.
    """
    task_to_coll, cfg_to_coll, gpu_parallel = load_config()
    subprocess_print = bool(SUBPROCESS_PRINT)
    os.environ["END_RESET_TO_INIT"] = "true" if END_RESET_TO_INIT else "false"

    log_file = setup_logging()
    log = logging.getLogger("main")
    log.info("[main] Log file: %s", log_file)
    log.info(
        "[main] TASK_TO_COLL=%s CFG_TO_COLL=%s GPU_PARALLEL=%s SUBPROCESS_PRINT=%s",
        task_to_coll,
        cfg_to_coll,
        gpu_parallel,
        subprocess_print,
    )

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
        (worker_jobs[i], gpu_parallel[i], SCRIPT_DIR, subprocess_print)
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
        """Kill all child processes (workers and their subprocesses) then exit. No return."""
        log.warning("[main] Shutdown requested; terminating all workers and exiting.")
        _kill_worker_groups(pool, log, grace_sec=2.0)
        pool.terminate()
        try:
            pool.join()
        except Exception:
            pass
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
        return 1
    log.success("[main] Done. All tasks completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
