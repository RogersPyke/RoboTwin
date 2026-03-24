#!/usr/bin/env python3
"""
Batch data collection launcher: Cartesian product of task list x config list,
run via collect_data.sh in parallel. Config is passed via env vars from collect_data_flow.sh.

Dependencies: Python 3.6+, standard library only (subprocess, logging, multiprocessing).
Usage: invoked by collect_data_flow.sh; expects env TASK_TO_COLL, CFG_TO_COLL, GPU_PARALLEL
  (comma-separated; GPU_PARALLEL values are integers).

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
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional


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
    _worker_watch_main_parent()


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
        try:
            os.killpg(os.getpgrp(), signal.SIGTERM)
        except OSError:
            pass
    signal.signal(signal.SIGTERM, _kill_process_group)


def _worker_watch_main_parent():
    """
    Watch main-process PID from env and terminate this worker group when main dies.
    This covers abrupt parent exits where no signal reaches workers.
    """
    main_pid_raw = os.environ.get("COLLECT_FLOW_MAIN_PID", "").strip()
    if not main_pid_raw.isdigit():
        return
    main_pid = int(main_pid_raw)

    def _watchdog():
        while True:
            try:
                os.kill(main_pid, 0)
            except OSError:
                try:
                    os.killpg(os.getpgrp(), signal.SIGTERM)
                except OSError:
                    pass
                os._exit(143)
            time.sleep(0.5)

    t = threading.Thread(target=_watchdog, daemon=True)
    t.start()


def _main_install_shutdown_handler(shutdown_event: threading.Event, log: logging.Logger):
    """
    Install SIGTERM handler so that on external kill (e.g. systemd, kill <pid>), the main
    process sets shutdown_event and then the main loop will terminate the pool and exit.
    Does not run cleanup inside the handler (not signal-safe); only sets the flag.
    """

    def _handler(_signum, _frame):
        shutdown_event.set()

    signal.signal(signal.SIGTERM, _handler)

# ---------------------------------------------------------------------------
# Config from env (set by collect_data_flow.sh)
# ---------------------------------------------------------------------------
def _parse_list(env_key: str) -> list:
    """Parse comma-separated env var into list of stripped strings. Returns empty list if unset."""
    val = os.environ.get(env_key, "")
    return [s.strip() for s in val.split(",") if s.strip()]


def _parse_gpu_parallel(env_key: str = "GPU_PARALLEL") -> list:
    """Parse comma-separated GPU IDs into list of int. Returns [0, 0] if unset."""
    val = os.environ.get(env_key, "0,0")
    parts = [s.strip() for s in val.split(",") if s.strip()]
    if not parts:
        return [0, 0]
    return [int(x) for x in parts]


def _parse_subprocess_print(env_key: str = "SUBPROCESS_PRINT") -> bool:
    """Parse SUBPROCESS_PRINT from environment. True only for 'true'/'True'/'1'; else False."""
    val = os.environ.get(env_key, "false").strip().lower()
    return val in ("true", "1")


def _parse_fail_seed_skip_threshold(env_key: str = "FAIL_SEED_SKIP_THRESHOLD") -> int:
    """Parse fail-seed threshold from environment; invalid/non-positive values fallback to 100."""
    raw = os.environ.get(env_key, "100").strip()
    try:
        value = int(raw)
        if value <= 0:
            return 100
        return value
    except Exception:
        return 100


def load_config():
    """Load TASK_TO_COLL, CFG_TO_COLL, GPU_PARALLEL from environment. Returns (tasks, configs, gpu_list)."""
    tasks = _parse_list("TASK_TO_COLL")
    configs = _parse_list("CFG_TO_COLL")
    gpu_list = _parse_gpu_parallel()
    return tasks, configs, gpu_list


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


def setup_logging():
    """
    Configure root logger: file in LOG_DIR named collect_data_flow_<timestamp>.log,
    and console with colors. Idempotent for repeated calls in same process.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"collect_data_flow_{_timestamp_utc8()}.log"
    fmt = "%(asctime)s [%(name)s] %(levelname)s %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    root = logging.getLogger()
    if root.handlers:
        return str(log_file)
    root.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    root.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
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
    m = re.search(r"simulate data episode (\d+) (success|fail)! \(seed = (\d+)\)", line_plain)
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


def _read_stdout_and_log_wrapper(
    proc: subprocess.Popen, state: dict, log: logging.Logger, task: str, cfg: str
) -> None:
    """Read subprocess stdout line by line and log wrapper progress messages. Used when SUBPROCESS_PRINT is False."""
    tag = _run_tag(task, cfg)
    while True:
        line = proc.stdout.readline()
        if not line:
            break
        msg = _wrapper_progress_message(line, state)
        if msg:
            log.info("%s wrapper: %s", tag, msg)


def run_collect_data_sh(
    task: str,
    cfg: str,
    gpu_id: int,
    script_dir: Path,
    subprocess_print: bool,
    fail_seed_skip_threshold: int,
) -> bool:
    """
    Run collect_data.sh for one (task, config) pair on the given GPU.
    Input: task/cfg (str), gpu_id (int), script_dir (Path), subprocess_print (bool).
    Output: True on success, False on failure. Logs to module logger.
    When subprocess_print is False, only wrapper progress (seed test, result, saving video) is printed.
    """
    log = logging.getLogger("run_collect")
    tag = _run_tag(task, cfg)
    cmd = ["bash", str(script_dir / "collect_data.sh"), task, cfg, str(gpu_id)]
    log.info("%s START GPU%s", tag, gpu_id)
    try:
        # Always read subprocess output to support fail-seed threshold skip logic.
        # Do not use start_new_session: keep subprocess in worker's process group
        # so that worker's SIGTERM handler (killpg) terminates this child too.
        proc = subprocess.Popen(
            cmd,
            cwd=str(script_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        state = {"in_data_collection": False, "data_collection_episode_index": 0}
        fail_seed_count = 0
        fail_seed_limit_hit = False

        while True:
            line = proc.stdout.readline()
            if not line:
                break

            line_clean = line.rstrip("\n")
            if subprocess_print:
                print(line_clean, flush=True)
            else:
                msg = _wrapper_progress_message(line_clean, state)
                if msg:
                    log.info("%s wrapper: %s", tag, msg)

            if "simulate data episode" in line_clean and " fail! (seed = " in line_clean:
                fail_seed_count += 1
                if fail_seed_count > fail_seed_skip_threshold:
                    fail_seed_limit_hit = True
                    log.warning(
                        "%s SKIP: fail-seed count %s exceeded threshold %s.",
                        tag,
                        fail_seed_count,
                        fail_seed_skip_threshold,
                    )
                    proc.terminate()
                    break

        try:
            proc.wait(timeout=3600)
        except subprocess.TimeoutExpired:
            proc.terminate()
            proc.wait()
            raise

        if fail_seed_limit_hit:
            return False

        result = type("Result", (), {"returncode": proc.returncode, "stdout": None, "stderr": None})()

        if result.returncode != 0:
            log.error("%s FAILED stderr: %s", tag, result.stderr or result.stdout)
            return False
        log.success("%s OK", tag)
        return True
    except subprocess.TimeoutExpired:
        log.error("%s TIMEOUT", tag)
        return False
    except Exception as e:
        log.exception("%s ERR: %s", tag, e)
        return False


def main() -> int:
    """
    Load config from env, build job list with strict order, assign to workers by len(GPU_PARALLEL).
    Order: (1) TASK order = TASK_TO_COLL order; (2) for each CFG, collect all TASKs then next CFG
    (i.e. outer loop CFG, inner loop TASK); (3) GPU parallel preserves this order, processing
    multiple jobs simultaneously. Exit 0 if all OK, 1 if any failed.
    When SUBPROCESS_PRINT is False, only wrapper progress (seed test, result, saving video) is printed.
    Each run_collect line is prefixed with [task][cfg] so parallel workers' output can be distinguished.
    """
    task_to_coll, cfg_to_coll, gpu_parallel = load_config()
    subprocess_print = _parse_subprocess_print()
    fail_seed_skip_threshold = _parse_fail_seed_skip_threshold()

    log_file = setup_logging()
    log = logging.getLogger("main")
    log.info("[main] Log file: %s", log_file)
    log.info("[main] TASK_TO_COLL=%s CFG_TO_COLL=%s GPU_PARALLEL=%s SUBPROCESS_PRINT=%s FAIL_SEED_SKIP_THRESHOLD=%s",
             task_to_coll, cfg_to_coll, gpu_parallel, subprocess_print, fail_seed_skip_threshold)

    # Order: for each CFG, all TASKs (CFG outer, TASK inner).
    product = [(task, cfg) for cfg in cfg_to_coll for task in task_to_coll]
    if not product:
        log.warning("[main] Empty Cartesian product; nothing to run.")
        return 0

    n_workers = len(gpu_parallel)
    # Dynamic refill: keep pool busy with up to n_workers jobs at all times.
    # GPU assignment follows submission order (round-robin over GPU_PARALLEL).
    job_args = []
    for i, (task, cfg) in enumerate(product):
        gpu_id = gpu_parallel[i % n_workers]
        job_args.append(
            (
                task,
                cfg,
                gpu_id,
                SCRIPT_DIR,
                subprocess_print,
                fail_seed_skip_threshold,
            )
        )
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
        pool.terminate()
        pool.join()
        os._exit(exit_code)

    os.environ["COLLECT_FLOW_MAIN_PID"] = str(os.getpid())
    pool = mp.Pool(processes=n_workers, initializer=_worker_init)
    try:
        results_obj = pool.starmap_async(run_collect_data_sh, job_args)
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

    total_failed = sum(0 if ok else 1 for ok in results)
    if total_failed > 0:
        log.error("[main] Done. %s task(s) failed.", total_failed)
        return 1
    log.success("[main] Done. All tasks completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
