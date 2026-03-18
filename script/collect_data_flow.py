#!/usr/bin/env python3
"""
Batch data collection launcher: Cartesian product of task list x config list,
run via collect_data.sh in parallel. Config is passed via env vars from collect_data_flow.sh.

Dependencies: Python 3.6+, standard library only (subprocess, itertools, logging, multiprocessing).
Usage: invoked by collect_data_flow.sh; expects env TASK_TO_COLL, CFG_TO_COLL, GPU_PARALLEL
  (comma-separated; GPU_PARALLEL values are integers).

Call-chain note (from this script as caller):
  This script only invokes: collect_data.sh -> collect_data.py -> task env (load_robot/set_planner).
  (1) "Robot has no attribute left_planner": Downstream Robot.set_planner() sets left_planner only
      when left and right curobo config paths are equal; otherwise it uses subprocess+pipe and does
      not set left_planner. Later update_world_pcd() always accesses left_planner, so that path
      triggers AttributeError. Caller cannot fix this without changing robot.py.
  (2) Ctrl+C not aborting: This script uses a multiprocessing Pool; the main process blocks on
      pool.starmap() so SIGINT is not handled until a worker returns. Workers can also receive
      SIGINT. Mitigation here: starmap_async + get(timeout) loop and worker initializer to ignore
      SIGINT, so only the main process handles Ctrl+C and terminates the pool.
"""

import itertools
import logging
import multiprocessing as mp
import os
import re
import signal
import subprocess
import sys
import threading
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional


def _worker_ignore_sigint():
    """Ignore SIGINT in pool workers so only the main process handles Ctrl+C."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

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


def _read_stdout_and_log_wrapper(proc: subprocess.Popen, state: dict, log: logging.Logger) -> None:
    """Read subprocess stdout line by line and log wrapper progress messages. Used when SUBPROCESS_PRINT is False."""
    while True:
        line = proc.stdout.readline()
        if not line:
            break
        msg = _wrapper_progress_message(line, state)
        if msg:
            log.info("[wrapper] %s", msg)


def run_collect_data_sh(
    task: str, cfg: str, gpu_id: int, script_dir: Path, subprocess_print: bool
) -> bool:
    """
    Run collect_data.sh for one (task, config) pair on the given GPU.
    Input: task/cfg (str), gpu_id (int), script_dir (Path), subprocess_print (bool).
    Output: True on success, False on failure. Logs to module logger.
    When subprocess_print is False, only wrapper progress (seed test, result, saving video) is printed.
    """
    log = logging.getLogger("run_collect")
    cmd = ["bash", str(script_dir / "collect_data.sh"), task, cfg, str(gpu_id)]
    log.info("[run_collect] %s %s GPU%s", task, cfg, gpu_id)
    try:
        if subprocess_print:
            result = subprocess.run(
                cmd,
                cwd=str(script_dir),
                capture_output=False,
                text=True,
                timeout=3600,
            )
            out, err = None, None
        else:
            proc = subprocess.Popen(
                cmd,
                cwd=str(script_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            state = {"in_data_collection": False, "data_collection_episode_index": 0}
            reader = threading.Thread(
                target=_read_stdout_and_log_wrapper,
                args=(proc, state, log),
                daemon=True,
            )
            reader.start()
            try:
                proc.wait(timeout=3600)
            except subprocess.TimeoutExpired:
                proc.terminate()
                proc.wait()
                raise
            reader.join(timeout=5.0)
            result = type("Result", (), {"returncode": proc.returncode, "stdout": None, "stderr": None})()

        if result.returncode != 0:
            log.error("[run_collect] FAILED %s %s stderr: %s", task, cfg, result.stderr or result.stdout)
            return False
        log.success("[run_collect] OK %s %s", task, cfg)
        return True
    except subprocess.TimeoutExpired:
        log.error("[run_collect] TIMEOUT %s %s", task, cfg)
        return False
    except Exception as e:
        log.exception("[run_collect] ERR %s %s: %s", task, cfg, e)
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
    Load config from env, build Cartesian product TASK_TO_COLL x CFG_TO_COLL,
    assign jobs to workers by len(GPU_PARALLEL), run workers in parallel.
    Exit 0 if all OK, 1 if any failed.
    When SUBPROCESS_PRINT is False, only wrapper progress (seed test, result, saving video) is printed.
    """
    task_to_coll, cfg_to_coll, gpu_parallel = load_config()
    subprocess_print = _parse_subprocess_print()

    log_file = setup_logging()
    log = logging.getLogger("main")
    log.info("[main] Log file: %s", log_file)
    log.info("[main] TASK_TO_COLL=%s CFG_TO_COLL=%s GPU_PARALLEL=%s SUBPROCESS_PRINT=%s",
             task_to_coll, cfg_to_coll, gpu_parallel, subprocess_print)

    product = list(itertools.product(task_to_coll, cfg_to_coll))
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
    pool = mp.Pool(processes=n_workers, initializer=_worker_ignore_sigint)
    try:
        results_obj = pool.starmap_async(worker, args_list)
        while True:
            try:
                results = results_obj.get(timeout=1.0)
                break
            except mp.TimeoutError:
                continue
    except KeyboardInterrupt:
        log.warning("[main] Interrupted (Ctrl+C); terminating workers.")
        pool.terminate()
        pool.join()
        return 130
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
