#!/usr/bin/env python3
"""
Batch data collection launcher: Cartesian product of task list x config list,
run via collect_data.sh in parallel. Config is passed via env vars from collect_data_flow.sh.

Dependencies: Python 3.6+, standard library only (subprocess, itertools, logging, multiprocessing).
Usage: invoked by collect_data_flow.sh; expects env TASK_TO_COLL, CFG_TO_COLL, GPU_PARALLEL
  (comma-separated; GPU_PARALLEL values are integers).
"""

import itertools
import logging
import multiprocessing as mp
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

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


def run_collect_data_sh(task: str, cfg: str, gpu_id: int, script_dir: Path) -> bool:
    """
    Run collect_data.sh for one (task, config) pair on the given GPU.
    Input: task/cfg (str), gpu_id (int), script_dir (Path to RoboTwin).
    Output: True on success, False on failure. Logs to module logger.
    """
    log = logging.getLogger("run_collect")
    cmd = ["bash", str(script_dir / "collect_data.sh"), task, cfg, str(gpu_id)]
    log.info("[run_collect] %s %s GPU%s", task, cfg, gpu_id)
    try:
        result = subprocess.run(
            cmd,
            cwd=str(script_dir),
            capture_output=False,
            text=True,
            timeout=3600,
        )
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


def worker(jobs: list, gpu_id: int, script_dir: Path) -> int:
    """
    Run a list of (task, cfg) jobs on one GPU. Returns number of failures.
    Input: jobs list of (task, cfg), gpu_id (int), script_dir (Path).
    Output: int, count of failed runs.
    """
    failed = 0
    for task, cfg in jobs:
        if not run_collect_data_sh(task, cfg, gpu_id, script_dir):
            failed += 1
    return failed


def main() -> int:
    """
    Load config from env, build Cartesian product TASK_TO_COLL x CFG_TO_COLL,
    assign jobs to workers by len(GPU_PARALLEL), run workers in parallel.
    Exit 0 if all OK, 1 if any failed.
    """
    task_to_coll, cfg_to_coll, gpu_parallel = load_config()

    log_file = setup_logging()
    log = logging.getLogger("main")
    log.info("[main] Log file: %s", log_file)
    log.info("[main] TASK_TO_COLL=%s CFG_TO_COLL=%s GPU_PARALLEL=%s", task_to_coll, cfg_to_coll, gpu_parallel)

    product = list(itertools.product(task_to_coll, cfg_to_coll))
    if not product:
        log.warning("[main] Empty Cartesian product; nothing to run.")
        return 0

    n_workers = len(gpu_parallel)
    worker_jobs = [[] for _ in range(n_workers)]
    for i, pair in enumerate(product):
        worker_jobs[i % n_workers].append(pair)

    args_list = [
        (worker_jobs[i], gpu_parallel[i], SCRIPT_DIR)
        for i in range(n_workers)
    ]
    with mp.Pool(processes=n_workers) as pool:
        results = pool.starmap(worker, args_list)

    total_failed = sum(results)
    if total_failed > 0:
        log.error("[main] Done. %s task(s) failed.", total_failed)
        return 1
    log.success("[main] Done. All tasks completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
