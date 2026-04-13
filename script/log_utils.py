"""
Purpose: Logging utilities with per-process log files and immediate flush.
Dependencies: Python 3.6+, logging, sys, os
Usage Example:
    from script.log_utils import setup_child_process_logging, flush_log
    log = setup_child_process_logging("collect_data", task_name="hanging_mug", cfg="demo_clean")
    log.info("Message")
    flush_log()

@input: Process name, task/config identifiers
@output: Logger instance with file and console handlers
@scenario: Each child process gets its own log file, all output flushed immediately
"""

import logging
import sys
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional


LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
UTC8 = timezone(timedelta(hours=8))

RED = "\033[31m"
GREEN = "\033[32m"
BLUE = "\033[34m"
YELLOW = "\033[33m"
RESET = "\033[0m"

SUCCESS_LEVEL = 25
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")


def success(self, msg, *args, **kwargs):
    if self.isEnabledFor(SUCCESS_LEVEL):
        self._log(SUCCESS_LEVEL, msg, args, **kwargs)


logging.Logger.success = success


class ImmediateFlushFileHandler(logging.FileHandler):
    """
    FileHandler that flushes after every emit.
    """

    def emit(self, record):
        super().emit(record)
        self.flush()


class ImmediateFlushStreamHandler(logging.StreamHandler):
    """
    StreamHandler that flushes after every emit.
    """

    def emit(self, record):
        super().emit(record)
        self.flush()


class ColoredFormatter(logging.Formatter):
    """
    Format log records with [stage] and color for level.
    """

    LEVEL_COLORS = {
        logging.DEBUG: BLUE,
        logging.INFO: RESET,
        SUCCESS_LEVEL: GREEN,
        logging.WARNING: YELLOW,
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
    """
    Return current timestamp string YYYYMMDDHHMMSS in UTC+8.

    @input: None
    @output: str, formatted timestamp
    @scenario: Generate unique timestamp for log filename
    """
    return datetime.now(UTC8).strftime("%Y%m%d%H%M%S")


def setup_process_logging(
    process_name: str,
    task_name: Optional[str] = None,
    cfg_name: Optional[str] = None,
    use_color: bool = True,
) -> logging.Logger:
    """
    Setup logging for a process with immediate flush to file and console.

    @input:
        process_name: str, name of the process (e.g., "collect_data", "collect_data_flow")
        task_name: str or None, task name for child processes
        cfg_name: str or None, config name for child processes
        use_color: bool, whether to use ANSI colors in console
    @output: logging.Logger, configured logger
    @scenario: Each process gets its own log file in ./logs/
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if task_name and cfg_name:
        log_filename = f"{process_name}_{task_name}_{cfg_name}_{_timestamp_utc8()}.log"
    else:
        log_filename = f"{process_name}_{_timestamp_utc8()}.log"

    log_file = LOG_DIR / log_filename

    logger = logging.getLogger(process_name)
    logger.setLevel(logging.DEBUG)
    logger.handlers = []

    fmt = "%(asctime)s [%(name)s] %(levelname)s %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    fh = ImmediateFlushFileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    logger.addHandler(fh)

    ch = ImmediateFlushStreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(ColoredFormatter(fmt=fmt, datefmt=datefmt, use_color=use_color))
    logger.addHandler(ch)

    logger.info(f"Log file: {log_file}")
    logger.info(
        f"Process: {process_name}"
        + (f", Task: {task_name}" if task_name else "")
        + (f", Config: {cfg_name}" if cfg_name else "")
    )

    return logger


def setup_child_process_logging(
    task_name: str,
    cfg_name: str,
) -> logging.Logger:
    """
    Setup logging for collect_data.py child process.

    @input:
        task_name: str, task name
        cfg_name: str, config name
    @output: logging.Logger, configured logger
    @scenario: Child process gets unique log file with task/config in name
    """
    return setup_process_logging(
        process_name="collect_data",
        task_name=task_name,
        cfg_name=cfg_name,
    )


def setup_parent_process_logging() -> logging.Logger:
    """
    Setup logging for collect_data_flow.py parent process.

    @input: None
    @output: logging.Logger, configured logger
    @scenario: Parent process gets its own log file
    """
    return setup_process_logging(process_name="collect_data_flow")


def flush_log(logger: Optional[logging.Logger] = None) -> None:
    """
    Flush all handlers for a logger.

    @input:
        logger: logging.Logger or None, if None flush root logger
    @output: None
    @scenario: Ensure all log output is written immediately
    """
    if logger is None:
        logger = logging.getLogger()

    for handler in logger.handlers:
        handler.flush()


def log_and_flush(
    logger: logging.Logger, level: int, msg: str, *args, **kwargs
) -> None:
    """
    Log a message and immediately flush.

    @input:
        logger: logging.Logger
        level: int, log level
        msg: str, message
        args, kwargs: additional arguments
    @output: None
    @scenario: Log and flush in one call for critical messages
    """
    logger.log(level, msg, *args, **kwargs)
    flush_log(logger)
