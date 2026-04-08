"""
Shared helpers for policy __flow.py schedulers: per-job log files, child env, failure excerpts.

Usage: imported from ACT/DP/TinyVLA __flow.py after adding RoboTwin/policy_util/ to sys.path.
"""

from __future__ import annotations

import re
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, TextIO

TRACE_MARK = "Traceback (most recent call last):"

# Tail window for failure excerpt (bytes read from end of file when file is large).
DEFAULT_MAX_TAIL_BYTES = 262144
# Max lines to print after choosing traceback block or plain tail.
DEFAULT_MAX_TAIL_LINES = 128


def utc8_now_str() -> str:
    tz8 = timezone(timedelta(hours=8))
    return datetime.now(tz8).strftime("%Y%m%d%H%M%S")


def safe_filename_part(s: str) -> str:
    out = re.sub(r"[^0-9A-Za-z._-]+", "_", s.strip())
    return out[:180] if len(out) > 180 else out


def ensure_logs_dir(base_dir: Path) -> Path:
    d = base_dir / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def rename_log_with_pid(tmp_path: Path, final_path: Path) -> None:
    try:
        if tmp_path != final_path and tmp_path.is_file():
            tmp_path.rename(final_path)
    except OSError:
        pass


def bash_lc_cmd(script: str) -> List[str]:
    inner = ["bash", "-lc", script]
    if shutil.which("stdbuf"):
        return ["stdbuf", "-oL", "-eL"] + inner
    return inner


def open_flow_text_log(tmp_path: Path) -> TextIO:
    return open(tmp_path, "w", buffering=1, encoding="utf-8", errors="replace")


def inject_flow_child_env(env: Dict[str, str]) -> Dict[str, str]:
    """Unbuffered Python stdio and no user-site for reproducible child behavior."""
    out = dict(env)
    out["PYTHONUNBUFFERED"] = "1"
    out["PYTHONNOUSERSITE"] = "1"
    return out


def _read_tail_bytes(path: Path, max_bytes: int) -> bytes:
    with open(path, "rb") as f:
        f.seek(0, 2)
        sz = f.tell()
        if sz <= max_bytes:
            f.seek(0)
            return f.read()
        f.seek(-max_bytes, 2)
        return f.read()


def dump_log_tail_to_stderr(
    log_path: Path,
    header: str,
    *,
    max_lines: int = DEFAULT_MAX_TAIL_LINES,
    max_tail_bytes: int = DEFAULT_MAX_TAIL_BYTES,
) -> None:
    """
    On subprocess failure, print an excerpt to stderr: prefer last Traceback block in the
    tail window; otherwise last max_lines lines of that window. Does not tee live output.
    """
    if not log_path.is_file():
        print(f"{header} log file missing: {log_path}", file=sys.stderr, flush=True)
        return
    try:
        raw = _read_tail_bytes(log_path, max_tail_bytes)
    except OSError as e:
        print(f"{header} cannot read log {log_path}: {e}", file=sys.stderr, flush=True)
        return
    text = raw.decode("utf-8", errors="replace")
    idx = text.rfind(TRACE_MARK)
    if idx != -1:
        snippet = text[idx:]
        lines = snippet.splitlines()
        if len(lines) > max_lines:
            snippet = "\n".join(lines[-max_lines:])
    else:
        lines = text.splitlines()
        snippet = "\n".join(lines[-max_lines:])
    print(f"{header} --- log excerpt ({log_path}) ---", file=sys.stderr, flush=True)
    print(snippet, file=sys.stderr, flush=True)
    print(f"{header} --- end log excerpt ---", file=sys.stderr, flush=True)
