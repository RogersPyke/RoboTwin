"""
Compatibility helpers for END_RESET_TO_INIT option.

This module provides a minimal implementation to satisfy imports from
envs._base_task while preserving current behavior.
"""


def get_end_reset_to_init(kwargs):
    """Read END_RESET_TO_INIT from kwargs with a safe default."""
    return bool(kwargs.get("END_RESET_TO_INIT", True))


def with_end_reset(task_env, end_reset_to_init):
    """Attach END_RESET_TO_INIT flag to task env.

    Current compatibility behavior is a no-op wrapper that stores the flag.
    """
    task_env._end_reset_to_init = bool(end_reset_to_init)
    return task_env
