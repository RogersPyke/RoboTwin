"""
DEPRECATED: This module is no longer used.

END_RESET_TO_INIT is now handled directly in _base_task.py:
    - _init_task_env_() loads END_RESET_TO_INIT from kwargs
    - _execute_end_reset() handles the reset logic
    - take_action() calls _execute_end_reset() after check_success()

This file is kept for backward compatibility but does nothing.
"""


def get_end_reset_to_init(kwargs):
    """DEPRECATED: Use kwargs.get('END_RESET_TO_INIT', False) directly."""
    return bool(kwargs.get("END_RESET_TO_INIT", False))


def with_end_reset(task_env, end_reset_to_init):
    """DEPRECATED: END_RESET_TO_INIT is now handled in _base_task._init_task_env_()."""
    pass
