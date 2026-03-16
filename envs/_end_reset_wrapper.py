# Purpose: Configurable wrapper so that (1) collected data contains full trajectory
#          init -> play_once -> reset to init; (2) evaluation requires task success AND reset to init.
# Dependencies: envs.utils (ArmTag), envs._base_task (task env with move, back_to_origin, robot).
# Usage:
#   - Data collection (eval_mode=False): trajectory = init -> play_once -> reset (full, for time reversal).
#   - Evaluation (eval_mode=True): after task success, run reset; eval success = check_success() and reset done.
#   - Default: FORCE_END_RESET_TO_INIT = True. Override via task_config yaml "force_end_reset_to_init".
#   - Applied automatically in Base_Task._init_task_env_() so collect_data/eval_policy/eval_policy_client
#     need no changes; config comes from setup_demo(**args).

import os
import logging
from datetime import datetime, timezone, timedelta

from .utils import ArmTag

# Default: force robot back to init state at end of each episode (for time-reversal compatibility).
# Overridable by task_config/<name>.yml key "force_end_reset_to_init".
FORCE_END_RESET_TO_INIT = True

# Log directory under project root; timestamp YYYYMMDDHHMMSS (UTC+8).
LOG_DIR_NAME = "logs"
UTC8 = timezone(timedelta(hours=8))


def _timestamp_utc8():
    """Return current timestamp string YYYYMMDDHHMMSS in UTC+8."""
    return datetime.now(UTC8).strftime("%Y%m%d%H%M%S")


def _ensure_logger():
    """Create or return module logger; logs to dedicated log dir with script-named file."""
    name = "end_reset_wrapper"
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.path.join(root, LOG_DIR_NAME)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{name}_{_timestamp_utc8()}.log")
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger


def get_force_end_reset_to_init(config_dict):
    """
    Resolve whether to force end-of-episode reset to init state.
    Input: config_dict (dict) - typically the loaded task_config yaml plus runtime args.
    Output: bool - True to force reset at end, False to skip.
    Usage: Override default FORCE_END_RESET_TO_INIT when config_dict contains
           key "force_end_reset_to_init"; otherwise use module default.
    """
    if not isinstance(config_dict, dict):
        return FORCE_END_RESET_TO_INIT
    return config_dict.get("force_end_reset_to_init", FORCE_END_RESET_TO_INIT)


def _do_reset_to_init(task_env, logger):
    """
    Move both arms to init pose; set task_env._eval_reset_completed when in eval_mode.
    Input: task_env - Base_Task with move(), back_to_origin(), robot; logger for messages.
    Output: None. Sets task_env._eval_reset_completed in eval_mode from task_env.plan_success.
    """
    if not getattr(task_env, "robot", None):
        if getattr(task_env, "eval_mode", False):
            task_env._eval_reset_completed = False
            task_env.plan_success = False
        logger.warning("\033[91m[end_reset_wrapper] No robot on task_env; skip end reset.\033[0m")
        return
    try:
        task_env.move(
            task_env.back_to_origin(ArmTag("left")),
            task_env.back_to_origin(ArmTag("right")),
        )
        if getattr(task_env, "eval_mode", False):
            task_env._eval_reset_completed = getattr(task_env, "plan_success", False)
            if not task_env._eval_reset_completed:
                task_env.plan_success = False
        logger.info("\033[92m[end_reset_wrapper] SUCCESS: robot reset to init state at episode end.\033[0m")
    except Exception as e:
        logger.error("\033[91m[end_reset_wrapper] ERR: end reset failed: %s\033[0m", e, exc_info=True)
        if getattr(task_env, "eval_mode", False):
            task_env._eval_reset_completed = False
            task_env.plan_success = False


def with_end_reset(task_env, force_end_reset_to_init):
    """
    Wrap task_env so: (1) play_once() runs task then reset when enabled (data + eval expert path).
    (2) Eval success = task success AND reset completed; take_action is wrapped so eval_success is
        set only after reset in eval_mode (no changes needed in eval scripts).
    Input: task_env - Base_Task with move(), back_to_origin(), robot, eval_mode, take_action.
           force_end_reset_to_init (bool) - if True, after play_once() run both arms to origin.
    Output: task_env (play_once and take_action patched; eval_reset_to_init attached for optional use).
    """
    logger = _ensure_logger()
    original_play_once = task_env.play_once
    original_take_action = task_env.take_action

    def _play_once():
        if getattr(task_env, "eval_mode", False):
            task_env._eval_reset_completed = False
        result = original_play_once()
        if not force_end_reset_to_init:
            return result
        _do_reset_to_init(task_env, logger)
        return result

    def _take_action(self, action, action_type="qpos"):
        if getattr(self, "eval_mode", False) and getattr(self, "take_action_cnt", 0) == 0:
            self._eval_reset_completed = False
        original_take_action(action, action_type)
        if getattr(self, "eval_mode", False) and getattr(self, "eval_success", False):
            _do_reset_to_init(self, logger)
            self.eval_success = getattr(self, "_eval_reset_completed", False)

    def _eval_reset_to_init():
        """Optional: call from eval script to run reset; wrapper already does this inside take_action in eval_mode."""
        _do_reset_to_init(task_env, logger)

    task_env.play_once = _play_once
    task_env.take_action = _take_action
    task_env.eval_reset_to_init = _eval_reset_to_init
    return task_env
