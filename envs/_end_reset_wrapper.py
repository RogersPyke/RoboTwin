# Purpose:
#   - Keep seed filtering path (play_once + reset function) unchanged.
#   - In policy eval, if END_RESET_TO_INIT=True, final success requires:
#       task success reached first, then model actions bring robot back to init within step limit.
#   - No hard-coded reset action is executed after policy success in eval mode.

import logging
import math
import os
from datetime import datetime, timedelta, timezone
from types import MethodType

import numpy as np

from .utils import ArmTag

END_RESET_TO_INIT = True

LOG_DIR_NAME = "logs"
UTC8 = timezone(timedelta(hours=8))

# Model reset-to-init tolerance in eval mode.
RESET_TO_INIT_POS_TOL_M = 0.08
RESET_TO_INIT_ROT_TOL_DEG = 25.0
RESET_TO_INIT_GRIPPER_TOL = 0.20


def _timestamp_utc8():
    return datetime.now(UTC8).strftime("%Y%m%d%H%M%S")


def _ensure_logger():
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


def get_end_reset_to_init(config_dict):
    if not isinstance(config_dict, dict):
        return END_RESET_TO_INIT
    v = config_dict.get("END_RESET_TO_INIT", END_RESET_TO_INIT)
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(v)


def _quat_angle_deg(q1, q2):
    q1 = np.array(q1, dtype=np.float64)
    q2 = np.array(q2, dtype=np.float64)
    n1 = np.linalg.norm(q1)
    n2 = np.linalg.norm(q2)
    if n1 < 1e-12 or n2 < 1e-12:
        return 180.0
    q1 = q1 / n1
    q2 = q2 / n2
    dot = float(np.clip(abs(np.dot(q1, q2)), 0.0, 1.0))
    return math.degrees(2.0 * math.acos(dot))


def _capture_eval_init_reference(task_env, logger):
    if getattr(task_env, "_eval_init_ref_ready", False):
        return
    task_env._eval_init_left_pose = np.array(task_env.robot.left_original_pose, dtype=np.float64)
    task_env._eval_init_right_pose = np.array(task_env.robot.right_original_pose, dtype=np.float64)
    task_env._eval_init_left_gripper = float(task_env.robot.get_left_gripper_val())
    task_env._eval_init_right_gripper = float(task_env.robot.get_right_gripper_val())
    task_env._eval_init_ref_ready = True
    logger.info("[end_reset_wrapper] Captured eval init reference pose and gripper.")


def _eval_model_reset_to_init_done(task_env, logger):
    if not getattr(task_env, "_eval_init_ref_ready", False):
        _capture_eval_init_reference(task_env, logger)
    left_now = np.array(task_env.robot.get_left_ee_pose(), dtype=np.float64)
    right_now = np.array(task_env.robot.get_right_ee_pose(), dtype=np.float64)
    left_ref = task_env._eval_init_left_pose
    right_ref = task_env._eval_init_right_pose
    left_pos_err = float(np.linalg.norm(left_now[:3] - left_ref[:3]))
    right_pos_err = float(np.linalg.norm(right_now[:3] - right_ref[:3]))
    left_rot_err = _quat_angle_deg(left_now[3:], left_ref[3:])
    right_rot_err = _quat_angle_deg(right_now[3:], right_ref[3:])
    left_gripper_err = abs(float(task_env.robot.get_left_gripper_val()) - float(task_env._eval_init_left_gripper))
    right_gripper_err = abs(float(task_env.robot.get_right_gripper_val()) - float(task_env._eval_init_right_gripper))

    passed = (
        left_pos_err <= RESET_TO_INIT_POS_TOL_M
        and right_pos_err <= RESET_TO_INIT_POS_TOL_M
        and left_rot_err <= RESET_TO_INIT_ROT_TOL_DEG
        and right_rot_err <= RESET_TO_INIT_ROT_TOL_DEG
        and left_gripper_err <= RESET_TO_INIT_GRIPPER_TOL
        and right_gripper_err <= RESET_TO_INIT_GRIPPER_TOL
    )
    logger.debug(
        (
            "[end_reset_wrapper] Model reset check: pass=%s, "
            "left_pos=%.4f right_pos=%.4f left_rot=%.2f right_rot=%.2f "
            "left_gripper=%.4f right_gripper=%.4f"
        ),
        passed,
        left_pos_err,
        right_pos_err,
        left_rot_err,
        right_rot_err,
        left_gripper_err,
        right_gripper_err,
    )
    return passed


def _do_reset_to_init(task_env, logger):
    if not getattr(task_env, "robot", None):
        if getattr(task_env, "eval_mode", False):
            task_env._eval_reset_completed = False
            task_env.plan_success = False
        logger.warning("[end_reset_wrapper] No robot on task_env; skip expert reset.")
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
        logger.info("[end_reset_wrapper] Expert reset-to-init finished for play_once filtering.")
    except Exception as e:
        logger.error("[end_reset_wrapper] Expert reset-to-init failed: %s", e, exc_info=True)
        if getattr(task_env, "eval_mode", False):
            task_env._eval_reset_completed = False
            task_env.plan_success = False


def _reset_eval_flags(task_env):
    task_env._eval_task_success_reached = False
    task_env._eval_reset_completed = False
    task_env._eval_init_ref_ready = False
    task_env._eval_reset_logged_success = False


def with_end_reset(task_env, end_reset_to_init):
    logger = _ensure_logger()
    original_play_once = task_env.play_once
    original_take_action = task_env.take_action

    def _play_once():
        if getattr(task_env, "eval_mode", False):
            _reset_eval_flags(task_env)
        result = original_play_once()
        if not end_reset_to_init:
            return result
        _do_reset_to_init(task_env, logger)
        return result

    def _take_action(self, action, action_type="qpos"):
        if getattr(self, "eval_mode", False) and getattr(self, "take_action_cnt", 0) == 0:
            _reset_eval_flags(self)
            if end_reset_to_init:
                _capture_eval_init_reference(self, logger)

        original_take_action(action, action_type)

        if not getattr(self, "eval_mode", False):
            return
        if not end_reset_to_init:
            return

        if getattr(self, "eval_success", False) and not getattr(self, "_eval_task_success_reached", False):
            self._eval_task_success_reached = True
            self.eval_success = False
            logger.info("[end_reset_wrapper] Task success reached. Waiting for model reset-to-init.")

        if getattr(self, "_eval_task_success_reached", False):
            self._eval_reset_completed = _eval_model_reset_to_init_done(self, logger)
            self.eval_success = self._eval_reset_completed
            if self.eval_success and not getattr(self, "_eval_reset_logged_success", False):
                self._eval_reset_logged_success = True
                logger.info("[end_reset_wrapper] Eval success: task success + model reset-to-init.")

    def _eval_reset_to_init():
        _do_reset_to_init(task_env, logger)

    task_env.play_once = _play_once
    task_env.take_action = MethodType(_take_action, task_env)
    task_env.eval_reset_to_init = _eval_reset_to_init
    return task_env
