# global configs
import os

ROOT_PATH = os.path.abspath(__file__)
ROOT_PATH = ROOT_PATH[:ROOT_PATH.rfind("/")]
ROOT_PATH = ROOT_PATH[:ROOT_PATH.rfind("/") + 1]

ASSETS_PATH = os.path.join(ROOT_PATH, "assets/")
EMBODIMENTS_PATH = os.path.join(ASSETS_PATH, "embodiments/")
TEXTURES_PATH = os.path.join(ASSETS_PATH, "background_texture/")
CONFIGS_PATH = os.path.join(ROOT_PATH, "task_config/")
SCRIPT_PATH = os.path.join(ROOT_PATH, "script/")
DESCRIPTION_PATH = os.path.join(ROOT_PATH, "description/")

# Euler angles in world coordinates
# t3d.euler.quat2euler(quat) returns (theta_x, theta_y, theta_z)
# theta_y controls the pitch, and theta_z controls rotation around the axis perpendicular to the tabletop plane
GRASP_DIRECTION_DIC = {
    "left": [0, 0, 0, -1],
    "front_left": [-0.383, 0, 0, -0.924],
    "front": [-0.707, 0, 0, -0.707],
    "front_right": [-0.924, 0, 0, -0.383],
    "right": [-1, 0, 0, 0],
    "top_down": [-0.5, 0.5, -0.5, -0.5],
    "down_right": [-0.707, 0, -0.707, 0],
    "down_left": [0, 0.707, 0, -0.707],
    "top_down_little_left": [-0.353523, 0.61239, -0.353524, -0.61239],
    "top_down_little_right": [-0.61239, 0.353523, -0.61239, -0.353524],
    "left_arm_perf": [-0.853532, 0.146484, -0.353542, -0.3536],
    "right_arm_perf": [-0.353518, 0.353564, -0.14642, -0.853568],
}

WORLD_DIRECTION_DIC = {
    "left": [0, -0.707, 0, 0.707],  # -z  -y  -x
    "front": [0.5, -0.5, 0.5, 0.5],  # y   z   -x
    "right": [0.707, 0, 0.707, 0],  # z   y   -x
    "top_down": [0, 0.707, -0.707, 0],  # -x  -y  -z
}

ROTATE_NUM = 10


def is_left_task(task_name: str) -> bool:
    """Whether the task's env module lives under envs/left/.

    Left-arm task families (and their camera variants) are kept in the
    ``envs/left/`` namespace so they stay separate from the original right-arm
    tasks.  CLI task names are bare (e.g. ``blocks_ranking_rgb_left``); this
    helper maps that bare name to the ``left`` namespace by file existence.
    """
    left_root = os.path.join(ROOT_PATH, "envs", "left")
    # The leading "" keeps standalone (non-wrapper) tasks such as
    # place_object_scale_left resolvable directly under envs/left/.
    return any(os.path.exists(os.path.join(left_root, folder, f"{task_name}.py"))
               for folder in ("", "base", "impl", "central_wide_cam", "left_oppo_cam",
                              "cen_arm_right_wide_cam", "cen_arm_near_side_cam",
                              "cen_arm_side_cam", "cen_arm_front_cam", "cen_arm_top_cam"))


def import_task_env(task_name: str):
    """Import and return the env module for a task, resolving envs/left/ first.

    Falls back to ``envs.{task_name}`` for original (non-left) tasks so a
    single bare task name works for both namespaces.
    """
    import importlib
    if is_left_task(task_name):
        left_root = os.path.join(ROOT_PATH, "envs", "left")
        for folder in ("cen_arm_right_wide_cam", "cen_arm_near_side_cam",
                       "cen_arm_side_cam", "cen_arm_front_cam", "cen_arm_top_cam",
                       "central_wide_cam", "left_oppo_cam",
                       "base", "impl"):
            if os.path.exists(os.path.join(left_root, folder, f"{task_name}.py")):
                return importlib.import_module(f"envs.left.{folder}.{task_name}")
        return importlib.import_module(f"envs.left.{task_name}")
    return importlib.import_module(f"envs.{task_name}")


def task_config_yml_path(task_config: str) -> str:
    """Path to the existing task config yml, preferring task_config/left/.

    Left-arm task configs are organized like ``envs/left`` — one subfolder per
    camera variant (``base`` holds the default central-cam configs, and utility
    tasks like ``place_object_scale_left`` sit at the left/ root) — while
    original configs stay in ``task_config/``.  The bare config name is
    resolved against every candidate and the first existing file wins, so the
    CLI argument is unchanged.  Keep the folder list in sync with
    ``import_task_env``.
    """
    left_root = os.path.join(ROOT_PATH, "task_config", "left")
    for folder in ("cen_arm_right_wide_cam", "cen_arm_near_side_cam",
                   "cen_arm_side_cam", "cen_arm_front_cam", "cen_arm_top_cam",
                   "central_wide_cam", "left_oppo_cam",
                   "base", ""):
        p = os.path.join(left_root, folder, f"{task_config}.yml")
        if os.path.exists(p):
            return p
    return os.path.join(ROOT_PATH, "task_config", f"{task_config}.yml")
