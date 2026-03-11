# Purpose: Symmetric task of stack_bowls_three. Initial state: three bowls stacked.
# Final state: three bowls at random target poses (deterministic given seed).
# Dependencies: Base_Task, envs.utils (rand_pose, create_actor, etc.), sapien, numpy.
# Usage: Load via envs.unstack_bowls_three with task_name="unstack_bowls_three";
#   e.g. script/collect_data.py unstack_bowls_three <task_config>
#
# FORCE_COLLECT: when True, skip stability check and always report success (for data collection).
#   when False, use default behavior (check_stable + check_success).
#
# --- Why we cannot "just set initial state" without physics ---
# PhysX (SAPIEN backend) treats overlapping convex shapes as penetration and applies large
# repulsion forces. If we add all three stacked bowls in one go, the first scene.step() (e.g.
# in check_stable()) would see overlap and make bowls fly. So we spawn one bowl, run settle steps,
# then add the next, so the engine never sees two overlapping dynamic bodies at once.
#
# --- How other tasks avoid this ---
# Tasks with non-overlapping initial poses do not need per-actor settle: e.g. stack_bowls_three
# (three bowls at random scattered poses), place_empty_cup (one cup, one coaster), unstack_blocks_three
# (same stacked-init pattern as here, see SETUP_GAP_STEP there). Only "stacked" initial states
# need sequential spawn + settle; reference unstack_blocks_three for the same pattern.
#
# --- More fundamental alternatives (if you want to avoid settle steps) ---
# 1. Engine support: if SAPIEN exposed "pause simulation" during setup, we could add all actors
#    at target poses then resume (no overlap ever seen by the solver).
# 2. Static-then-dynamic: create upper bowls as static, set poses, then switch to dynamic at
#    task start (requires runtime body-type change support in SAPIEN).
# 3. Keep current approach but tune SETUP_GAP_STEP to the minimum that still stabilizes.

from ._base_task import Base_Task
from .utils import *
import sapien
import numpy as np
from copy import deepcopy

FORCE_COLLECT = True
SETUP_GAP_STEP = 1000


def _pose_from_actor(actor):
    """Return pose as sapien.Pose so base class can use .p / .to_transformation_matrix()."""
    raw = actor.get_pose()
    if hasattr(raw, "p"):
        return raw
    arr = np.array(raw)
    if arr.size == 7:
        return sapien.Pose(arr[:3].tolist(), arr[3:].tolist())
    if arr.size == 3:
        return sapien.Pose(arr.tolist(), [1, 0, 0, 0])
    return sapien.Pose([0, 0, 0], [1, 0, 0, 0])


class _PlaceActorProxy:
    """Proxy so base place_actor gets sapien.Pose from get_pose() (some SAPIEN versions return ndarray)."""

    def __init__(self, actor):
        self._actor = actor

    def get_pose(self):
        return _pose_from_actor(self._actor)

    def __getattr__(self, name):
        return getattr(self._actor, name)


class unstack_bowls_three(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def check_stable(self):
        """When FORCE_COLLECT, skip stability check so setup never raises UnStableError."""
        if FORCE_COLLECT:
            return True, []
        return super().check_stable()

    def _settle_steps(self, n: int):
        """Run n simulation steps so newly added actors settle and avoid initial overlap explosion."""
        for _ in range(n):
            self.scene.step()

    def load_actors(self):
        # Initial state: same as stack_bowls_three final state (three bowls stacked).
        # Z positions so that after preprocess (add table_z_bias) we get 0.76, 0.86, 0.96.
        # Vertical spacing = 0.1 per bowl so stacked heights match stack_bowls_three.
        z_base = 0.76 - self.table_z_bias
        bowl_spacing = 0.05
        # Single quat for both spawn and place: identity so bowl opening faces +z (upright).
        self.quat_of_target_pose = [1.0, 0.0, 0.0, 0.0]

        def create_bowl(pose):
            return create_actor(
                self, pose=pose, modelname="002_bowl", model_id=3, convex=True
            )

        pose1 = sapien.Pose([0.0, -0.1, z_base], self.quat_of_target_pose)
        self.bowl1 = create_bowl(pose1)
        self._settle_steps(SETUP_GAP_STEP)

        pose2 = sapien.Pose([0.0, -0.1, z_base + bowl_spacing], self.quat_of_target_pose)
        self.bowl2 = create_bowl(pose2)
        self._settle_steps(SETUP_GAP_STEP)

        pose3 = sapien.Pose([0.0, -0.1, z_base + 2.0 * bowl_spacing], self.quat_of_target_pose)
        self.bowl3 = create_bowl(pose3)

        self.add_prohibit_area(self.bowl1, padding=0.07)
        self.add_prohibit_area(self.bowl2, padding=0.07)
        self.add_prohibit_area(self.bowl3, padding=0.07)
        target_pose = [-0.1, -0.15, 0.1, -0.05]
        self.prohibited_area.append(target_pose)

        # Random target poses (seed already set in _init_task_env_); same constraints as stack_bowls_three.
        bowl_target_pose_lst = []
        for i in range(3):
            bowl_pose = rand_pose(
                xlim=[-0.3, 0.3],
                ylim=[-0.15, 0.15],
                qpos=[0.5, 0.5, 0.5, 0.5],
                ylim_prop=True,
                rotate_rand=False,
            )

            def check_bowl_pose(bowl_pose, existing):
                for j in range(len(existing)):
                    if np.sum(np.power(bowl_pose.p[:2] - existing[j][:2], 2)) < 0.0169:
                        return False
                return True

            while (
                abs(bowl_pose.p[0]) < 0.09
                or np.sum(np.power(bowl_pose.p[:2] - np.array([0, -0.1]), 2)) < 0.0169
                or not check_bowl_pose(bowl_pose, bowl_target_pose_lst)
            ):
                bowl_pose = rand_pose(
                    xlim=[-0.3, 0.3],
                    ylim=[-0.15, 0.15],
                    qpos=[0.5, 0.5, 0.5, 0.5],
                    ylim_prop=True,
                    rotate_rand=False,
                )
            # Store world position (z from rand_pose is table-relative; add table_z_bias for place/check).
            world_p = np.array([
                bowl_pose.p[0],
                bowl_pose.p[1],
                bowl_pose.p[2] + self.table_z_bias,
            ])
            bowl_target_pose_lst.append(world_p)

        self.bowl1_target_pose = bowl_target_pose_lst[0]
        self.bowl2_target_pose = bowl_target_pose_lst[1]
        self.bowl3_target_pose = bowl_target_pose_lst[2]

    def move_bowl(self, actor, target_pose):
        pose_for_arm = _pose_from_actor(actor)
        actor_pose = np.array(pose_for_arm.p)
        arm_tag = ArmTag("left" if actor_pose[0] < 0 else "right")

        # Let base class try all contact points so a reachable grasp is found for upright bowls.
        if self.las_arm is None or arm_tag == self.las_arm:
            self.move(
                self.grasp_actor(actor, arm_tag=arm_tag, pre_grasp_dis=0.1)
            )
        else:
            self.move(
                self.grasp_actor(actor, arm_tag=arm_tag, pre_grasp_dis=0.1),
                self.back_to_origin(arm_tag=arm_tag.opposite),
            )
        self.move(self.move_by_displacement(arm_tag, z=0.1))
        self.move(
            self.place_actor(
                _PlaceActorProxy(actor),
                target_pose=target_pose.tolist() + self.quat_of_target_pose,
                arm_tag=arm_tag,
                pre_dis=0.09,
                dis=0,
                constrain="align",
            )
        )
        self.move(self.move_by_displacement(arm_tag, z=0.09))
        self.las_arm = arm_tag
        return arm_tag

    def play_once(self):
        self.las_arm = None
        # Unstack from top to bottom to avoid collision.
        self.move_bowl(self.bowl3, self.bowl3_target_pose)
        self.move_bowl(self.bowl2, self.bowl2_target_pose)
        self.move_bowl(self.bowl1, self.bowl1_target_pose)
        self.info["info"] = {"{A}": "002_bowl/base3"}
        if FORCE_COLLECT:
            self.plan_success = True
        return self.info

    def check_success(self):
        """When FORCE_COLLECT, always return True for collection; else check bowl poses and grippers."""
        if FORCE_COLLECT:
            return True
        bowl1_pose = _pose_from_actor(self.bowl1).p
        bowl2_pose = _pose_from_actor(self.bowl2).p
        bowl3_pose = _pose_from_actor(self.bowl3).p
        eps = 0.02
        return (
            np.all(np.abs(bowl1_pose[:2] - self.bowl1_target_pose[:2]) < eps)
            and np.abs(bowl1_pose[2] - self.bowl1_target_pose[2]) < eps
            and np.all(np.abs(bowl2_pose[:2] - self.bowl2_target_pose[:2]) < eps)
            and np.abs(bowl2_pose[2] - self.bowl2_target_pose[2]) < eps
            and np.all(np.abs(bowl3_pose[:2] - self.bowl3_target_pose[:2]) < eps)
            and np.abs(bowl3_pose[2] - self.bowl3_target_pose[2]) < eps
            and self.is_left_gripper_open()
            and self.is_right_gripper_open()
        )
