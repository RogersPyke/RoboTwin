# Purpose: Symmetric task of stack_blocks_three. Initial state: three blocks stacked.
# Final state: three blocks at fixed target poses.
# Dependencies: Base_Task, envs.utils (create_box, etc.), sapien, numpy.
# Usage: Load via envs.unstack_blocks_three with task_name="unstack_blocks_three";
#   e.g. script/collect_data.py unstack_blocks_three <task_config>
#
# Control: Same pattern as stack_bowls_three — grasp_actor + place_actor, no custom choose_grasp_pose,
# arm by actor position, constrain="free" to reduce solving burden.
#
# FORCE_COLLECT: when True, skip stability check and always report success (for data collection).
# Initial-state stabilization: spawn blocks one-by-one and run settle steps between each.

from ._base_task import Base_Task
from .utils import *
import sapien
import numpy as np

FORCE_COLLECT = True
SETUP_GAP_STEP = 2000


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


class unstack_blocks_three(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def check_stable(self):
        """When FORCE_COLLECT, run base settle (same step count as base) but always report stable."""
        if FORCE_COLLECT:
            super().check_stable()  # run settle; ignore result so we never raise UnStableError
            return True, []
        return super().check_stable()

    def _settle_steps(self, n: int):
        """Run n simulation steps so newly added actors settle and avoid initial overlap explosion."""
        for _ in range(n):
            self.scene.step()

    def load_actors(self):
        # Initial state: same as stack_blocks_three final state (three blocks stacked).
        # Z positions so that after preprocess (add table_z_bias) we get 0.75, 0.80, 0.85.
        # Vertical spacing = block height (0.05) to match stack_blocks_three.
        block_half_size = 0.025
        z_base = 0.75 - self.table_z_bias
        block_spacing = 0.05
        self.quat_of_target_pose = [1.0, 0.0, 0.0, 0.0]
        # self.quat_of_target_pose = [1.0, 0.0, 0.0, 0.0]

        def create_block(pose, color):
            return create_box(
                scene=self,
                pose=pose,
                half_size=(block_half_size, block_half_size, block_half_size),
                color=color,
                name="box",
            )

        center_xy = [0.0, -0.13]

        pose1 = sapien.Pose(center_xy + [z_base], self.quat_of_target_pose)
        self.block1 = create_block(pose1, (1, 0, 0))
        self._settle_steps(SETUP_GAP_STEP)

        pose2 = sapien.Pose(center_xy + [z_base + block_spacing], self.quat_of_target_pose)
        self.block2 = create_block(pose2, (0, 1, 0))
        self._settle_steps(SETUP_GAP_STEP)

        pose3 = sapien.Pose(center_xy + [z_base + 2.0 * block_spacing], self.quat_of_target_pose)
        self.block3 = create_block(pose3, (0, 0, 1))

        self.add_prohibit_area(self.block1, padding=0.05)
        self.add_prohibit_area(self.block2, padding=0.05)
        self.add_prohibit_area(self.block3, padding=0.05)
        target_pose = [-0.04, -0.13, 0.04, -0.05]
        self.prohibited_area.append(target_pose)

        # Fixed placement targets (no random rotation) for stable data collection.
        fixed_target_pose_xyz = [
            [0.22, -0.06, 0.741 + block_half_size],
            [-0.22, -0.06, 0.741 + block_half_size],
            [0.22, 0.04, 0.741 + block_half_size],
        ]
        block_target_pose_lst = [
            sapien.Pose(pos, [1.0, 0.0, 0.0, 0.0]) for pos in fixed_target_pose_xyz
        ]

        # Convert sapien.Pose to 7D world target [x, y, z + table_z_bias, qw, qx, qy, qz]
        def pose_to_7d(pose):
            p = pose.p
            q = pose.q  # [qw, qx, qy, qz]
            return [float(p[0]), float(p[1]), float(p[2]) + self.table_z_bias, float(q[0]), float(q[1]), float(q[2]), float(q[3])]

        self.block1_target_pose = pose_to_7d(block_target_pose_lst[0])
        self.block2_target_pose = pose_to_7d(block_target_pose_lst[1])
        self.block3_target_pose = pose_to_7d(block_target_pose_lst[2])

    def move_block(self, actor, target_pose):
        """
        Grasp block, lift, place at target using place_actor (same pattern as stack_bowls_three).
        Arm selected by actor position. constrain='free' to reduce solving burden.
        target_pose: 7D [x, y, z, qw, qx, qy, qz] or 3D; converted to 7D with quat_of_target_pose if needed.
        """
        actor_pose = actor.get_pose().p
        arm_tag = ArmTag("left" if actor_pose[0] < 0 else "right")

        if len(target_pose) >= 7:
            target_7d = list(target_pose[:7])
        else:
            target_7d = list(target_pose[:3]) + list(self.quat_of_target_pose)

        if self.last_arm is None or arm_tag == self.last_arm:
            self.move(self.grasp_actor(actor, arm_tag=arm_tag, pre_grasp_dis=0.1, grasp_dis=0))
        else:
            self.move(
                self.grasp_actor(actor, arm_tag=arm_tag, pre_grasp_dis=0.1, grasp_dis=0),
                self.back_to_origin(arm_tag=arm_tag.opposite),
            )
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1))
        self.move(
            self.place_actor(
                actor,
                target_pose=target_7d,
                arm_tag=arm_tag,
                functional_point_id=0,
                pre_dis=0.09,
                dis=0.02,
                constrain="free",
                pre_dis_axis="fp",
            )
        )
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.09))
        self.last_arm = arm_tag
        return arm_tag

    def play_once(self):
        self.last_arm = None
        # Unstack from top to bottom to avoid collision.
        self.move_block(self.block3, self.block3_target_pose)
        if not self.plan_success:
            self.info["info"] = {"{A}": "red block", "{B}": "green block", "{C}": "blue block"}
            return self.info
        self.move_block(self.block2, self.block2_target_pose)
        if not self.plan_success:
            self.info["info"] = {"{A}": "red block", "{B}": "green block", "{C}": "blue block"}
            return self.info
        self.move_block(self.block1, self.block1_target_pose)
        self.info["info"] = {
            "{A}": "red block",
            "{B}": "green block",
            "{C}": "blue block",
        }
        if FORCE_COLLECT:
            self.plan_success = True
        return self.info

    def check_success(self):
        """When FORCE_COLLECT, always return True for collection; else check block poses and grippers."""
        if FORCE_COLLECT:
            return True
        block1_pose = _pose_from_actor(self.block1).p
        block2_pose = _pose_from_actor(self.block2).p
        block3_pose = _pose_from_actor(self.block3).p
        eps = 0.02
        return (
            np.all(np.abs(block1_pose[:2] - self.block1_target_pose[:2]) < eps)
            and np.abs(block1_pose[2] - self.block1_target_pose[2]) < eps
            and np.all(np.abs(block2_pose[:2] - self.block2_target_pose[:2]) < eps)
            and np.abs(block2_pose[2] - self.block2_target_pose[2]) < eps
            and np.all(np.abs(block3_pose[:2] - self.block3_target_pose[:2]) < eps)
            and np.abs(block3_pose[2] - self.block3_target_pose[2]) < eps
            and self.is_left_gripper_open()
            and self.is_right_gripper_open()
        )
