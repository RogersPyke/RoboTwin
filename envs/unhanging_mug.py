# Purpose: Unhang the mug from the rack and place it on the table.
# Initial state: mug is hanging on the rack (mug's functional point 0 aligned with rack's functional point 0).
# Task: grasp the mug from the rack, lift it off, and place it on the table at a target pose.
# Design: Off-rack move uses place_actor with motion constrained along the rack functional-point axis.
# This is the reverse of hanging_mug (place onto rack along fp axis); constrained motion along the fp
# axis yields feasible IK and avoids collision with the rack edge.
# Dependencies: Base_Task, envs.utils (rand_pose, create_actor, etc.), _GLOBAL_CONFIGS, numpy.
# Usage: envs.unhanging_mug, e.g. script/collect_data.py unhanging_mug <task_config>
from ._base_task import Base_Task
from .utils import *
import numpy as np
import sapien
import transforms3d as t3d
from ._GLOBAL_CONFIGS import *

# Physics steps after placing rack so scene settles before reading rack fp for mug placement.
SETTLE_STEP = 3500
# Small +z offset when placing mug on rack to avoid spawn collision.
MUG_HANG_Z_OFFSET = 0.02

# When True: skip all robot planning/IK and movement; record env-only video (only way to block planning raises).
SKIP_ROBOT_MOVEMENT = False
# When True: force episode as success so the script records/saves video (does not block planning raises).
SKIP_SUCCESS_CHECK = False
# Number of frames to record when skipping robot movement (env-only video).
SKIP_SUCCESS_CHECK_VIDEO_FRAMES = 100
# Distance along rack functional-point axis to move the mug off the rack (meters).
# Off-rack motion is constrained along this axis (reverse of hanging_mug place_actor).
OFF_RACK_DIST = 0.05


class unhanging_mug(Base_Task):

    def setup_demo(self, is_test=False, **kwags):
        self.skip_robot_movement = SKIP_ROBOT_MOVEMENT
        self.skip_success_check = SKIP_SUCCESS_CHECK
        self.skip_success_check_video_frames = SKIP_SUCCESS_CHECK_VIDEO_FRAMES
        super()._init_task_env_(**kwags)

    def _get_off_rack_place_pose(self, along_fp_dist: float) -> list:
        """
        Target pose for moving the mug off the rack along the rack functional-point axis.
        This is the reverse of hanging_mug: hanging uses place_actor with pre_dis_axis='fp'
        to approach along the fp axis; we move the mug in the opposite direction by
        placing at (rack_fp_pos - along_fp_dist * rack_fp_z). Constrained motion along
        the fp axis yields feasible IK because it mirrors the hang motion.
        Returns: 7-dim list [x, y, z, qw, qx, qy, qz] for place_actor target_pose.
        """
        rack_fp_pose = self.rack.get_functional_point(0, "pose")
        rack_fp_mat = rack_fp_pose.to_transformation_matrix()
        rack_fp_pos = np.array(rack_fp_mat[:3, 3], dtype=np.float64)
        rack_fp_z = np.array(rack_fp_mat[:3, 2], dtype=np.float64)
        rack_fp_z = rack_fp_z / np.linalg.norm(rack_fp_z)
        off_pos = rack_fp_pos - along_fp_dist * rack_fp_z
        q = t3d.quaternions.mat2quat(rack_fp_mat[:3, :3])
        return off_pos.tolist() + q.tolist()

    def load_actors(self):
        self.mug_id = np.random.choice([i for i in range(10)])
        # Rack: same position and pose randomization as hanging_mug (rand_pose default zlim=[0.741]).
        rack_pose = rand_pose(
            xlim=[0.1, 0.3],
            ylim=[0.13, 0.17],
            rotate_rand=True,
            rotate_lim=[0, 0.2, 0],
            qpos=[-0.22, -0.22, 0.67, 0.67],
        )
        self.rack = create_actor(self, pose=rack_pose, modelname="040_rack", is_static=True, convex=True)
        for _ in range(SETTLE_STEP):
            self.scene.step()

        # Mug: place so its functional point 0 aligns with rack's actual fp (after settle), slightly higher to avoid collision.
        # Use ACTUAL rack pose/functional point from the scene, not the initial set pose.
        rack_fp_pose = self.rack.get_functional_point(0, "pose")
        rack_fp_mat = rack_fp_pose.to_transformation_matrix()
        rack_fp_mat[2, 3] += MUG_HANG_Z_OFFSET  # target position for mug fp: actual rack fp + z offset

        mug_identity_pose = sapien.Pose([0, 0, 0], [1, 0, 0, 0])
        self.mug = create_actor(
            self,
            pose=mug_identity_pose,
            modelname="039_mug",
            convex=True,
            model_id=self.mug_id,
        )
        # fp in world = actor_pose @ fp_local => fp_local = inv(actor_pose) @ fp_world.
        actor_mat = self.mug.get_pose().to_transformation_matrix()
        mug_fp_world_mat = self.mug.get_functional_point(0, "matrix")
        fp_local_mat = np.linalg.inv(actor_mat) @ mug_fp_world_mat
        mug_mat = rack_fp_mat @ np.linalg.inv(fp_local_mat)
        mug_pose = sapien.Pose(mug_mat[:3, 3], t3d.quaternions.mat2quat(mug_mat[:3, :3]))
        self.mug.actor.set_pose(mug_pose)

        self.add_prohibit_area(self.mug, padding=0.1)
        self.add_prohibit_area(self.rack, padding=0.1)
        # Target pose on table for placing the mug (same xy range as hanging_mug middle_pos).
        z_table = 0.75 + getattr(self, "table_z_bias", 0)
        self.mug_target_pose = [0.0, -0.15, z_table, 1, 0, 0, 0]

    def play_once(self):
        if getattr(self, "skip_robot_movement", False):
            # Skip all planning/IK and movement; record env-only video (blocks planning raises).
            mug_x = self.mug.get_pose().p[0]
            self._hang_arm_tag = ArmTag("right") if mug_x > 0 else ArmTag("left")
            self.info["info"] = {"{A}": f"039_mug/base{self.mug_id}", "{B}": "040_rack/base0"}
            save_data_orig = self.save_data
            self.save_data = True
            n = getattr(self, "skip_success_check_video_frames", 100)
            for _ in range(n):
                self._take_picture()
            self.merge_pkl_to_hdf5_video()
            self.save_data = save_data_orig
            self.plan_success = True
            return self.info

        # Grasp the mug from the rack: right arm if mug x > 0 else left (front approach).
        mug_x = self.mug.get_pose().p[0]
        hang_arm_tag = ArmTag("right") if mug_x > 0 else ArmTag("left")
        self._hang_arm_tag = hang_arm_tag
        # Grasp mug on rack, then lift off.
        # contact_point_id: restrict to mug handle/body points; None = try all from model_data.
        # If the robot never moves, choose_grasp_pose likely returned (None, None) (no valid grasp).
        # Try [0, 1, 2, 3] (common for mugs/blocks) or inspect with _print_all_grasp_pose_of_contact_points(self.mug).
        self.move(self.grasp_actor(self.mug,
            arm_tag=hang_arm_tag,
            pre_grasp_dis=0.05,
            contact_point_id=[0,1,2,3,4,5],
        ))
        # Move the mug off the rack by constrained motion along the rack functional-point axis.
        # Reason: move_by_displacement in an arbitrary direction often had no IK solution after grasp.
        # Fix: use place_actor to a pose along the fp axis (reverse of hanging_mug). Success is because
        # the motion is constrained along the functional-point axis and is the exact reverse of the
        # placing (hanging) action, so the same motion pattern that works for hang works for unhang.
        off_rack_target = self._get_off_rack_place_pose(OFF_RACK_DIST)
        self.move(
            self.place_actor(
                self.mug,
                arm_tag=hang_arm_tag,
                target_pose=off_rack_target,
                functional_point_id=0,
                constrain="align",
                pre_dis=0.0,
                dis=0.0,
                is_open=False,
                pre_dis_axis="fp",
            )
        )
        # Move the eef : lift to avoid collision with table.
        self.move(self.move_by_displacement(arm_tag=hang_arm_tag, z=0.1, move_axis="world"))
        # Place the mug on the table at target pose.
        # Mirror hanging_mug's middle_pos placement: no functional_point_id, no pre_dis_axis,
        # so approach uses default "grasp" axis and placement uses actor pose (z_transform=True).
        # This avoids over-constraining and yields feasible IK like the inverse task.
        self.move(
            self.place_actor(
                self.mug,
                arm_tag=hang_arm_tag,
                target_pose=self.mug_target_pose,
                pre_dis=0.05,
                dis=0.0,
                constrain="free",
            )
        )
        self.move(self.move_by_displacement(arm_tag=hang_arm_tag, z=0.1, move_axis="world"))
        self.info["info"] = {"{A}": f"039_mug/base{self.mug_id}", "{B}": "040_rack/base0"}
        if getattr(self, "skip_success_check", False):
            self.plan_success = True
        return self.info

    def check_success(self):
        """Mug is on the table (not on rack), gripper is open."""
        if getattr(self, "skip_success_check", False):
            return True
        mug_pos = self.mug.get_pose().p
        target_z = self.mug_target_pose[2]
        eps_xy = 0.02
        eps_z = 0.02
        on_table = (
            np.all(np.abs(mug_pos[:2] - np.array(self.mug_target_pose[:2])) < eps_xy)
            and np.abs(mug_pos[2] - target_z) < eps_z
        )
        gripper_open = self.is_right_gripper_open() if getattr(self, "_hang_arm_tag", ArmTag("right")) == ArmTag("right") else self.is_left_gripper_open()
        return on_table and gripper_open
