# Purpose: Unhang the mug from the rack and place it on the table.
# Initial state: mug is hanging on the rack (mug's functional point 0 aligned with rack's functional point 0).
# Task: grasp the mug from the rack, lift it off, place at a middle pose on the table, then the other arm
#       grasps from the middle and places at the final target (same two-arm handoff strategy as hanging_mug).
# Design: Mirrors hanging_mug: fixed grasp_arm_tag=left (table side), hang_arm_tag=right (rack side);
#        off-rack move uses place_actor along the rack fp axis; then place at middle_pos; handoff via
#        back_to_origin(hang_arm) + grasp_actor(grasp_arm); grasp_arm places at mug_target_pose.
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
        # Middle pose on table for handoff (same as hanging_mug.middle_pos).
        z_table = 0.75 + getattr(self, "table_z_bias", 0)
        self.middle_pos = [0.0, -0.15, z_table, 1, 0, 0, 0]
        # Final target: mirror hanging_mug's initial mug region (left side), so grasp_arm has a real role.
        # Same xlim/ylim as hanging_mug rand_create_actor for mug; upright quat.
        target_pose = rand_pose(
            xlim=[-0.25, -0.1],
            ylim=[-0.05, 0.05],
            zlim=[z_table, z_table],
            qpos=[1, 0, 0, 0],
            rotate_rand=False,
        )
        self.mug_target_pose = [target_pose.p[0], target_pose.p[1], target_pose.p[2], 1, 0, 0, 0]

    def play_once(self):
        # Same arm roles as hanging_mug: grasp_arm (left) = table side, hang_arm (right) = rack side.
        grasp_arm_tag = ArmTag("left")
        hang_arm_tag = ArmTag("right")
        self._grasp_arm_tag = grasp_arm_tag
        self._hang_arm_tag = hang_arm_tag

        if getattr(self, "skip_robot_movement", False):
            # Skip all planning/IK and movement; record env-only video (blocks planning raises).
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

        # Hang arm: grasp the mug from the rack — use contact points convenient for approach from rack (right) side.
        self.move(self.grasp_actor(
            self.mug,
            arm_tag=hang_arm_tag,
            pre_grasp_dis=0.05,
            # contact_point_id=MUG_CONTACT_RACK_SIDE,
        ))
        # Hang arm: move the mug off the rack along the rack fp axis (reverse of hanging_mug place onto rack).
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
        self.move(self.move_by_displacement(arm_tag=hang_arm_tag, z=0.1, move_axis="world"))

        # Hang arm: place the mug at middle_pos on the table (mirror of hanging_mug: grasp_arm places at middle).
        self.move(
            self.place_actor(
                self.mug,
                arm_tag=hang_arm_tag,
                target_pose=self.middle_pos,
                pre_dis=0.05,
                dis=0.0,
                constrain="free",
            )
        )
        self.move(self.move_by_displacement(arm_tag=hang_arm_tag, z=0.1))

        # Handoff: hang arm back to origin, grasp arm grasps the mug from the middle.
        # Use contact points convenient for left-arm approach (table side), not the same as rack-side grasp.
        self.move(
            self.back_to_origin(hang_arm_tag),
            self.grasp_actor(
                self.mug,
                arm_tag=grasp_arm_tag,
                pre_grasp_dis=0.05,
                # contact_point_id=MUG_CONTACT_TABLE_SIDE,
            ),
        )
        # Lift in z only; do NOT pass quat. move_by_displacement(quat=...) overwrites EE orientation
        # (origin_pose[3:]=quat), forcing the gripper and grasped mug to rotate to that pose.
        # GRASP_DIRECTION_DIC["front"] = [-0.707,0,0,-0.707] (EE "facing forward") would make
        # the mug axis horizontal (parallel to XOY); omit quat to keep mug upright after grasp.
        self.move(self.move_by_displacement(arm_tag=grasp_arm_tag, z=0.1))

        # Grasp arm: place the mug at final target on the table.
        self.move(
            self.place_actor(
                self.mug,
                arm_tag=grasp_arm_tag,
                target_pose=self.mug_target_pose,
                pre_dis=0.05,
                dis=0.0,
                constrain="free",
            )
        )
        self.move(self.move_by_displacement(arm_tag=grasp_arm_tag, z=0.1))

        self.info["info"] = {"{A}": f"039_mug/base{self.mug_id}", "{B}": "040_rack/base0"}
        if getattr(self, "skip_success_check", False):
            self.plan_success = True
        return self.info

    def check_success(self):
        """
        Eval-only relaxed success.
        NOTE:
        - This logic is for EV branch only.
        - For strict data-generation seed filtering, use data branch.
        """
        if getattr(self, "skip_success_check", False):
            return True
        mug_pose = self.mug.get_pose()
        mug_pos = mug_pose.p
        mug_mat = mug_pose.to_transformation_matrix()

        table_z = 0.74 + getattr(self, "table_z_bias", 0.0)
        on_table = (mug_pos[2] >= table_z) and (mug_pos[2] <= table_z + 0.20)

        # Cup mouth up (geometric approximation): mug local +Z should align with world +Z.
        mug_local_z_in_world = mug_mat[:3, 2]
        mouth_up = mug_local_z_in_world[2] > 0.70
        return on_table and mouth_up
