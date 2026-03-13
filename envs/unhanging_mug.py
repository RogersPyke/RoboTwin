# Purpose: Unhang the mug from the rack and place it on the table.
# Initial state: mug is hanging on the rack (mug's functional point 0 aligned with rack's functional point 0).
# Task: grasp the mug from the rack, lift it off, and place it on the table at a target pose.
# Dependencies: Base_Task, envs.utils (rand_pose, create_actor, etc.), _GLOBAL_CONFIGS, numpy.
# Usage: envs.unhanging_mug, e.g. script/collect_data.py unhanging_mug <task_config>
from ._base_task import Base_Task
from .utils import *
import numpy as np
import sapien
import transforms3d as t3d
from ._GLOBAL_CONFIGS import *


class unhanging_mug(Base_Task):

    def setup_demo(self, is_test=False, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self.mug_id = np.random.choice([i for i in range(10)])
        # Create rack first (same as hanging_mug).
        rack_pose = rand_pose(
            xlim=[0.1, 0.3],
            ylim=[0.13, 0.17],
            rotate_rand=True,
            rotate_lim=[0, 0.2, 0],
            qpos=[-0.22, -0.22, 0.67, 0.67],
        )
        self.rack = create_actor(self, pose=rack_pose, modelname="040_rack", is_static=True, convex=True)

        # Create mug at identity pose (preprocess will add table_z_bias to z).
        mug_identity_pose = sapien.Pose([0, 0, 0], [1, 0, 0, 0])
        self.mug = create_actor(
            self,
            pose=mug_identity_pose,
            modelname="039_mug",
            convex=True,
            model_id=self.mug_id,
        )
        # Compute mug pose so that mug's functional point 0 is at rack's functional point 0 (hanging state).
        # fp in world = actor_pose @ fp_local => fp_local = inv(actor_pose) @ fp_world.
        actor_mat = self.mug.get_pose().to_transformation_matrix()
        mug_fp_world_mat = self.mug.get_functional_point(0, "matrix")
        fp_local_mat = np.linalg.inv(actor_mat) @ mug_fp_world_mat
        rack_fp_pose = self.rack.get_functional_point(0, "pose")
        rack_fp_mat = rack_fp_pose.to_transformation_matrix()
        mug_mat = rack_fp_mat @ np.linalg.inv(fp_local_mat)
        mug_pose = sapien.Pose(mug_mat[:3, 3], t3d.quaternions.mat2quat(mug_mat[:3, :3]))
        self.mug.actor.set_pose(mug_pose)

        self.add_prohibit_area(self.mug, padding=0.1)
        self.add_prohibit_area(self.rack, padding=0.1)
        # Target pose on table for placing the mug (same xy range as hanging_mug middle_pos).
        z_table = 0.75 + getattr(self, "table_z_bias", 0)
        self.mug_target_pose = [0.0, -0.15, z_table, 1, 0, 0, 0]

    def play_once(self):
        # Grasp the mug from the rack with the right arm (front approach, same as hanging arm in hanging_mug).
        hang_arm_tag = ArmTag("right")
        # Grasp mug on rack, then lift off.
        self.move(self.grasp_actor(self.mug, arm_tag=hang_arm_tag, pre_grasp_dis=0.05))
        self.move(self.move_by_displacement(arm_tag=hang_arm_tag, z=0.1, quat=GRASP_DIRECTION_DIC["front"]))
        # Place the mug on the table at target pose.
        self.move(
            self.place_actor(
                self.mug,
                arm_tag=hang_arm_tag,
                target_pose=self.mug_target_pose,
                functional_point_id=0,
                pre_dis=0.05,
                dis=0.0,
                constrain="free",
                pre_dis_axis="fp",
            )
        )
        self.move(self.move_by_displacement(arm_tag=hang_arm_tag, z=0.1, move_axis="arm"))
        self.info["info"] = {"{A}": f"039_mug/base{self.mug_id}", "{B}": "040_rack/base0"}
        return self.info

    def check_success(self):
        """Mug is on the table (not on rack), gripper is open."""
        mug_pos = self.mug.get_pose().p
        target_z = self.mug_target_pose[2]
        eps_xy = 0.02
        eps_z = 0.02
        on_table = (
            np.all(np.abs(mug_pos[:2] - np.array(self.mug_target_pose[:2])) < eps_xy)
            and np.abs(mug_pos[2] - target_z) < eps_z
        )
        return on_table and self.is_right_gripper_open()
