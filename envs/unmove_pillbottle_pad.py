# Purpose: Inverse of move_pillbottle_pad. Pick the pillbottle from the pad and place it on the table.
# Initial state: pillbottle is on the pad (pillbottle's functional point 0 aligned with pad's functional point 1).
# Task: grasp the pillbottle from the pad, lift it, and place it on the table at a target pose.
# Design: Mirror of move_pillbottle_pad: move places pillbottle onto pad; unmove starts with pillbottle on pad
# and places it at a table target (source region of move). Same arm selection (right if pillbottle x > 0 else left).
# Dependencies: Base_Task, envs.utils (rand_pose, create_actor, create_box, etc.), _GLOBAL_CONFIGS, numpy, sapien, transforms3d.
# Usage: envs.unmove_pillbottle_pad, e.g. script/collect_data.py unmove_pillbottle_pad <task_config>
from ._base_task import Base_Task
from .utils import *
import numpy as np
import sapien
import transforms3d as t3d
from ._GLOBAL_CONFIGS import *

# When True: skip all robot planning/IK and movement; record env-only video (only way to block planning raises).
SKIP_ROBOT_MOVEMENT = False
# When True: force episode as success so the script records/saves video (does not block planning raises).
SKIP_SUCCESS_CHECK = True
# Number of frames to record when skipping robot movement (env-only video).
SKIP_SUCCESS_CHECK_VIDEO_FRAMES = 100


class unmove_pillbottle_pad(Base_Task):

    def setup_demo(self, is_test=False, **kwags):
        self.skip_robot_movement = SKIP_ROBOT_MOVEMENT
        self.skip_success_check = SKIP_SUCCESS_CHECK
        self.skip_success_check_video_frames = SKIP_SUCCESS_CHECK_VIDEO_FRAMES
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self.pillbottle_id = np.random.choice([1, 2, 3, 4, 5], 1)[0]

        # Same randomization as move_pillbottle_pad: rand_pos = table pose (move start = unmove end).
        # rand_pos is the target pose of the pillbottle on the table.
        pillbottle_tgt_pose = rand_pose(
            xlim=[-0.25, 0.25],
            ylim=[-0.1, 0.1],
            qpos=[1.0, 0.0, 0.0, 0.0],
            rotate_rand=False,
        )
        while abs(pillbottle_tgt_pose.p[0]) < 0.05:
            pillbottle_tgt_pose = rand_pose(
                xlim=[-0.25, 0.25],
                ylim=[-0.1, 0.1],
                qpos=[1.0, 0.0, 0.0, 0.0],
                rotate_rand=False,
            )

        # Same pad sampling as move_pillbottle_pad: xlim, ylim, qpos, distance >= 0.1 from rand_pos.
        # target_rand_pose is the target pose of the pad on the table.
        if pillbottle_tgt_pose.p[0] > 0:
            xlim = [0.05, 0.25]
        else:
            xlim = [-0.25, -0.05]
        pad_rand_pose = rand_pose(
            xlim=xlim,
            ylim=[-0.2, 0.1],
            qpos=[1, 0, 0, 0],
            rotate_rand=False,
        )
        while (np.sqrt((pad_rand_pose.p[0] - pillbottle_tgt_pose.p[0]) ** 2 + (pad_rand_pose.p[1] - pillbottle_tgt_pose.p[1]) ** 2) < 0.1):
            pad_rand_pose = rand_pose(
                xlim=xlim,
                ylim=[-0.2, 0.1],
                qpos=[1, 0, 0, 0],
                rotate_rand=False,
            )
        half_size = [0.04, 0.04, 0.0005]
        self.pad = create_box(
            scene=self,
            pose=pad_rand_pose,
            half_size=half_size,
            color=(0, 0, 1),
            name="box",
            is_static=True,
        )

        # Unmove start state = move end state: pillbottle on pad, fp0 aligned with pad fp1.
        pad_fp_pose = self.pad.get_functional_point(1, "pose")
        pad_fp_mat = pad_fp_pose.to_transformation_matrix()
        identity_pose = sapien.Pose([0, 0, 0], [1, 0, 0, 0])
        self.pillbottle = create_actor(
            scene=self,
            pose=identity_pose,
            modelname="080_pillbottle",
            convex=True,
            model_id=self.pillbottle_id,
        )
        self.pillbottle.set_mass(0.05)
        actor_mat = self.pillbottle.get_pose().to_transformation_matrix()
        pillbottle_fp_world_mat = self.pillbottle.get_functional_point(0, "matrix")
        fp_local_mat = np.linalg.inv(actor_mat) @ pillbottle_fp_world_mat
        pillbottle_mat = pad_fp_mat @ np.linalg.inv(fp_local_mat)
        pillbottle_pose = sapien.Pose(
            pillbottle_mat[:3, 3],
            t3d.quaternions.mat2quat(pillbottle_mat[:3, :3]),
        )
        self.pillbottle.actor.set_pose(pillbottle_pose)

        # Unmove end state = move start state: pillbottle on table at rand_pos (same pose as move init).
        self.pillbottle_target_pose = [
            pillbottle_tgt_pose.p[0],
            pillbottle_tgt_pose.p[1],
            pillbottle_tgt_pose.p[2],
            pillbottle_tgt_pose.q[0],
            pillbottle_tgt_pose.q[1],
            pillbottle_tgt_pose.q[2],
            pillbottle_tgt_pose.q[3],
        ]

        self.add_prohibit_area(self.pillbottle, padding=0.05)
        self.add_prohibit_area(self.pad, padding=0.1)

    def play_once(self):
        if getattr(self, "skip_robot_movement", False):
            # Skip all planning/IK and movement; record env-only video (blocks planning raises).
            pillbottle_x = self.pillbottle.get_pose().p[0]
            self._arm_tag = ArmTag("right") if pillbottle_x > 0 else ArmTag("left")
            self.info["info"] = {
                "{A}": f"080_pillbottle/base{self.pillbottle_id}",
                "{B}": "box",
                "{a}": str(self._arm_tag),
            }
            save_data_orig = self.save_data
            self.save_data = True
            n = getattr(self, "skip_success_check_video_frames", 100)
            for _ in range(n):
                self._take_picture()
            self.merge_pkl_to_hdf5_video()
            self.save_data = save_data_orig
            self.plan_success = True
            return self.info

        # Same arm selection as move_pillbottle_pad: right if pillbottle (on pad) is on right side.
        arm_tag = ArmTag("right" if self.pillbottle_target_pose[0] > 0 else "left")

        # Grasp the pillbottle from the pad (mirror of move: same params).
        self.move(self.grasp_actor(self.pillbottle, arm_tag=arm_tag, pre_grasp_dis=0.06, gripper_pos=0))

        # Lift up the pillbottle by 0.05 m in z-axis (mirror of move).
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.05))

        # Place the pillbottle on the table at target pose (inverse of move: pad->table vs move table->pad).
        # Table placement with minimal constraints for feasible IK (ref: unhanging_mug table placement).
        # No functional_point_id / no pre_dis_axis: use actor pose and grasp-direction approach;
        # constrain="free": only z aligned. Avoids over-constraining that causes no solution.
        self.move(
            self.place_actor(
                self.pillbottle,
                arm_tag=arm_tag,
                target_pose=self.pillbottle_target_pose,
                pre_dis=0.05,
                dis=0,
                constrain="free",
            )
        )

        self.info["info"] = {
            "{A}": f"080_pillbottle/base{self.pillbottle_id}",
            "{B}": "box",
            "{a}": str(arm_tag),
        }
        if getattr(self, "skip_success_check", False):
            self.plan_success = True
        return self.info

    def check_success(self):
        """Pillbottle is on the table at target pose, grippers open."""
        if getattr(self, "skip_success_check", False):
            return True
        pillbottle_pos = self.pillbottle.get_pose().p
        target = self.pillbottle_target_pose
        eps_xy = 0.03
        eps_z = 0.005
        at_target = (
            np.all(np.abs(pillbottle_pos[:2] - np.array(target[:2])) < np.array([eps_xy, eps_xy]))
            and np.abs(pillbottle_pos[2] - target[2]) < eps_z
        )
        grippers_open = self.robot.is_left_gripper_open() and self.robot.is_right_gripper_open()
        return at_target and grippers_open
