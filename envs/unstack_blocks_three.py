"""
Minimal test: ONE block on table, place to ONE fixed target.
Aligned with stack_blocks_three so grasp/place behave the same as working tasks.
"""
from ._base_task import Base_Task
from .utils import *
import sapien
import math


class unstack_blocks_three(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        block_half_size = 0.025
        z_table = 0.741 + block_half_size  # block center on table

        def create_block(block_pose, color):
            return create_box(
                scene=self,
                pose=block_pose,
                half_size=(block_half_size, block_half_size, block_half_size),
                color=color,
                name="box",
            )

        # Simplest: ONE block on table at center (no stack).
        center_xy = [0.0, -0.13]
        z_base = 0.75 - block_half_size
        self.block1 = create_block(sapien.Pose(center_xy + [z_base], [1, 0, 0, 0]), (1, 0, 0))
        self.block2 = self.block3 = None

        self.add_prohibit_area(self.block1, padding=0.05)
        target_pose = [-0.04, -0.13, 0.04, -0.05]
        self.prohibited_area.append(target_pose)

        # Place target: same format as stack_blocks_three first block — quat (0,1,0,0), no constrain="free".
        self.block1_target_pose = [0.22, -0.13, 0.75 + self.table_z_bias, 0, 1, 0, 0]

    def play_once(self):
        self.last_gripper = None
        self.last_actor = None

        arm_tag = self.unstack_and_place_block(self.block1, self.block1_target_pose)

        self.info["info"] = {
            "{A}": "red block",
            "{B}": "green block",
            "{C}": "blue block",
            "{a}": str(arm_tag),
            "{b}": str(arm_tag),
            "{c}": str(arm_tag),
        }
        return self.info

    def unstack_and_place_block(self, block: Actor, target_pose, arm_tag=None):
        # Arm by target side (left if target x < 0 else right), like other place tasks.
        if arm_tag is None:
            target_x = target_pose[0] if len(target_pose) >= 1 else 0
            arm_tag = ArmTag("left" if target_x < 0 else "right")

        # Match stack_blocks_three: no contact_point_id (let choose_grasp_pose pick a feasible grasp).
        self.move(self.grasp_actor(block, arm_tag=arm_tag, pre_grasp_dis=0.09))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07))

        # Match stack_blocks_three place_actor exactly: no constrain=, same pre_dis/dis/pre_dis_axis.
        self.move(
            self.place_actor(
                block,
                target_pose=target_pose,
                arm_tag=arm_tag,
                functional_point_id=0,
                pre_dis=0.05,
                dis=0.0,
                pre_dis_axis="fp",
            ))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07))

        self.last_gripper = arm_tag
        self.last_actor = block
        return str(arm_tag)

    def check_success(self):
        z_table_min = 0.74 + self.table_z_bias
        block1_pose = self.block1.get_pose().p
        on_table = block1_pose[2] >= z_table_min
        grippers_open = self.is_left_gripper_open() and self.is_right_gripper_open()
        return on_table and grippers_open
