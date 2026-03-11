"""
Two-block unstack: place both blocks (top first, then bottom) to their targets.
- Grasp: top-down contact points only (box 0,1,2,3).
- Place: default place_actor (pre_dis_axis="fp").
"""
from ._base_task import Base_Task
from .utils import *
import sapien
import math

# Minimum standoff from block surface to avoid gripper going too deep (double-grasp).
GRASP_MIN_STANDOFF = 0.02


class unstack_blocks_three(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        block_half_size = 0.025
        center_xy = [0.0, -0.13]
        z_base = 0.75 - block_half_size  # block center so top of bottom block = 0.75

        def create_block(block_pose, color):
            return create_box(
                scene=self,
                pose=block_pose,
                half_size=(block_half_size, block_half_size, block_half_size),
                color=color,
                name="box",
            )

        # Two blocks stacked at center (block1 bottom, block2 top).
        self.block1 = create_block(sapien.Pose(center_xy + [z_base], [1, 0, 0, 0]), (1, 0, 0))
        self.block2 = create_block(sapien.Pose(center_xy + [z_base + 0.05], [1, 0, 0, 0]), (0, 1, 0))
        self.block3 = None

        self.add_prohibit_area(self.block1, padding=0.05)
        self.add_prohibit_area(self.block2, padding=0.05)
        target_pose = [-0.04, -0.13, 0.04, -0.05]
        self.prohibited_area.append(target_pose)

        # Fixed place targets: same format as stack (quat 0,1,0,0).
        q = [0, 1, 0, 0]
        z_t = 0.75 + self.table_z_bias
        self.block1_target_pose = [-0.22, -0.13, z_t, *q]   # left
        self.block2_target_pose = [0.22, -0.13, z_t, *q]     # right

    def play_once(self):
        self.last_gripper = None
        self.last_actor = None

        # Unstack top first, then bottom; place both to their targets.
        arm_tag2 = self.unstack_and_place_block(self.block2, self.block2_target_pose)
        arm_tag1 = self.unstack_and_place_block(self.block1, self.block1_target_pose)

        self.info["info"] = {
            "{A}": "red block",
            "{B}": "green block",
            "{C}": "blue block",
            "{a}": str(arm_tag1),
            "{b}": str(arm_tag2),
            "{c}": str(arm_tag2),
        }
        return self.info

    def unstack_and_place_block(self, block: Actor, target_pose, arm_tag=None):
        if arm_tag is None:
            target_x = target_pose[0] if len(target_pose) >= 1 else 0
            arm_tag = ArmTag("left" if target_x < 0 else "right")

        # Grasp: top-down only; grasp_dis = standoff so gripper keeps minimum distance from block surface.
        grasp_kw = dict(
            actor=block,
            arm_tag=arm_tag,
            pre_grasp_dis=0.09,
            grasp_dis=GRASP_MIN_STANDOFF,
            contact_point_id=[0, 1, 2, 3],
        )
        if self.last_gripper is not None and (self.last_gripper != arm_tag):
            self.move(
                self.grasp_actor(**grasp_kw),
                self.back_to_origin(arm_tag=arm_tag.opposite),
            )
        else:
            self.move(self.grasp_actor(**grasp_kw))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07))

        # Place: top-down approach (pre_dis_axis="fp" = along target z). No align_axis — it can cause
        # "Eigenvalues did not converge" when actor_axis is block z (degenerate in get_place_pose).
        self.move(
            self.place_actor(
                block,
                target_pose=target_pose,
                arm_tag=arm_tag,
                functional_point_id=0,
                pre_dis=0.05,
                dis=0.0,
                pre_dis_axis="fp",
            )
        )
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07))

        self.last_gripper = arm_tag
        self.last_actor = block
        return str(arm_tag)

    def check_success(self):
        z_table_min = 0.74 + self.table_z_bias
        block1_pose = self.block1.get_pose().p
        block2_pose = self.block2.get_pose().p
        # Both blocks placed on table.
        on_table = block1_pose[2] >= z_table_min and block2_pose[2] >= z_table_min
        grippers_open = self.is_left_gripper_open() and self.is_right_gripper_open()
        return on_table and grippers_open
