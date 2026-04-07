"""
Three-block unstack: place all three blocks (top first) to random targets.
- Place targets: random (x, y) and random yaw per episode; block stays upright.
- Grasp: top-down contact points only (box 0,1,2,3).
- Place: default place_actor (pre_dis_axis="fp").
"""
from ._base_task import Base_Task
from .utils import *
import sapien
import math
import numpy as np
import transforms3d as t3d

# Minimum standoff from block surface to avoid gripper going too deep (double-grasp).
GRASP_MIN_STANDOFF = 0.02
# Vertical (z) gap between consecutive block centers at init. Next block placed by reading previous pose, same xy/quat, z += BLOCK_GAP_DIST.
BLOCK_GAP_DIST = 0.02
# Physics steps after each block placement so the stack settles before placing the next.
SETTLE_STEP = 3500
# Random place target bounds (position and orientation).
TARGET_XLIM = [-0.28, 0.28]
TARGET_YLIM = [-0.18, 0.02]
MIN_TARGET_SEP = 0.12  # minimum xy distance between any two targets
MIN_TARGET_TO_STACK = 0.10  # minimum xy distance from stack center (0, -0.13)
BASE_QUAT = [0, 1, 0, 0]  # upright; random yaw applied around world z


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

        # Place one block then settle; repeat. Use previous block's settled pose (same xy and quat, z += BLOCK_GAP_DIST).
        # Reading the actual pose after settle ensures proper alignment and avoids interpenetration/collision explosion.
        self.block1 = create_block(sapien.Pose(center_xy + [z_base], [1, 0, 0, 0]), (1, 0, 0))
        for _ in range(SETTLE_STEP):
            self.scene.step()
        prev_pose = self.block1.get_pose()
        pos2 = [prev_pose.p[0], prev_pose.p[1], prev_pose.p[2] + BLOCK_GAP_DIST]
        self.block2 = create_block(sapien.Pose(pos2, prev_pose.q), (0, 1, 0))
        for _ in range(SETTLE_STEP):
            self.scene.step()
        prev_pose = self.block2.get_pose()
        pos3 = [prev_pose.p[0], prev_pose.p[1], prev_pose.p[2] + BLOCK_GAP_DIST]
        self.block3 = create_block(sapien.Pose(pos3, prev_pose.q), (0, 0, 1))
        for _ in range(SETTLE_STEP):
            self.scene.step()

        self.add_prohibit_area(self.block1, padding=0.05)
        self.add_prohibit_area(self.block2, padding=0.05)
        self.add_prohibit_area(self.block3, padding=0.05)
        target_pose = [-0.04, -0.13, 0.04, -0.05]
        self.prohibited_area.append(target_pose)

        # Random place targets: random (x, y) and random yaw (upright block, rotation in table plane).
        z_t = 0.75 + self.table_z_bias
        stack_xy = np.array([0.0, -0.13])

        def random_target_quat():
            yaw = np.random.uniform(-np.pi, np.pi)
            return t3d.quaternions.qmult(BASE_QUAT, t3d.euler.euler2quat(0, 0, yaw)).tolist()

        def valid_three_targets(p1, p2, p3):
            if np.linalg.norm(p1 - stack_xy) < MIN_TARGET_TO_STACK:
                return False
            if np.linalg.norm(p2 - stack_xy) < MIN_TARGET_TO_STACK:
                return False
            if np.linalg.norm(p3 - stack_xy) < MIN_TARGET_TO_STACK:
                return False
            if np.linalg.norm(p1 - p2) < MIN_TARGET_SEP or np.linalg.norm(p1 - p3) < MIN_TARGET_SEP or np.linalg.norm(p2 - p3) < MIN_TARGET_SEP:
                return False
            return True

        for _ in range(200):
            p1 = np.array([np.random.uniform(TARGET_XLIM[0], TARGET_XLIM[1]), np.random.uniform(TARGET_YLIM[0], TARGET_YLIM[1])])
            p2 = np.array([np.random.uniform(TARGET_XLIM[0], TARGET_XLIM[1]), np.random.uniform(TARGET_YLIM[0], TARGET_YLIM[1])])
            p3 = np.array([np.random.uniform(TARGET_XLIM[0], TARGET_XLIM[1]), np.random.uniform(TARGET_YLIM[0], TARGET_YLIM[1])])
            if not valid_three_targets(p1, p2, p3):
                continue
            q1, q2, q3 = random_target_quat(), random_target_quat(), random_target_quat()
            self.block1_target_pose = [p1[0], p1[1], z_t, *q1]
            self.block2_target_pose = [p2[0], p2[1], z_t, *q2]
            self.block3_target_pose = [p3[0], p3[1], z_t, *q3]
            break
        else:
            q = BASE_QUAT
            self.block1_target_pose = [-0.22, -0.13, z_t, *q]
            self.block2_target_pose = [0.0, -0.13, z_t, *q]
            self.block3_target_pose = [0.22, -0.13, z_t, *q]

    def play_once(self):
        self.last_gripper = None
        self.last_actor = None

        # Unstack top first, then middle, then bottom; place all three to their targets.
        arm_tag3 = self.unstack_and_place_block(self.block3, self.block3_target_pose)
        arm_tag2 = self.unstack_and_place_block(self.block2, self.block2_target_pose)
        arm_tag1 = self.unstack_and_place_block(self.block1, self.block1_target_pose)

        self.info["info"] = {
            "{A}": "red block",
            "{B}": "green block",
            "{C}": "blue block",
            "{a}": str(arm_tag1),
            "{b}": str(arm_tag2),
            "{c}": str(arm_tag3),
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

        # Place: use target pose position and orientation. constrain="align", align_axis=None
        # so transforms.get_place_pose uses target_pose rotation (align actor to target's x axis).
        self.move(
            self.place_actor(
                block,
                target_pose=target_pose,
                arm_tag=arm_tag,
                functional_point_id=0,
                pre_dis=0.05,
                dis=0.0,
                pre_dis_axis="fp",
                constrain="align",
                align_axis=None,
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
        block3_pose = self.block3.get_pose().p
        on_table = (
            block1_pose[2] >= z_table_min
            and block2_pose[2] >= z_table_min
            and block3_pose[2] >= z_table_min
        )
        grippers_open = self.is_left_gripper_open() and self.is_right_gripper_open()
        return on_table and grippers_open
