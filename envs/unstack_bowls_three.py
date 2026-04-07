# Purpose: Three-bowl unstack. Initial state: three bowls stacked vertically with BOWL_GAP_DIST
# as vertical (z) gap. Init flow in load_actors: place one bowl -> settle SETTLE_STEP -> repeat
# for each bowl. First bowl at INIT_BOWL_XY + INIT_BOWL_QUAT; each next bowl above previous by
# BOWL_GAP_DIST in z. Task: grasp from top (top-down), then place each bowl to a random target
# position (same xy range and validation as stack_bowls_three). Orientation: bowl opening up
# (QUAT_OF_TARGET_POSE). Dependencies: Base_Task, envs.utils (rand_pose, create_actor, etc.),
# sapien, numpy. Usage: envs.unstack_bowls_three, e.g. script/collect_data.py unstack_bowls_three <task_config>
from ._base_task import Base_Task
from .utils import *
import sapien
import numpy as np

# Gap above table for init spawn to avoid interpenetration (blow away).
DESK_GAP_DIST = 0.035
# Vertical (z) distance between consecutive bowls at init (bowl2/bowl3 each BOWL_GAP_DIST above previous in z).
BOWL_GAP_DIST = 0.04
# Table surface z in unbiased coords (create_actor adds table_z_bias).
TABLE_Z = 0.741
# Physics steps to run after loading actors so the bowl settles.
SETTLE_STEP = 3500
# Init: first bowl at INIT_BOWL_XY (xy), z = TABLE_Z + GAP_DIST; each next bowl same xy, z += BOWL_GAP_DIST. Orientation (quat w,x,y,z).
INIT_BOWL_XY = [0.0, -0.1]
INIT_BOWL_QUAT = [1.0, 0.0, 0.0, 0.0]
# Fallback when target is 3d only (position); 7d target has its own quat.
QUAT_OF_TARGET_POSE = [0.0, 0.707, 0.707, 0.0]
# Random target: same range and method as stack_bowls_three (rand_pose + validation).
# Min distance^2 between targets and from [0, -0.1] (stack uses 0.0169 = 0.13^2).
MIN_TARGET_TO_INIT = 0.13

# Place step tuning (if planning fails, adjust in order below).
# - PLACE_PRE_DIS: height of pre-place waypoint above target (m). Larger = easier to plan (try 0.10, 0.12, 0.15).
# - PLACE_DIS: final approach distance (m). Keep 0.0 to place on target.
# - PLACE_CONSTRAIN: "align" = use target orientation (deterministic EE); "free" = only z aligned (looser, try if align fails).
# - functional_point_id: which functional point of the actor to align to the target. Role: place_actor
#   uses it as place_start_pose (alignment reference). For 002_bowl the only valid value is 0 (defined in
#   model_data; invalid id causes get_functional_point to return None and place_actor to raise
#   'NoneType' has no attribute 'to_transformation_matrix').
PLACE_PRE_DIS = 0.10
PLACE_DIS = 0.0
# "free" = only z aligned (recommended in code_gen/prompt.py for general placement; "align" can yield no IK).
PLACE_CONSTRAIN = "free"
FUNCTIONAL_POINT_ID = 0

# Grasp depth: passed as grasp_dis to grasp_actor. Positive value = shallower grasp (arm stops short of
# nominal contact), reducing penetration so the gripper does not close too deep and lift two stacked bowls.
# Tune upward (e.g. 0.02, 0.03) if both bowls are still lifted; tune downward (e.g. 0.01, 0) if grasp fails.
GRASP_DIS = 0.01


class unstack_bowls_three(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def check_stable(self):
        return super().check_stable()

    def load_actors(self):
        def create_bowl(pose):
            return create_actor(
                self, pose=pose, modelname="002_bowl", model_id=3, convex=True
            )

        # Place one bowl then settle; repeat. First at INIT_BOWL_XY + z_init; each next: 
        # !!! MUST read previous bowl pose, same xy and quat, z += BOWL_GAP_DIST.
        # NOT Using the previous actor's settled pose - 
        # CAUSES PENETRATION AND COLLISION EXPLOSION.
        z_init = TABLE_Z + DESK_GAP_DIST
        pose1 = list(INIT_BOWL_XY) + [z_init]
        self.bowl1 = create_bowl(sapien.Pose(pose1, INIT_BOWL_QUAT))
        for _ in range(SETTLE_STEP):
            self.scene.step()
        prev_pose = self.bowl1.get_pose()
        pos2 = [prev_pose.p[0], prev_pose.p[1], prev_pose.p[2] + BOWL_GAP_DIST]
        self.bowl2 = create_bowl(sapien.Pose(pos2, prev_pose.q))
        for _ in range(SETTLE_STEP):
            self.scene.step()
        prev_pose = self.bowl2.get_pose()
        pos3 = [prev_pose.p[0], prev_pose.p[1], prev_pose.p[2] + BOWL_GAP_DIST]
        self.bowl3 = create_bowl(sapien.Pose(pos3, prev_pose.q))
        for _ in range(SETTLE_STEP):
            self.scene.step()

        self.add_prohibit_area(self.bowl1, padding=0.07)
        self.add_prohibit_area(self.bowl2, padding=0.07)
        self.add_prohibit_area(self.bowl3, padding=0.07)
        self.prohibited_area.append([-0.1, -0.15, 0.1, -0.05])

        # Use settled poses for target validity (min distance from init stack).
        init1_xy = np.array(self.bowl1.get_pose().p[:2])
        init2_xy = np.array(self.bowl2.get_pose().p[:2])
        init3_xy = np.array(self.bowl3.get_pose().p[:2])

        z_t = TABLE_Z + self.table_z_bias
        # Same range and method as stack_bowls_three: rand_pose + validation (|x|>=0.09, dist^2 from [0,-0.1]>=0.0169).
        def valid_target_pose(pose, existing_xy_list):
            if abs(pose.p[0]) < 0.09:
                return False
            if np.sum(np.power(pose.p[:2] - np.array([0, -0.1]), 2)) < 0.0169:
                return False
            p_xy = np.array(pose.p[:2])
            if np.linalg.norm(p_xy - init1_xy) < MIN_TARGET_TO_INIT:
                return False
            if np.linalg.norm(p_xy - init2_xy) < MIN_TARGET_TO_INIT:
                return False
            if np.linalg.norm(p_xy - init3_xy) < MIN_TARGET_TO_INIT:
                return False
            for ex in existing_xy_list:
                if np.sum(np.power(p_xy - ex, 2)) < 0.0169:
                    return False
            return True

        # Orientation for placement: bowl opening up so place_actor IK is feasible (do not use rand_pose quat).
        target_quat = list(QUAT_OF_TARGET_POSE)
        for _ in range(200):
            pose1 = rand_pose(
                xlim=[-0.3, 0.3],
                ylim=[-0.15, 0.15],
                zlim=[z_t, z_t],
                qpos=[0.5, 0.5, 0.5, 0.5],
                ylim_prop=True,
                rotate_rand=False,
            )
            if not valid_target_pose(pose1, []):
                continue
            p1_xy = np.array(pose1.p[:2])
            pose2 = rand_pose(
                xlim=[-0.3, 0.3],
                ylim=[-0.15, 0.15],
                zlim=[z_t, z_t],
                qpos=[0.5, 0.5, 0.5, 0.5],
                ylim_prop=True,
                rotate_rand=False,
            )
            if not valid_target_pose(pose2, [p1_xy]):
                continue
            p2_xy = np.array(pose2.p[:2])
            pose3 = rand_pose(
                xlim=[-0.3, 0.3],
                ylim=[-0.15, 0.15],
                zlim=[z_t, z_t],
                qpos=[0.5, 0.5, 0.5, 0.5],
                ylim_prop=True,
                rotate_rand=False,
            )
            if not valid_target_pose(pose3, [p1_xy, p2_xy]):
                continue
            self.bowl1_target_pose = [pose1.p[0], pose1.p[1], pose1.p[2], *target_quat]
            self.bowl2_target_pose = [pose2.p[0], pose2.p[1], pose2.p[2], *target_quat]
            self.bowl3_target_pose = [pose3.p[0], pose3.p[1], pose3.p[2], *target_quat]
            break
        else:
            self.bowl1_target_pose = [-0.22, -0.1, z_t, *target_quat]
            self.bowl2_target_pose = [0.0, -0.1, z_t, *target_quat]
            self.bowl3_target_pose = [0.22, -0.1, z_t, *target_quat]

        self.quat_of_target_pose = QUAT_OF_TARGET_POSE

    def move_bowl(self, actor, target_pose, arm_tag=None):
        # Arm: same as unstack_blocks_three: target on left (target_pose[0] < 0) -> left arm, else right.
        if arm_tag is None:
            target_x = target_pose[0] if len(np.array(target_pose).flatten()) >= 1 else 0
            arm_tag = ArmTag("left" if target_x < 0 else "right")
        # Grasp: no contact_point_id so all contact points tried; last_gripper/last_actor, pre_grasp_dis=0.09,
        # grasp_dis=GRASP_DIS for shallower grip to avoid lifting two stacked bowls.
        # Place: see PLACE_* constants. pre_dis_axis="fp" = vertical approach.
        flat = np.array(target_pose).flatten()
        if flat.size >= 7:
            target_pose_7d = flat[:7].tolist()
        else:
            target_pose_7d = flat.tolist() + list(self.quat_of_target_pose)

        if self.last_gripper is not None and (self.last_gripper != arm_tag):
            self.move(
                self.grasp_actor(actor, arm_tag=arm_tag, 
                    pre_grasp_dis=0.09, 
                    grasp_dis=GRASP_DIS),
                self.back_to_origin(arm_tag=arm_tag.opposite),
            )
        else:
            self.move(self.grasp_actor(actor, arm_tag=arm_tag, 
            pre_grasp_dis=0.09, 
            grasp_dis=GRASP_DIS))

        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.14))

        self.move(
            self.place_actor(
                actor,
                target_pose=target_pose_7d,
                arm_tag=arm_tag,
                functional_point_id=FUNCTIONAL_POINT_ID,
                pre_dis=PLACE_PRE_DIS,
                dis=PLACE_DIS,
                pre_dis_axis="fp",
                constrain=PLACE_CONSTRAIN,
                align_axis=None,
            )
        )
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07))

        self.last_gripper = arm_tag
        self.last_actor = actor
        return str(arm_tag)

    def play_once(self):
        if getattr(self, "save_data", False):
            self._take_picture()
        self.last_gripper = None
        self.last_actor = None
        # Unstack top-down: bowl3 -> bowl2 -> bowl1 (avoid collision and wrong grasp).
        self.move_bowl(self.bowl3, self.bowl3_target_pose)
        self.move_bowl(self.bowl2, self.bowl2_target_pose)
        self.move_bowl(self.bowl1, self.bowl1_target_pose)
        # Only {A} so instructions (schema: "{A} notifies the bowls") pass filter_instructions.
        self.info["info"] = {"{A}": "002_bowl/base3"}
        return self.info

    def check_success(self):
        """All three bowls at targets and grippers open."""
        t1 = np.array(self.bowl1_target_pose).flatten()
        t2 = np.array(self.bowl2_target_pose).flatten()
        t3 = np.array(self.bowl3_target_pose).flatten()
        p1 = self.bowl1.get_pose().p
        p2 = self.bowl2.get_pose().p
        p3 = self.bowl3.get_pose().p
        eps = 0.02
        return (
            np.all(np.abs(p1[:2] - t1[:2]) < eps)
            and np.abs(p1[2] - t1[2]) < eps
            and np.all(np.abs(p2[:2] - t2[:2]) < eps)
            and np.abs(p2[2] - t2[2]) < eps
            and np.all(np.abs(p3[:2] - t3[:2]) < eps)
            and np.abs(p3[2] - t3[2]) < eps
            and self.is_left_gripper_open()
            and self.is_right_gripper_open()
        )
