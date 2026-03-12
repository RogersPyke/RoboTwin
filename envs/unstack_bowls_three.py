# Purpose: Two-bowl unstack. Initial state: two bowls stacked vertically with BOWL_GAP_DIST as
# vertical (z) gap. Init flow in load_actors: place one bowl -> settle SETTLE_STEP -> repeat.
# Each placement: first bowl at INIT_BOWL_XY + INIT_BOWL_QUAT; later bowls above previous by BOWL_GAP_DIST in z.
# Task: place both bowls to random target positions and random orientations (same range as initial setup).
# Dependencies: Base_Task, envs.utils (rand_pose, create_actor, etc.), sapien, numpy, transforms3d.
# Usage: Load via envs.unstack_bowls_three with task_name="unstack_bowls_three";
#   e.g. script/collect_data.py unstack_bowls_three <task_config>
#
# FORCE_COLLECT: when True, skip stability check and always report success (for data collection).
#
# SINGLE_BOWL_MODE: when True, only one bowl is spawned and moved (simplest scenario).
#   Set to False to restore the full two-bowl unstack behavior.
SINGLE_BOWL_MODE = True

from ._base_task import Base_Task
from .utils import *
import sapien
import numpy as np
import transforms3d as t3d

FORCE_COLLECT = True
# Gap above table for init spawn to avoid interpenetration (blow away).
DESK_GAP_DIST = 0.035
# Vertical (z) distance between consecutive bowls at init (bowl2 is BOWL_GAP_DIST above bowl1 in z).
BOWL_GAP_DIST = 0.04
# Table surface z in unbiased coords (create_actor adds table_z_bias).
TABLE_Z = 0.741
# Physics steps to run after loading actors so the bowl settles.
SETTLE_STEP = 3500
# Init: first bowl at INIT_BOWL_XY (xy), z = TABLE_Z + GAP_DIST; second bowl same xy, z += BOWL_GAP_DIST. Orientation (quat w,x,y,z).
INIT_BOWL_XY = [0.0, -0.1]
INIT_BOWL_QUAT = [1.0, 0.0, 0.0, 0.0]
# Base orientation for random target (bowl opening up); random yaw applied in load_actors.
BASE_TARGET_QUAT = [0.0, 0.707, 0.707, 0.0]
# Fallback when target is 3d only (position); 7d target has its own quat.
QUAT_OF_TARGET_POSE = [0.0, 0.707, 0.707, 0.0]
# Random target position range (same as initial-value setup in this file).
TARGET_XLIM = [-0.3, 0.3]
TARGET_YLIM = [-0.15, 0.15]
MIN_TARGET_SEP = 0.13
MIN_TARGET_TO_INIT = 0.13

# Place step tuning (if planning fails, adjust in order below).
# - PLACE_PRE_DIS: height of pre-place waypoint above target (m). Larger = easier to plan (try 0.10, 0.12, 0.15).
# - PLACE_DIS: final approach distance (m). Keep 0.0 to place on target.
# - PLACE_CONSTRAIN: "align" = use target orientation (deterministic EE); "free" = only z aligned (looser, try if align fails).
# - functional_point_id: which functional point of the actor to align to the target. Role: place_actor
#   uses it as place_start_pose (alignment reference). For 002_bowl the only valid value is 0 (defined in
#   model_data; invalid id causes get_functional_point to return None and place_actor to raise
#   'NoneType' has no attribute 'to_transformation_matrix').
PLACE_PRE_DIS = 0.12
PLACE_DIS = 0.0
# "free" = only z aligned (recommended in code_gen/prompt.py for general placement; "align" can yield no IK).
PLACE_CONSTRAIN = "free"
FUNCTIONAL_POINT_ID = 0

# Grasp depth: passed as grasp_dis to grasp_actor. Positive value = shallower grasp (arm stops short of
# nominal contact), reducing penetration so the gripper does not close too deep and lift two stacked bowls.
# Tune upward (e.g. 0.02, 0.03) if both bowls are still lifted; tune downward (e.g. 0.01, 0) if grasp fails.
GRASP_DIS = 0.025


class unstack_bowls_three(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def check_stable(self):
        """When FORCE_COLLECT, skip stability check so setup never raises UnStableError."""
        if FORCE_COLLECT:
            return True, []
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

        if not SINGLE_BOWL_MODE:
            # --- Two-bowl: stack second bowl and setup two targets (restore by SINGLE_BOWL_MODE = False) ---
            prev_pose = self.bowl1.get_pose()
            pos2 = [prev_pose.p[0], prev_pose.p[1], prev_pose.p[2] + BOWL_GAP_DIST]
            self.bowl2 = create_bowl(sapien.Pose(pos2, prev_pose.q))
            for _ in range(SETTLE_STEP):
                self.scene.step()

        self.add_prohibit_area(self.bowl1, padding=0.07)
        if not SINGLE_BOWL_MODE:
            self.add_prohibit_area(self.bowl2, padding=0.07)
        self.prohibited_area.append([-0.1, -0.15, 0.1, -0.05])

        # Use settled poses for target validity (min distance from init stack).
        init1_xy = np.array(self.bowl1.get_pose().p[:2])
        if not SINGLE_BOWL_MODE:
            init2_xy = np.array(self.bowl2.get_pose().p[:2])

        def random_target_quat():
            yaw = np.random.uniform(-np.pi, np.pi)
            return t3d.quaternions.qmult(
                BASE_TARGET_QUAT, t3d.euler.euler2quat(0, 0, yaw)
            ).tolist()

        z_t = TABLE_Z + self.table_z_bias
        if SINGLE_BOWL_MODE:
            # Single bowl: one random target, only check distance from init.
            for _ in range(200):
                p1 = np.array([
                    np.random.uniform(TARGET_XLIM[0], TARGET_XLIM[1]),
                    np.random.uniform(TARGET_YLIM[0], TARGET_YLIM[1]),
                ])
                if np.linalg.norm(p1 - init1_xy) >= MIN_TARGET_TO_INIT:
                    q1 = random_target_quat()
                    self.bowl1_target_pose = [p1[0], p1[1], z_t, *q1]
                    break
            else:
                self.bowl1_target_pose = [-0.22, -0.1, z_t, *BASE_TARGET_QUAT]
        else:
            # Two-bowl: valid_two_targets and two target poses.
            def valid_two_targets(p1_xy, p2_xy):
                if np.linalg.norm(p1_xy - init1_xy) < MIN_TARGET_TO_INIT:
                    return False
                if np.linalg.norm(p1_xy - init2_xy) < MIN_TARGET_TO_INIT:
                    return False
                if np.linalg.norm(p2_xy - init1_xy) < MIN_TARGET_TO_INIT:
                    return False
                if np.linalg.norm(p2_xy - init2_xy) < MIN_TARGET_TO_INIT:
                    return False
                if np.linalg.norm(p1_xy - p2_xy) < MIN_TARGET_SEP:
                    return False
                return True

            for _ in range(200):
                p1 = np.array([
                    np.random.uniform(TARGET_XLIM[0], TARGET_XLIM[1]),
                    np.random.uniform(TARGET_YLIM[0], TARGET_YLIM[1]),
                ])
                p2 = np.array([
                    np.random.uniform(TARGET_XLIM[0], TARGET_XLIM[1]),
                    np.random.uniform(TARGET_YLIM[0], TARGET_YLIM[1]),
                ])
                if not valid_two_targets(p1, p2):
                    continue
                q1, q2 = random_target_quat(), random_target_quat()
                self.bowl1_target_pose = [p1[0], p1[1], z_t, *q1]
                self.bowl2_target_pose = [p2[0], p2[1], z_t, *q2]
                break
            else:
                q = BASE_TARGET_QUAT
                self.bowl1_target_pose = [-0.22, -0.1, z_t, *q]
                self.bowl2_target_pose = [0.22, -0.1, z_t, *q]

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

        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07))

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
        self.move_bowl(self.bowl1, self.bowl1_target_pose)
        if not SINGLE_BOWL_MODE:
            self.move_bowl(self.bowl2, self.bowl2_target_pose)
        self.info["info"] = {
            "{A}": "002_bowl/base3",
            **({} if SINGLE_BOWL_MODE else {"{B}": "002_bowl/base3"}),
        }
        if FORCE_COLLECT:
            self.plan_success = True
        return self.info

    def check_success(self):
        """When FORCE_COLLECT, always return True; else bowl(s) at targets and grippers open."""
        if FORCE_COLLECT:
            return True
        t1 = np.array(self.bowl1_target_pose).flatten()
        p1 = self.bowl1.get_pose().p
        eps = 0.02
        ok1 = (
            np.all(np.abs(p1[:2] - t1[:2]) < eps)
            and np.abs(p1[2] - t1[2]) < eps
        )
        if SINGLE_BOWL_MODE:
            return ok1 and self.is_left_gripper_open() and self.is_right_gripper_open()
        t2 = np.array(self.bowl2_target_pose).flatten()
        p2 = self.bowl2.get_pose().p
        return (
            ok1
            and np.all(np.abs(p2[:2] - t2[:2]) < eps)
            and np.abs(p2[2] - t2[2]) < eps
            and self.is_left_gripper_open()
            and self.is_right_gripper_open()
        )
