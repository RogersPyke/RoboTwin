"""
Purpose: Perturbation variant of unstack_bowls_three with segment-level control.
Dependencies:
    - envs._pert_mixin.PerturbationMixin
    - envs.unstack_bowls_three.unstack_bowls_three
Usage Example:
    ./collect_data.sh unstack_bowls_three_pert demo_clean_pert <gpu_id>

Segment Design:
    - grasp_segment[0]: approach phase (top-down from stack)
    - grasp_segment[1]: descent phase (shallow grasp to avoid double-pick)
    - place_segment[0]: approach phase
    - place_segment[1]: descent phase
    - move_segment: lift/retreat movement

Note: Each segment MUST specify 'enabled' key explicitly.
       Unstack requires shallow grasp to avoid lifting multiple bowls.
"""

from ._pert_mixin import PerturbationMixin
from .unstack_bowls_three import (
    unstack_bowls_three,
    PLACE_PRE_DIS,
    PLACE_DIS,
    PLACE_CONSTRAIN,
    FUNCTIONAL_POINT_ID,
    GRASP_DIS,
)
from .utils.action import ArmTag
import numpy as np


class unstack_bowls_three_pert(PerturbationMixin, unstack_bowls_three):
    """
    Unstack three bowls task with segment-level perturbation control.

    Task Flow:
        For each bowl (top to bottom):
            1. Grasp bowl from stack (shallow grasp)
            2. Lift bowl
            3. Place bowl at target position
            4. Retreat

    Note: Shallow grasp (GRASP_DIS > 0) to avoid picking multiple bowls.
    """

    def _grasp_bowl_from_stack(self, actor, arm_tag):
        """
        Grasp bowl from stack with shallow grip.

        @input:
            actor: Actor, bowl to grasp
            arm_tag: str or ArmTag, which arm to use
        @output:
            Tuple[ArmTag, List[Action]], grasp actions with config
        @scenario:
            Grasp bowl from top of stack. Shallow grasp to avoid
            lifting multiple bowls. Top-down approach.
        """
        return self._wrap_grasp(
            actor=actor,
            arm_tag=arm_tag,
            pre_grasp_dis=0.09,
            grasp_dis=GRASP_DIS,
            segments=[
                {"enabled": True, "xy_jitter": 0.008, "yaw_jitter_deg": 6.0},
                {"enabled": True, "xy_jitter": 0.002, "yaw_jitter_deg": 2.0},
            ],
        )

    def _place_bowl(self, actor, arm_tag, target_pose):
        """
        Place bowl at target position.

        @input:
            actor: Actor, bowl to place
            arm_tag: str or ArmTag, which arm to use
            target_pose: List[float], target position
        @output:
            Tuple[ArmTag, List[Action]], place actions with config
        @scenario:
            Place bowl at random target position.
        """
        return self._wrap_place(
            actor=actor,
            arm_tag=arm_tag,
            target_pose=target_pose,
            functional_point_id=FUNCTIONAL_POINT_ID,
            pre_dis=PLACE_PRE_DIS,
            dis=PLACE_DIS,
            pre_dis_axis="fp",
            constrain=PLACE_CONSTRAIN,
            align_axis=None,
            segments=[
                {"enabled": True, "xy_jitter": 0.010, "yaw_jitter_deg": 8.0},
                {"enabled": True, "xy_jitter": 0.003, "yaw_jitter_deg": 3.0},
            ],
        )

    def _lift(self, arm_tag, z):
        """
        Lift bowl.

        @input:
            arm_tag: str or ArmTag, which arm to use
            z: float, lift distance
        @output:
            Tuple[ArmTag, List[Action]], lift action with config
        @scenario:
            Vertical lift movement.
        """
        return self._wrap_move(
            arm_tag=arm_tag,
            z=z,
            segment={"enabled": True, "xy_jitter": 0.008, "yaw_jitter_deg": 6.0},
        )

    def _back_to_origin(self, arm_tag):
        """
        Return arm to original position.

        @input:
            arm_tag: str or ArmTag, which arm to use
        @output:
            Tuple[ArmTag, List[Action]], return action with config
        @scenario:
            Return to home position.
        """
        return self._wrap_back_to_origin(
            arm_tag=arm_tag,
            segment={"enabled": True, "xy_jitter": 0.010, "yaw_jitter_deg": 8.0},
        )

    def move_bowl(self, actor, target_pose, arm_tag=None):
        """
        Move a single bowl from stack to target position.

        @input:
            actor: Actor, bowl to move
            target_pose: List[float], target position
            arm_tag: str or ArmTag or None, which arm to use
        @output:
            str, arm tag used
        @scenario:
            Complete bowl movement with shallow grasp.
        """
        if arm_tag is None:
            target_x = (
                target_pose[0] if len(np.array(target_pose).flatten()) >= 1 else 0
            )
            arm_tag = ArmTag("left" if target_x < 0 else "right")

        flat = np.array(target_pose).flatten()
        if flat.size >= 7:
            target_pose_7d = flat[:7].tolist()
        else:
            target_pose_7d = flat.tolist() + list(self.quat_of_target_pose)

        if self.last_gripper is not None and (self.last_gripper != arm_tag):
            self.move(
                self._grasp_bowl_from_stack(actor, arm_tag),
                self._back_to_origin(arm_tag.opposite),
            )
        else:
            self.move(self._grasp_bowl_from_stack(actor, arm_tag))

        self.move(self._lift(arm_tag, z=0.14))
        self.move(self._place_bowl(actor, arm_tag, target_pose_7d))
        self.move(self._lift(arm_tag, z=0.07))

        self.last_gripper = arm_tag
        self.last_actor = actor
        return str(arm_tag)

    def play_once(self):
        """
        Execute unstack bowls task with perturbation.

        @input: None
        @output: Dict, task info
        @scenario: Complete unstack bowls task with segment-level perturbation
        """
        if getattr(self, "save_data", False):
            self._take_picture()

        self.last_gripper = None
        self.last_actor = None

        self.move_bowl(self.bowl3, self.bowl3_target_pose)
        self.move_bowl(self.bowl2, self.bowl2_target_pose)
        self.move_bowl(self.bowl1, self.bowl1_target_pose)

        self.info["info"] = {"{A}": "002_bowl/base3"}

        result = self.info

        if self._end_reset_to_init:
            self.move(
                self._wrap_back_to_origin(ArmTag("left"), segment={"enabled": False})
            )
            self.move(
                self._wrap_back_to_origin(ArmTag("right"), segment={"enabled": False})
            )

        return result
