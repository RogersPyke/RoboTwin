"""
Purpose: Perturbation variant of unstack_blocks_three with segment-level control.
Dependencies:
    - envs._pert_mixin.PerturbationMixin
    - envs.unstack_blocks_three.unstack_blocks_three
Usage Example:
    ./collect_data.sh unstack_blocks_three_pert demo_clean_pert <gpu_id>

Segment Design:
    - grasp_segment[0]: approach phase
    - grasp_segment[1]: descent phase
    - place_segment[0]: approach phase
    - place_segment[1]: descent phase
    - move_segment: lift/retreat movement

Note: Each segment MUST specify 'enabled' key explicitly.
"""

from ._pert_mixin import PerturbationMixin
from .unstack_blocks_three import (
    unstack_blocks_three,
    GRASP_MIN_STANDOFF,
)
from .utils.action import ArmTag


class unstack_blocks_three_pert(PerturbationMixin, unstack_blocks_three):
    """
    Unstack three blocks task with segment-level perturbation control.
    """

    def _grasp_block_from_stack(self, actor, arm_tag):
        """
        Grasp block from stack.

        @input:
            actor: Actor, block to grasp
            arm_tag: str or ArmTag, which arm to use
        @output:
            Tuple[ArmTag, List[Action]], grasp actions with config
        @scenario:
            Grasp block from stack with standoff to avoid double-pick.
        """
        return self._wrap_grasp(
            actor=actor,
            arm_tag=arm_tag,
            contact_point_id=0,
            pre_grasp_dis=0.08,
            grasp_dis=GRASP_MIN_STANDOFF,
            segments=[
                {"enabled": True, "xy_jitter": 0.008, "yaw_jitter_deg": 6.0},
                {"enabled": True, "xy_jitter": 0.002, "yaw_jitter_deg": 2.0},
            ],
        )

    def _place_block(self, actor, arm_tag, target_pose):
        """
        Place block at target position.

        @input:
            actor: Actor, block to place
            arm_tag: str or ArmTag, which arm to use
            target_pose: List[float], target position
        @output:
            Tuple[ArmTag, List[Action]], place actions with config
        @scenario:
            Place block at target position.
        """
        return self._wrap_place(
            actor=actor,
            arm_tag=arm_tag,
            target_pose=target_pose,
            functional_point_id=0,
            pre_dis=0.10,
            dis=0.0,
            constrain="free",
            segments=[
                {"enabled": True, "xy_jitter": 0.010, "yaw_jitter_deg": 8.0},
                {"enabled": True, "xy_jitter": 0.003, "yaw_jitter_deg": 3.0},
            ],
        )

    def _lift(self, arm_tag, z):
        """
        Lift block.

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

    def move_block(self, actor, target_pose, arm_tag=None):
        """
        Move a single block from stack to target position.

        @input:
            actor: Actor, block to move
            target_pose: List[float], target position
            arm_tag: str or ArmTag or None, which arm to use
        @output:
            str, arm tag used
        @scenario:
            Complete block movement.
        """
        if arm_tag is None:
            target_x = target_pose[0] if len(target_pose) >= 1 else 0
            arm_tag = ArmTag("left" if target_x < 0 else "right")

        if self.last_gripper is not None and self.last_gripper != arm_tag:
            self.move(
                self._grasp_block_from_stack(actor, arm_tag),
                self._back_to_origin(arm_tag.opposite),
            )
        else:
            self.move(self._grasp_block_from_stack(actor, arm_tag))

        self.move(self._lift(arm_tag, z=0.12))
        self.move(self._place_block(actor, arm_tag, target_pose))
        self.move(self._lift(arm_tag, z=0.1))

        self.last_gripper = arm_tag
        self.last_actor = actor
        return str(arm_tag)

    def play_once(self):
        """
        Execute unstack blocks task with perturbation.

        @input: None
        @output: Dict, task info
        @scenario: Complete unstack blocks task with segment-level perturbation
        """
        self.last_gripper = None
        self.last_actor = None

        self.move_block(self.block3, self.block3_target_pose)
        self.move_block(self.block2, self.block2_target_pose)
        self.move_block(self.block1, self.block1_target_pose)

        self.info["info"] = {"{A}": "block"}

        result = self.info

        if self._end_reset_to_init:
            self.move(
                self._wrap_back_to_origin(ArmTag("left"), segment={"enabled": False})
            )
            self.move(
                self._wrap_back_to_origin(ArmTag("right"), segment={"enabled": False})
            )

        return result
