"""
Purpose: Perturbation variant of stack_blocks_three with segment-level control.
Dependencies:
    - envs._pert_mixin.PerturbationMixin
    - envs.stack_blocks_three.stack_blocks_three
Usage Example:
    ./collect_data.sh stack_blocks_three_pert demo_clean_pert <gpu_id>

Segment Design:
    - grasp_segment[0]: approach phase
    - grasp_segment[1]: descent phase
    - place_segment[0]: approach phase
    - place_segment[1]: descent phase (stacking precision)
    - move_segment: lift/retreat movement

Note: Each segment MUST specify 'enabled' key explicitly.
"""

from ._pert_mixin import PerturbationMixin
from .stack_blocks_three import stack_blocks_three
from .utils.action import ArmTag


class stack_blocks_three_pert(PerturbationMixin, stack_blocks_three):
    """
    Stack three blocks task with segment-level perturbation control.
    """

    def _grasp_block(self, actor, arm_tag):
        """
        Grasp block from table.

        @input:
            actor: Actor, block to grasp
            arm_tag: str or ArmTag, which arm to use
        @output:
            Tuple[ArmTag, List[Action]], grasp actions with config
        @scenario:
            Grasp block with standard perturbation settings.
        """
        return self._wrap_grasp(
            actor=actor,
            arm_tag=arm_tag,
            pre_grasp_dis=0.09,
            segments=[
                {"enabled": True, "xy_jitter": 0.006, "yaw_jitter_deg": 4.0},
                {"enabled": True, "xy_jitter": 0.002, "yaw_jitter_deg": 2.0},
            ],
        )

    def _place_block(self, actor, arm_tag, target_pose):
        """
        Place block on stack.

        @input:
            actor: Actor, block to place
            arm_tag: str or ArmTag, which arm to use
            target_pose: List[float], target position
        @output:
            Tuple[ArmTag, List[Action]], place actions with config
        @scenario:
            Place block with precision for stacking.
        """
        return self._wrap_place(
            actor=actor,
            arm_tag=arm_tag,
            target_pose=target_pose,
            functional_point_id=0,
            pre_dis=0.05,
            dis=0.0,
            pre_dis_axis="fp",
            segments=[
                {"enabled": True, "xy_jitter": 0.005, "yaw_jitter_deg": 3.0},
                {"enabled": True, "xy_jitter": 0.002, "yaw_jitter_deg": 1.5},
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
            segment={"enabled": True, "xy_jitter": 0.005, "yaw_jitter_deg": 3.0},
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
            segment={"enabled": True, "xy_jitter": 0.006, "yaw_jitter_deg": 4.0},
        )

    def pick_and_place_block(self, block):
        """
        Pick and place a single block.

        @input:
            block: Actor, block to move
        @output:
            str, arm tag used
        @scenario:
            Complete block movement with grasp, lift, place, retreat.
        """
        block_pose = block.get_pose().p
        arm_tag = ArmTag("left" if block_pose[0] < 0 else "right")

        if self.last_gripper is not None and (self.last_gripper != arm_tag):
            self.move(
                self._grasp_block(block, arm_tag),
                self._back_to_origin(arm_tag.opposite),
            )
        else:
            self.move(self._grasp_block(block, arm_tag))

        self.move(self._lift(arm_tag, z=0.07))

        if self.last_actor is None:
            target_pose = [0, -0.13, 0.75 + self.table_z_bias, 0, 1, 0, 0]
        else:
            target_pose = self.last_actor.get_functional_point(1)

        self.move(self._place_block(block, arm_tag, target_pose))
        self.move(self._lift(arm_tag, z=0.07))

        self.last_gripper = arm_tag
        self.last_actor = block
        return str(arm_tag)

    def play_once(self):
        """
        Execute stack blocks task with perturbation.

        @input: None
        @output: Dict, task info
        @scenario: Complete stack blocks task with segment-level perturbation
        """
        self.last_gripper = None
        self.last_actor = None

        arm_tag1 = self.pick_and_place_block(self.block1)
        arm_tag2 = self.pick_and_place_block(self.block2)
        arm_tag3 = self.pick_and_place_block(self.block3)

        self.info["info"] = {
            "{A}": "red block",
            "{B}": "green block",
            "{C}": "blue block",
            "{a}": arm_tag1,
            "{b}": arm_tag2,
            "{c}": arm_tag3,
        }

        result = self.info

        if self._end_reset_to_init:
            self.move(
                self._wrap_back_to_origin(ArmTag("left"), segment={"enabled": False})
            )
            self.move(
                self._wrap_back_to_origin(ArmTag("right"), segment={"enabled": False})
            )

        return result
