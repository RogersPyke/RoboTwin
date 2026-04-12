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
            contact_point_id=[0, 2][int(arm_tag == "left")],
            pre_grasp_dis=0.1,
            segments=[
                {
                    "enabled": True,
                    "xy_jitter": 0.010,
                    "yaw_jitter_deg": 8.0,
                    "plan_aug_enabled": True,
                    "waypoint_count_max": 2,
                },
                {
                    "enabled": True,
                    "xy_jitter": 0.003,
                    "yaw_jitter_deg": 3.0,
                    "plan_aug_enabled": False,
                },
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
            target_pose=target_pose.tolist() + self.quat_of_target_pose,
            functional_point_id=0,
            pre_dis=0.1,
            dis=0,
            constrain="align",
            segments=[
                {
                    "enabled": True,
                    "xy_jitter": 0.008,
                    "yaw_jitter_deg": 6.0,
                    "plan_aug_enabled": True,
                    "waypoint_count_max": 2,
                },
                {
                    "enabled": True,
                    "xy_jitter": 0.002,
                    "yaw_jitter_deg": 2.0,
                    "plan_aug_enabled": False,
                },
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
            segment={
                "enabled": True,
                "xy_jitter": 0.008,
                "yaw_jitter_deg": 6.0,
                "plan_aug_enabled": True,
                "waypoint_count_max": 1,
            },
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
            segment={
                "enabled": True,
                "xy_jitter": 0.010,
                "yaw_jitter_deg": 8.0,
                "plan_aug_enabled": True,
                "waypoint_count_max": 2,
            },
        )

    def move_block(self, actor, target_pose):
        """
        Move a single block to target position.

        @input:
            actor: Actor, block to move
            target_pose: np.ndarray, target position
        @output:
            str, arm tag used
        @scenario:
            Complete block movement with grasp, lift, place, retreat.
        """
        actor_pose = actor.get_pose().p
        arm_tag = ArmTag("left" if actor_pose[0] < 0 else "right")

        if self.las_arm is None or arm_tag == self.las_arm:
            self.move(self._grasp_block(actor, arm_tag))
        else:
            self.move(
                self._grasp_block(actor, arm_tag),
                self._back_to_origin(arm_tag.opposite),
            )

        self.move(self._lift(arm_tag, z=0.1))
        self.move(self._place_block(actor, arm_tag, target_pose))
        self.move(self._lift(arm_tag, z=0.1))

        self.las_arm = arm_tag
        return arm_tag

    def play_once(self):
        """
        Execute stack blocks task with perturbation.

        @input: None
        @output: Dict, task info
        @scenario: Complete stack blocks task with segment-level perturbation
        """
        self.las_arm = None

        self.move_block(self.block1, self.block1_target_pose)
        self.move_block(self.block2, self.block2_target_pose)
        self.move_block(self.block3, self.block3_target_pose)

        result = self.info

        if self._end_reset_to_init:
            self.move(
                self._wrap_back_to_origin(ArmTag("left"), segment={"enabled": False})
            )
            self.move(
                self._wrap_back_to_origin(ArmTag("right"), segment={"enabled": False})
            )

        return result
