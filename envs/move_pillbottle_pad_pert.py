"""
Purpose: Perturbation variant of move_pillbottle_pad with segment-level control.
Dependencies:
    - envs._pert_mixin.PerturbationMixin
    - envs.move_pillbottle_pad.move_pillbottle_pad
Usage Example:
    ./collect_data.sh move_pillbottle_pad_pert demo_clean_pert <gpu_id>

Segment Design:
    - grasp_segment[0]: approach phase
    - grasp_segment[1]: descent phase (precision required)
    - place_segment[0]: approach phase
    - place_segment[1]: descent phase (precision required)
    - move_segment: lift movement

Note: Each segment MUST specify 'enabled' key explicitly.
"""

from ._pert_mixin import PerturbationMixin
from .move_pillbottle_pad import move_pillbottle_pad
from .utils.action import ArmTag


class move_pillbottle_pad_pert(PerturbationMixin, move_pillbottle_pad):
    """
    Move pillbottle to pad task with segment-level perturbation control.

    Task Flow:
        1. Grasp pillbottle from table
        2. Lift pillbottle
        3. Place pillbottle on pad
        4. Retreat

    Each action is wrapped with explicit segment configuration.
    """

    def _grasp_pillbottle(self, arm_tag):
        """
        Grasp pillbottle from table.

        @input:
            arm_tag: str or ArmTag, which arm to use
        @output:
            Tuple[ArmTag, List[Action]], grasp actions with config
        @scenario:
            Grasp pillbottle from random table position.
        """
        return self._wrap_grasp(
            actor=self.pillbottle,
            arm_tag=arm_tag,
            pre_grasp_dis=0.06,
            gripper_pos=0,
            segments=[
                {"enabled": True, "xy_jitter": 0.006, "yaw_jitter_deg": 4.0},
                {"enabled": True, "xy_jitter": 0.002, "yaw_jitter_deg": 2.0},
            ],
        )

    def _place_on_pad(self, arm_tag, target_pose):
        """
        Place pillbottle on pad.

        @input:
            arm_tag: str or ArmTag, which arm to use
            target_pose: List[float], pad functional point
        @output:
            Tuple[ArmTag, List[Action]], place actions with config
        @scenario:
            Place pillbottle on target pad position.
        """
        return self._wrap_place(
            actor=self.pillbottle,
            arm_tag=arm_tag,
            target_pose=target_pose,
            pre_dis=0.05,
            dis=0,
            functional_point_id=0,
            pre_dis_axis="fp",
            segments=[
                {"enabled": True, "xy_jitter": 0.006, "yaw_jitter_deg": 4.0},
                {"enabled": True, "xy_jitter": 0.003, "yaw_jitter_deg": 2.0},
            ],
        )

    def _lift(self, arm_tag, z):
        """
        Lift pillbottle.

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

    def play_once(self):
        """
        Execute move pillbottle task with perturbation.

        @input: None
        @output: Dict, task info
        @scenario: Complete move pillbottle task with segment-level perturbation
        """
        arm_tag = ArmTag("right" if self.pillbottle.get_pose().p[0] > 0 else "left")

        self.move(self._grasp_pillbottle(arm_tag))
        self.move(self._lift(arm_tag, z=0.05))

        target_pose = self.pad.get_functional_point(1)
        self.move(self._place_on_pad(arm_tag, target_pose))

        self.info["info"] = {
            "{A}": f"080_pillbottle/base{self.pillbottle_id}",
            "{a}": str(arm_tag),
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
