"""
Purpose: Perturbation variant of unmove_pillbottle_pad with segment-level control.
Dependencies:
    - envs._pert_mixin.PerturbationMixin
    - envs.unmove_pillbottle_pad.unmove_pillbottle_pad
Usage Example:
    ./collect_data.sh unmove_pillbottle_pad_pert demo_clean_pert <gpu_id>

Segment Design:
    - grasp_segment[0]: approach phase
    - grasp_segment[1]: descent phase
    - place_segment[0]: approach phase
    - place_segment[1]: descent phase
    - move_segment: lift/retreat movement

Note: Each segment MUST specify 'enabled' key explicitly.
"""

from ._pert_mixin import PerturbationMixin
from .unmove_pillbottle_pad import unmove_pillbottle_pad
from .utils.action import ArmTag


class unmove_pillbottle_pad_pert(PerturbationMixin, unmove_pillbottle_pad):
    """
    Unmove pillbottle from pad task with segment-level perturbation control.

    Task Flow:
        1. Grasp pillbottle from pad
        2. Lift pillbottle
        3. Place pillbottle on table
        4. Retreat
    """

    def _grasp_pillbottle_from_pad(self, arm_tag):
        """
        Grasp pillbottle from pad.

        @input:
            arm_tag: str or ArmTag, which arm to use
        @output:
            Tuple[ArmTag, List[Action]], grasp actions with config
        @scenario:
            Grasp pillbottle from pad surface.
        """
        return self._wrap_grasp(
            actor=self.pillbottle,
            arm_tag=arm_tag,
            pre_grasp_dis=0.06,
            gripper_pos=0,
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

    def _place_pillbottle_on_table(self, arm_tag, target_pose):
        """
        Place pillbottle on table.

        @input:
            arm_tag: str or ArmTag, which arm to use
            target_pose: List[float], target position
        @output:
            Tuple[ArmTag, List[Action]], place actions with config
        @scenario:
            Place pillbottle on table surface.
        """
        return self._wrap_place(
            actor=self.pillbottle,
            arm_tag=arm_tag,
            target_pose=target_pose,
            pre_dis=0.05,
            dis=0,
            constrain="free",
            segments=[
                {
                    "enabled": True,
                    "xy_jitter": 0.012,
                    "yaw_jitter_deg": 10.0,
                    "plan_aug_enabled": True,
                    "waypoint_count_max": 2,
                },
                {
                    "enabled": True,
                    "xy_jitter": 0.004,
                    "yaw_jitter_deg": 4.0,
                    "plan_aug_enabled": False,
                },
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
            segment={
                "enabled": True,
                "xy_jitter": 0.008,
                "yaw_jitter_deg": 6.0,
                "plan_aug_enabled": True,
                "waypoint_count_max": 1,
            },
        )

    def play_once(self):
        """
        Execute unmove pillbottle task with perturbation.

        @input: None
        @output: Dict, task info
        @scenario: Complete unmove pillbottle task with segment-level perturbation
        """
        arm_tag = ArmTag("right" if self.pillbottle.get_pose().p[0] > 0 else "left")

        self.move(self._grasp_pillbottle_from_pad(arm_tag))
        self.move(self._lift(arm_tag, z=0.05))

        target_pose = self.get_target_pose(arm_tag)
        self.move(self._place_pillbottle_on_table(arm_tag, target_pose))

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
