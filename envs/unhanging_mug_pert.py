"""
Purpose: Perturbation variant of unhanging_mug with segment-level control.
Dependencies:
    - envs._pert_mixin.PerturbationMixin
    - envs.unhanging_mug.unhanging_mug
Usage Example:
    ./collect_data.sh unhanging_mug_pert demo_clean_pert <gpu_id>

Segment Design:
    - grasp_segment[0]: approach phase (from rack, constrained)
    - grasp_segment[1]: descent phase (lifting off rack, high precision)
    - place_segment[0]: approach phase
    - place_segment[1]: descent phase
    - move_segment: lift/retreat movement

Note: Each segment MUST specify 'enabled' key explicitly.
      Unhanging requires high precision to avoid collision with rack.
"""

from ._pert_mixin import PerturbationMixin
from .unhanging_mug import unhanging_mug
from .utils.action import ArmTag


class unhanging_mug_pert(PerturbationMixin, unhanging_mug):
    """
    Unhanging mug task with segment-level perturbation control.

    Task Flow:
        1. Grasp mug from rack (high precision required)
        2. Lift mug off rack
        3. Place mug on table
        4. Retreat

    Note: Approach and descent from rack require careful precision
          to avoid collision and ensure successful grasp.
    """

    def _grasp_mug_from_rack(self, arm_tag):
        """
        Grasp mug from rack.

        @input:
            arm_tag: str or ArmTag, which arm to use
        @output:
            Tuple[ArmTag, List[Action]], grasp actions with config
        @scenario:
            Grasp mug from rack. Very small jitter required for
            successful unhanging without collision.
        """
        return self._wrap_grasp(
            actor=self.mug,
            arm_tag=arm_tag,
            pre_grasp_dis=0.05,
            segments=[
                {
                    "enabled": True,
                    "xy_jitter": 0.005,
                    "yaw_jitter_deg": 4.0,
                    "plan_aug_enabled": False,
                },
                {
                    "enabled": True,
                    "xy_jitter": 0.001,
                    "yaw_jitter_deg": 1.5,
                    "plan_aug_enabled": False,
                },
            ],
        )

    def _place_mug_on_table(self, arm_tag, target_pose):
        """
        Place mug on table.

        @input:
            arm_tag: str or ArmTag, which arm to use
            target_pose: List[float], target position
        @output:
            Tuple[ArmTag, List[Action]], place actions with config
        @scenario:
            Place mug on table surface.
        """
        return self._wrap_place(
            actor=self.mug,
            arm_tag=arm_tag,
            target_pose=target_pose,
            pre_dis=0.05,
            dis=0.0,
            constrain="free",
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

    def _lift(self, arm_tag, z):
        """
        Lift mug.

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
        Execute unhanging mug task with perturbation.

        @input: None
        @output: Dict, task info
        @scenario: Complete unhanging mug task with segment-level perturbation
        """
        arm_tag = ArmTag("right" if self.mug.get_pose().p[0] > 0 else "left")

        self.move(self._grasp_mug_from_rack(arm_tag))
        self.move(self._lift(arm_tag, z=0.08))

        target_pose = self.get_target_pose(arm_tag)
        self.move(self._place_mug_on_table(arm_tag, target_pose))
        self.move(self._lift(arm_tag, z=0.08))

        self.info["info"] = {
            "{A}": f"039_mug/base{self.mug_id}",
            "{B}": "040_rack/base0",
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
