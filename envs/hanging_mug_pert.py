"""
Purpose: Perturbation variant of hanging_mug with segment-level control.
Dependencies:
    - envs._pert_mixin.PerturbationMixin
    - envs.hanging_mug.hanging_mug
Usage Example:
    ./collect_data.sh hanging_mug_pert demo_clean_pert <gpu_id>

Segment Design:
    - grasp_segment[0]: approach phase (free movement)
    - grasp_segment[1]: descent phase (constrained, precision required)
    - place_segment[0]: approach phase (free movement)
    - place_segment[1]: descent phase (hanging requires high precision)
    - move_segment: lift/retreat movement

Note: Each segment MUST specify 'enabled' key explicitly.
      segments=None or missing 'enabled' will raise ValueError.
"""

from ._pert_mixin import PerturbationMixin
from .hanging_mug import hanging_mug
from .utils.action import ArmTag
from ._GLOBAL_CONFIGS import GRASP_DIRECTION_DIC


class hanging_mug_pert(PerturbationMixin, hanging_mug):
    """
    Hanging mug task with segment-level perturbation control.

    Task Flow:
        1. Grasp mug with left arm
        2. Lift and place to middle position
        3. Transfer mug to right arm
        4. Hang mug on rack (high precision required)

    Each action is wrapped with explicit segment configuration.
    """

    # =========================================================================
    # Semantic Wrapper Functions
    # =========================================================================

    def _grasp_mug_from_table(self, arm_tag):
        """
        Grasp mug from table surface (first grasp).

        @input:
            arm_tag: str or ArmTag, which arm to use
        @output:
            Tuple[ArmTag, List[Action]], grasp actions with config
        @scenario:
            Initial grasp from table. Approach can have larger perturbation,
            descent requires precision for successful grasp.
        """
        return self._wrap_grasp(
            actor=self.mug,
            arm_tag=arm_tag,
            pre_grasp_dis=0.05,
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
                    "xy_jitter": 0.002,
                    "yaw_jitter_deg": 2.0,
                    "plan_aug_enabled": False,
                },
            ],
        )

    def _grasp_mug_handover(self, arm_tag):
        """
        Grasp mug for handover (from other arm).

        @input:
            arm_tag: str or ArmTag, which arm to use
        @output:
            Tuple[ArmTag, List[Action]], grasp actions with config
        @scenario:
            Grasp from other arm's position. Slightly tighter tolerance
            than table grasp due to mid-air position.
        """
        return self._wrap_grasp(
            actor=self.mug,
            arm_tag=arm_tag,
            pre_grasp_dis=0.05,
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

    def _place_to_middle(self, arm_tag):
        """
        Place mug to intermediate middle position.

        @input:
            arm_tag: str or ArmTag, which arm to use
        @output:
            Tuple[ArmTag, List[Action]], place actions with config
        @scenario:
            Temporary placement for arm transfer. Less precision needed.
        """
        return self._wrap_place(
            actor=self.mug,
            arm_tag=arm_tag,
            target_pose=self.middle_pos,
            pre_dis=0.05,
            dis=0.0,
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
                    "xy_jitter": 0.005,
                    "yaw_jitter_deg": 4.0,
                    "plan_aug_enabled": False,
                },
            ],
        )

    def _place_to_rack(self, arm_tag, target_pose):
        """
        Place mug onto rack (final hanging).

        @input:
            arm_tag: str or ArmTag, which arm to use
            target_pose: List[float], rack functional point
        @output:
            Tuple[ArmTag, List[Action]], place actions with config
        @scenario:
            Final placement on rack. Highest precision required for
            successful hanging. Very small jitter on descent.
        """
        return self._wrap_place(
            actor=self.mug,
            arm_tag=arm_tag,
            target_pose=target_pose,
            functional_point_id=0,
            constrain="align",
            pre_dis=0.05,
            dis=-0.05,
            pre_dis_axis="fp",
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
                    "xy_jitter": 0.001,
                    "yaw_jitter_deg": 1.5,
                    "plan_aug_enabled": False,
                },
            ],
        )

    def _lift_after_grasp(self, arm_tag, z, quat=None, move_axis="world"):
        """
        Lift object after grasp.

        @input:
            arm_tag: str or ArmTag, which arm to use
            z: float, lift distance in meters
            quat: List[float] or None, optional quaternion override
            move_axis: str, "world" or "arm"
        @output:
            Tuple[ArmTag, List[Action]], lift action with config
        @scenario:
            Vertical lift movement. Moderate perturbation acceptable.
        """
        return self._wrap_move(
            arm_tag=arm_tag,
            z=z,
            quat=quat,
            move_axis=move_axis,
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
            Return to home position. Can have perturbation for diversity.
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

    # =========================================================================
    # Override play_once with Semantic Wrappers
    # =========================================================================

    def play_once(self):
        """
        Execute hanging mug task with perturbation.

        Task Flow:
            1. Grasp mug with left arm from table
            2. Lift mug
            3. Place mug at middle position
            4. Lift and transfer to right arm
            5. Grasp mug with right arm
            6. Lift mug
            7. Hang mug on rack
            8. Retreat

        @input: None
        @output: Dict, task info
        @scenario: Complete hanging mug task with segment-level perturbation
        """
        grasp_arm_tag = ArmTag("left")
        hang_arm_tag = ArmTag("right")

        self.move(self._grasp_mug_from_table(grasp_arm_tag))
        self.move(self._lift_after_grasp(grasp_arm_tag, z=0.08))

        self.move(self._place_to_middle(grasp_arm_tag))
        self.move(self._lift_after_grasp(grasp_arm_tag, z=0.1))

        self.move(
            self._back_to_origin(grasp_arm_tag),
            self._grasp_mug_handover(hang_arm_tag),
        )
        self.move(
            self._lift_after_grasp(
                hang_arm_tag, z=0.1, quat=GRASP_DIRECTION_DIC["front"]
            )
        )

        target_pose = self.rack.get_functional_point(0)
        self.move(self._place_to_rack(hang_arm_tag, target_pose))
        self.move(self._lift_after_grasp(hang_arm_tag, z=0.1, move_axis="arm"))

        self.info["info"] = {
            "{A}": f"039_mug/base{self.mug_id}",
            "{B}": "040_rack/base0",
        }

        return self.info
