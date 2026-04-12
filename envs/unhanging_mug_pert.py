"""
Purpose: Perturbation variant of unhanging_mug with segment-level control.
Dependencies:
    - envs._pert_mixin.PerturbationMixin
    - envs.unhanging_mug.unhanging_mug
Usage Example:
    ./collect_data.sh unhanging_mug_pert demo_clean_pert <gpu_id>

Segment Design:
    - grasp_segment[0]: approach phase
    - grasp_segment[1]: descent phase
    - place_segment[0]: approach phase
    - place_segment[1]: descent phase
    - move_segment: lift/retreat movement

Note: Each segment MUST specify 'enabled' key explicitly.
      Unhanging requires high precision to avoid collision with rack.
"""

from ._pert_mixin import PerturbationMixin
from .unhanging_mug import unhanging_mug, OFF_RACK_DIST
from .utils.action import ArmTag


class unhanging_mug_pert(PerturbationMixin, unhanging_mug):
    """
    Unhanging mug task with segment-level perturbation control.

    Task Flow:
        1. Hang arm (right) grasps mug from rack
        2. Move mug off rack along rack fp axis
        3. Hang arm places mug at middle_pos
        4. Handoff: hang arm back to origin, grasp arm grasps mug
        5. Grasp arm (left) places mug at final target
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
                {"enabled": True, "xy_jitter": 0.005, "yaw_jitter_deg": 4.0},
                {"enabled": True, "xy_jitter": 0.001, "yaw_jitter_deg": 1.5},
            ],
        )

    def _grasp_mug_from_middle(self, arm_tag):
        """
        Grasp mug from middle position.

        @input:
            arm_tag: str or ArmTag, which arm to use
        @output:
            Tuple[ArmTag, List[Action]], grasp actions with config
        @scenario:
            Grasp mug from middle position during handoff.
        """
        return self._wrap_grasp(
            actor=self.mug,
            arm_tag=arm_tag,
            pre_grasp_dis=0.05,
            segments=[
                {"enabled": True, "xy_jitter": 0.008, "yaw_jitter_deg": 6.0},
                {"enabled": True, "xy_jitter": 0.002, "yaw_jitter_deg": 2.0},
            ],
        )

    def _place_off_rack(self, arm_tag, target_pose):
        """
        Place mug off rack along rack fp axis.

        @input:
            arm_tag: str or ArmTag, which arm to use
            target_pose: List[float], off-rack target pose
        @output:
            Tuple[ArmTag, List[Action]], place actions with config
        @scenario:
            Move mug off rack along rack functional point axis.
            High precision required to avoid collision.

        IMPORTANT: Both segments MUST have enabled=True.
        Moving along rack functional point axis with enabled=False
        would bypass trajectory perturbation and cause collision with rack.
        """
        return self._wrap_place(
            actor=self.mug,
            arm_tag=arm_tag,
            target_pose=target_pose,
            functional_point_id=0,
            constrain="align",
            pre_dis=0.0,
            dis=0.0,
            is_open=False,
            pre_dis_axis="fp",
            segments=[
                {"enabled": True, "xy_jitter": 0.003, "yaw_jitter_deg": 2.0},
                {"enabled": True, "xy_jitter": 0.001, "yaw_jitter_deg": 1.0},
            ],
        )

    def _place_to_middle(self, arm_tag):
        """
        Place mug at middle position.

        @input:
            arm_tag: str or ArmTag, which arm to use
        @output:
            Tuple[ArmTag, List[Action]], place actions with config
        @scenario:
            Place mug at intermediate middle position.
        """
        return self._wrap_place(
            actor=self.mug,
            arm_tag=arm_tag,
            target_pose=self.middle_pos,
            pre_dis=0.05,
            dis=0.0,
            constrain="free",
            segments=[
                {"enabled": True, "xy_jitter": 0.010, "yaw_jitter_deg": 8.0},
                {"enabled": True, "xy_jitter": 0.003, "yaw_jitter_deg": 3.0},
            ],
        )

    def _place_to_target(self, arm_tag, target_pose):
        """
        Place mug at final target position.

        @input:
            arm_tag: str or ArmTag, which arm to use
            target_pose: List[float], final target pose
        @output:
            Tuple[ArmTag, List[Action]], place actions with config
        @scenario:
            Place mug at final target on table.
        """
        return self._wrap_place(
            actor=self.mug,
            arm_tag=arm_tag,
            target_pose=target_pose,
            pre_dis=0.05,
            dis=0.0,
            constrain="free",
            segments=[
                {"enabled": True, "xy_jitter": 0.010, "yaw_jitter_deg": 8.0},
                {"enabled": True, "xy_jitter": 0.003, "yaw_jitter_deg": 3.0},
            ],
        )

    def _lift(self, arm_tag, z, move_axis="world"):
        """
        Lift mug.

        @input:
            arm_tag: str or ArmTag, which arm to use
            z: float, lift distance
            move_axis: str, "world" or "arm"
        @output:
            Tuple[ArmTag, List[Action]], lift action with config
        @scenario:
            Vertical lift movement.
        """
        return self._wrap_move(
            arm_tag=arm_tag,
            z=z,
            move_axis=move_axis,
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

    def play_once(self):
        """
        Execute unhanging mug task with perturbation.

        Task Flow:
            1. Hang arm (right) grasps mug from rack
            2. Move mug off rack along rack fp axis
            3. Hang arm places mug at middle_pos
            4. Handoff: hang arm back to origin, grasp arm grasps mug
            5. Grasp arm (left) places mug at final target

        @input: None
        @output: Dict, task info
        @scenario: Complete unhanging mug task with segment-level perturbation
        """
        grasp_arm_tag = ArmTag("left")
        hang_arm_tag = ArmTag("right")
        self._grasp_arm_tag = grasp_arm_tag
        self._hang_arm_tag = hang_arm_tag

        if getattr(self, "skip_robot_movement", False):
            self.info["info"] = {
                "{A}": f"039_mug/base{self.mug_id}",
                "{B}": "040_rack/base0",
            }
            save_data_orig = self.save_data
            self.save_data = True
            n = getattr(self, "skip_success_check_video_frames", 100)
            for _ in range(n):
                self._take_picture()
            self.merge_pkl_to_hdf5_video()
            self.save_data = save_data_orig
            self.plan_success = True
            return self.info

        self.move(self._grasp_mug_from_rack(hang_arm_tag))

        off_rack_target = self._get_off_rack_place_pose(OFF_RACK_DIST)
        self.move(self._place_off_rack(hang_arm_tag, off_rack_target))
        self.move(self._lift(hang_arm_tag, z=0.1, move_axis="world"))

        self.move(self._place_to_middle(hang_arm_tag))
        self.move(self._lift(hang_arm_tag, z=0.1))

        self.move(
            self._back_to_origin(hang_arm_tag),
            self._grasp_mug_from_middle(grasp_arm_tag),
        )
        self.move(self._lift(grasp_arm_tag, z=0.1))

        self.move(self._place_to_target(grasp_arm_tag, self.mug_target_pose))
        self.move(self._lift(grasp_arm_tag, z=0.1))

        self.info["info"] = {
            "{A}": f"039_mug/base{self.mug_id}",
            "{B}": "040_rack/base0",
        }

        if getattr(self, "skip_success_check", False):
            self.plan_success = True

        result = self.info

        if self._end_reset_to_init:
            self.move(
                self._wrap_back_to_origin(ArmTag("left"), segment={"enabled": False})
            )
            self.move(
                self._wrap_back_to_origin(ArmTag("right"), segment={"enabled": False})
            )

        return result
