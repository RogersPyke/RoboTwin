"""
Purpose: Perturbation variant of stack_bowls_three with segment-level control.
Dependencies:
    - envs._pert_mixin.PerturbationMixin
    - envs.stack_bowls_three.stack_bowls_three
Usage Example:
    ./collect_data.sh stack_bowls_three_pert demo_clean_pert <gpu_id>

Segment Design:
    - grasp_segment[0]: approach phase
    - grasp_segment[1]: descent phase
    - place_segment[0]: approach phase (stacking needs precision)
    - place_segment[1]: descent phase (stacking needs high precision)
    - move_segment: lift/retreat movement

Note: Each segment MUST specify 'enabled' key explicitly.
"""

from ._pert_mixin import PerturbationMixin
from .stack_bowls_three import stack_bowls_three
from .utils.action import ArmTag


class stack_bowls_three_pert(PerturbationMixin, stack_bowls_three):
    """
    Stack three bowls task with segment-level perturbation control.

    Task Flow:
        For each bowl:
            1. Grasp bowl from table
            2. Lift bowl
            3. Place bowl on target (stack)
            4. Lift/retreat

    Note: Stacking requires increasing precision for higher positions.
    """

    def _grasp_bowl(self, actor, arm_tag):
        """
        Grasp bowl from table.

        @input:
            actor: Actor, bowl to grasp
            arm_tag: str or ArmTag, which arm to use
        @output:
            Tuple[ArmTag, List[Action]], grasp actions with config
        @scenario:
            Grasp bowl with standard perturbation settings.
        """
        return self._wrap_grasp(
            actor=actor,
            arm_tag=arm_tag,
            contact_point_id=[0, 2][int(arm_tag == "left")],
            pre_grasp_dis=0.1,
            segments=[
                {"enabled": True, "xy_jitter": 0.006, "yaw_jitter_deg": 4.0},
                {"enabled": True, "xy_jitter": 0.002, "yaw_jitter_deg": 2.0},
            ],
        )

    def _place_bowl(self, actor, arm_tag, target_pose):
        """
        Place bowl on stack.

        @input:
            actor: Actor, bowl to place
            arm_tag: str or ArmTag, which arm to use
            target_pose: List[float], target position
        @output:
            Tuple[ArmTag, List[Action]], place actions with config
        @scenario:
            Place bowl on stack with precision for alignment.
        """
        return self._wrap_place(
            actor=actor,
            arm_tag=arm_tag,
            target_pose=target_pose.tolist() + self.quat_of_target_pose,
            functional_point_id=0,
            pre_dis=0.09,
            dis=0,
            constrain="align",
            segments=[
                {"enabled": True, "xy_jitter": 0.005, "yaw_jitter_deg": 3.0},
                {"enabled": True, "xy_jitter": 0.002, "yaw_jitter_deg": 1.5},
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

    def move_bowl(self, actor, target_pose):
        """
        Move a single bowl to target position.

        @input:
            actor: Actor, bowl to move
            target_pose: np.ndarray, target position
        @output:
            str, arm tag used
        @scenario:
            Complete bowl movement with grasp, lift, place, retreat.
        """
        actor_pose = actor.get_pose().p
        arm_tag = ArmTag("left" if actor_pose[0] < 0 else "right")

        if self.las_arm is None or arm_tag == self.las_arm:
            self.move(self._grasp_bowl(actor, arm_tag))
        else:
            self.move(
                self._grasp_bowl(actor, arm_tag),
                self._back_to_origin(arm_tag.opposite),
            )

        self.move(self._lift(arm_tag, z=0.1))
        self.move(self._place_bowl(actor, arm_tag, target_pose))
        self.move(self._lift(arm_tag, z=0.09))

        self.las_arm = arm_tag
        return arm_tag

    def play_once(self):
        """
        Execute stack bowls task with perturbation.

        @input: None
        @output: Dict, task info
        @scenario: Complete stack bowls task with segment-level perturbation
        """
        self.las_arm = None

        self.move_bowl(self.bowl1, self.bowl1_target_pose)
        self.move_bowl(self.bowl2, self.bowl1.get_pose().p + [0, 0, 0.05])
        self.move_bowl(self.bowl3, self.bowl2.get_pose().p + [0, 0, 0.05])

        self.info["info"] = {"{A}": f"002_bowl/base3"}

        result = self.info

        if self._end_reset_to_init:
            self.move(
                self._wrap_back_to_origin(ArmTag("left"), segment={"enabled": False})
            )
            self.move(
                self._wrap_back_to_origin(ArmTag("right"), segment={"enabled": False})
            )

        return result
