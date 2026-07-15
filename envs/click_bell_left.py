"""Single-left-Piper bell press task used by Embodied Mixup experiments."""

from ._base_task import Base_Task
from .utils import *


class click_bell_left(Base_Task):
    """Press a bell using only the left robot arm.

    @input: RoboTwin task setup arguments including a dual-Piper embodiment.
    @output: A task episode where only the left Piper is required to move.
    @scenario: Provide a deterministic single-arm benchmark while RoboTwin
        still instantiates two robot entities.
    """

    def setup_demo(self, **kwargs):
        """Initialize one task episode.

        @input: kwargs [dict] RoboTwin task configuration values.
        @output: None.
        @scenario: Delegate scene construction to the RoboTwin base task.
        """
        super()._init_task_env_(**kwargs)

    def load_actors(self):
        """Create one bell inside the left Piper workspace.

        @input: None.
        @output: None.
        @scenario: Prevent the expert planner from selecting the right arm.
        """
        rand_pos = rand_pose(
            xlim=[-0.25, -0.05],
            ylim=[-0.20, 0.0],
            qpos=[0.5, 0.5, 0.5, 0.5],
        )
        self.bell_id = np.random.choice([0, 1], 1)[0]
        self.bell = create_actor(
            scene=self,
            pose=rand_pos,
            modelname="050_bell",
            convex=True,
            model_id=self.bell_id,
            is_static=True,
        )
        self.add_prohibit_area(self.bell, padding=0.07)

    def play_once(self):
        """Generate the left-arm expert demonstration.

        @input: None.
        @output: dict task metadata.
        @scenario: Press the bell without commanding the right Piper.
        """
        arm_tag = ArmTag("left")
        self.move(
            self.grasp_actor(
                self.bell,
                arm_tag=arm_tag,
                pre_grasp_dis=0.1,
                grasp_dis=0.1,
                contact_point_id=0,
            )
        )
        self.move(self.move_by_displacement(arm_tag, z=-0.045))
        self.check_success()
        self.move(self.move_by_displacement(arm_tag, z=0.045))
        self.info["info"] = {"{A}": f"050_bell/base{self.bell_id}", "{a}": "left"}
        return self.info

    def check_success(self):
        """Return whether the left gripper has pressed the bell.

        @input: None.
        @output: bool success flag.
        @scenario: Exclude the inactive right Piper from task success.
        """
        if self.stage_success_tag:
            return True
        if not self.is_left_gripper_close():
            return False
        bell_pose = self.bell.get_contact_point(0)[:3]
        for position in self.get_gripper_actor_contact_position("050_bell"):
            if np.all(np.abs(position[:2] - bell_pose[:2]) < [0.025, 0.025]) and abs(position[2] - bell_pose[2]) < 0.03:
                self.stage_success_tag = True
                return True
        return False
