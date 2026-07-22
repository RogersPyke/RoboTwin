"""High-reliability left-arm Piper place-object-on-scale task.

This task is intentionally optimized for reliable expert-data collection, not
for equivalence with RoboTwin's original dual-arm task.  It uses separated
Piper arms (0.60 m spacing) and keeps all random placements in the left arm's
validated workspace.  Object and scale assets retain their fixed upright
asset-frame rotation; random roll/pitch is disabled.  The electronic scale is
static so its functional point cannot move during planning.

The expert uses a 4 cm clearance after grasping, then preserves the grasp
orientation while translating the held object's centre to 2.5 cm above the
scale functional point before releasing.  This replaces the generic
``place_actor(..., constrain="free")`` path, whose scale-model frame
differences can generate unreachable end-effector poses.

The standard collection protocol uses the former level-2 expansion from the
first seed onward: object x/y ranges ``[-0.18, -0.075]``/``[-0.11, -0.01]``,
scale x/y ranges ``[-0.32, -0.22]``/``[-0.29, -0.19]``, and bounded yaw in
``[-30, 30]`` degrees.  In the 500-episode validation run, 500/577 seeds
succeeded (86.66%); the longest failure streak was two seeds.
"""

import glob
import os

import numpy as np

from ._base_task import Base_Task
from .utils import *


class place_object_scale_left(Base_Task):
    """Place one randomized object on a scale with the left Piper arm.

    @input: RoboTwin setup kwargs for a left-arm task.
    @output: A task episode with one active arm and one idle arm.
    @scenario: Provide a strict single-arm pick-and-place benchmark while both
        Piper robots remain present in the scene.
    """

    def setup_demo(self, **kwargs):
        """Initialize the left-arm task environment.

        @input: RoboTwin task configuration dictionary.
        @output: None; stores the selected arm and initializes the scene.
        @scenario: Keep the active arm fixed to the left arm.
        """
        self.arm_side = "left"
        super()._init_task_env_(**kwargs)

    def load_actors(self):
        """Create a randomized object and scale on the left side.

        @input: None; uses the seeded RoboTwin random generator.
        @output: None; stores ``object`` and ``scale`` actors.
        @scenario: Keep the original task variation while using the left arm.
        """
        # Standard collection workspace: the former level-2 expansion is now
        # used from the first seed through the full episode budget.
        object_xlim, object_ylim = [-0.18, -0.075], [-0.11, -0.01]
        scale_xlim, scale_ylim = [-0.32, -0.22], [-0.29, -0.19]
        yaw_limit = np.pi / 6

        # This task uses two independently mounted Piper arms (0.60 m apart).
        # Keep the scene in the left arm's well-conditioned workspace rather
        # than merely on the left half of the table.  In particular, y=0.05
        # plus the old 0.15 m lift is outside the reliable IK range.
        # The scale functional point has a positive y offset in its asset
        # frame, and placement approaches it from positive y.  Position the
        # scale nearer the arm base so that its pre-placement pose remains
        # reachable; keep the grasp object forward/right of it.
        rand_pos = rand_pose(
            xlim=object_xlim,
            ylim=object_ylim,
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=True,
            rotate_lim=[0, 0, yaw_limit],
        )

        def get_available_model_ids(modelname):
            """Return available model ids for one RoboTwin object family.

            @input: Object asset directory name.
            @output: Sorted list of integer model ids.
            @scenario: Preserve the original task's object randomization.
            """
            asset_path = os.path.join("assets/objects", modelname)
            ids = []
            for file_path in glob.glob(os.path.join(asset_path, "model_data*.json")):
                name = os.path.basename(file_path)
                try:
                    ids.append(int(name.replace("model_data", "").replace(".json", "")))
                except ValueError:
                    continue
            return sorted(ids)

        object_list = ["047_mouse", "048_stapler", "050_bell"]
        self.selected_modelname = np.random.choice(object_list)
        available_ids = get_available_model_ids(self.selected_modelname)
        if not available_ids:
            raise ValueError(f"No model ids found for {self.selected_modelname}.")
        self.selected_model_id = int(np.random.choice(available_ids))
        self.object = create_actor(
            scene=self,
            pose=rand_pos,
            modelname=self.selected_modelname,
            convex=True,
            model_id=self.selected_model_id,
        )
        self.object.set_mass(0.05)

        target_rand_pose = rand_pose(
            xlim=scale_xlim,
            ylim=scale_ylim,
            qpos=[0.5, 0.5, 0.5, 0.5],
        )
        while np.linalg.norm(target_rand_pose.p[:2] - rand_pos.p[:2]) < 0.15:
            target_rand_pose = rand_pose(
                xlim=scale_xlim,
                ylim=scale_ylim,
                qpos=[0.5, 0.5, 0.5, 0.5],
            )

        self.scale_id = int(np.random.choice([0, 1, 5, 6]))
        self.scale = create_actor(
            scene=self,
            pose=target_rand_pose,
            modelname="072_electronicscale",
            model_id=self.scale_id,
            convex=True,
            # The scale is the fixed placement target.  A dynamic scale can
            # topple and slide before planning finishes, invalidating its
            # functional point and creating unreachable goals.
            is_static=True,
        )
        self.add_prohibit_area(self.object, padding=0.05)
        self.add_prohibit_area(self.scale, padding=0.05)

    def play_once(self):
        """Execute the left-arm pick-and-place trajectory.

        @input: Initialized task scene.
        @output: Task info dictionary.
        @scenario: Move the object to the scale while the opposite arm stays home.
        """
        self.arm_tag = ArmTag(self.arm_side)
        self.move(self.grasp_actor(self.object, arm_tag=self.arm_tag))
        # A small clearance is enough before moving to the placement pre-pose.
        # The former 0.15 m lift made many otherwise valid left-arm grasps
        # unreachable with the separated Piper embodiment.
        self.move(self.move_by_displacement(arm_tag=self.arm_tag, z=0.04))
        if self.plan_success:
            # The scale models have incompatible functional-frame
            # orientations.  ``place_actor(..., constrain="free")`` therefore
            # produces an unreachable wrist pose for some model ids.  Preserve
            # the known-good grasp orientation and translate the held object's
            # centre directly above the scale's functional point instead.
            object_pose = self.object.get_pose().p
            scale_point = np.asarray(self.scale.get_functional_point(0), dtype=np.float64)[:3]
            target_object_pose = scale_point.copy()
            target_object_pose[2] += 0.025
            end_effector_pose = np.asarray(self.get_arm_pose(self.arm_tag), dtype=np.float64).reshape(-1)
            transfer_pose = end_effector_pose.copy()
            transfer_pose[:3] += target_object_pose - object_pose
            self.move(self.move_to_pose(self.arm_tag, transfer_pose))

        if self.plan_success:
            self.move(self.open_gripper(self.arm_tag))
        self.info["info"] = {
            "{A}": f"072_electronicscale/base{self.scale_id}",
            "{B}": f"{self.selected_modelname}/base{self.selected_model_id}",
            "{a}": self.arm_side,
        }
        return self.info

    def check_success(self):
        """Check placement and active-arm release conditions.

        @input: Current task scene state.
        @output: Boolean success indicator.
        @scenario: Match the original task's placement criterion.
        """
        object_pose = self.object.get_pose().p
        scale_pose = self.scale.get_functional_point(0)
        distance = np.linalg.norm(np.asarray(scale_pose[:2]) - np.asarray(object_pose[:2]))
        check_arm = self.is_left_gripper_open if self.arm_side == "left" else self.is_right_gripper_open
        return bool(distance < 0.035 and object_pose[2] > scale_pose[2] - 0.01 and check_arm())
