"""LEFT_TASK_DESIGN:
Source task: place_burger_fries
Single-arm semantic change: source grasps hamburg with the left arm and
    french fries with the right arm concurrently and places both onto the tray
    in one parallel group; the derived task fixes the tray inside the left
    workspace and serially transfers each food with the left arm, returning
    home between transfers.
Left workspace manifest: place_burger_fries, version 2 (2026-08-13)
Actors and clearance: tray (static, min 0.08 m), hamburg (dynamic, y in
    [0.10,0.16], x in [-0.46,-0.34], min 0.08 m), frenchfries (dynamic, y in
    [-0.35,-0.27], x in [-0.46,-0.34], min 0.08 m).
Expert sequence: left grasp hamburg, lift z=0.10, place at tray functional
    point zero, release, return home; left grasp frenchfries, place at tray
    functional point one, release, return home.
Success predicate: preserve both tray-functional-point containment distances
    (0.08 m) and the left-open condition; replace the right-gripper condition
    with right-home immobility.
Instruction change: {A}=hamburg, {B}=tray, {C}=frenchfries, {a}=left; wording
    says the left arm places the hamburger and the french fries into the tray.
Pilot evidence: central-cam 30-seed pilot on version 0 0.60 (18/30 success),
    manifest hash 4bb8c437289fcdf5; version 2 (widened food pickup bands)
    50-seed pilot 0.44 (22/50 success), manifest hash dd5bbabf4e2616cf
"""

from __future__ import annotations

import numpy as np
import sapien

from .left_task_base import LeftArmTaskError, LeftTaskBase, SceneRejectedError
from .left_task_manifests import get_manifest
from ..utils import *  # noqa: F401,F403

# ---------------------------------------------------------------------------
# LEGACY_RANDOMIZATION_PARAM
# The randomization protocol used before the current version 2 design.
# Intentionally unused: kept as a declared header constant so the previous
# geometry is reproducible and comparable.
#   Version 0 (original source-task geometry, manifest 4bb8c437289fcdf5):
#     - tray:      static x in [-0.44, -0.08], y in [-0.16, -0.04], yaw 0,
#                  scale 2.0, clearance 0.08 m
#     - hamburg:   x in [-0.44, -0.36], y in [0.10, 0.14], yaw in [0, 0.20],
#                  clearance 0.08 m
#     - fries:     x in [-0.44, -0.36], y in [-0.33, -0.27], yaw in [0, 0.20],
#                  clearance 0.08 m
LEGACY_RANDOMIZATION_PARAM: dict[str, object] = {
    "tray_x": (-0.44, -0.08),
    "tray_y": (-0.16, -0.04),
    "tray_yaw_rad": (0.0, 0.00),
    "tray_clearance_m": 0.08,
    "hamburg_x": (-0.44, -0.36),
    "hamburg_y": (0.10, 0.14),
    "hamburg_yaw_rad": (0.0, 0.20),
    "fries_x": (-0.44, -0.36),
    "fries_y": (-0.33, -0.27),
    "fries_yaw_rad": (0.0, 0.20),
    "food_clearance_m": 0.08,
    "manifest_hash": "4bb8c437289fcdf5",
}


class PlaceBurgerFriesLeftImpl(LeftTaskBase):
    """Left-arm-only hamburger-and-fries-into-tray task.

    The tray is static in the left workspace; the left arm transfers the
    hamburger then the french fries onto two tray functional points serially.
    """

    def __init__(self) -> None:
        super().__init__()
        self.manifest = get_manifest("place_burger_fries")

    def sample_layout(self) -> dict[str, object]:
        specs = {name: self.manifest.actor_specs[name] for name in
                 ("tray", "hamburg", "frenchfries")}
        clearance = max(spec.minimum_clearance_m for spec in specs.values())
        for _ in range(128):
            tray = self.sample_spec_pose(specs["tray"])
            hamburg = self.sample_spec_pose(specs["hamburg"])
            frenchfries = self.sample_spec_pose(specs["frenchfries"])
            if (self._clear(hamburg, frenchfries, clearance)
                    and self._clear(hamburg, tray, clearance)
                    and self._clear(frenchfries, tray, clearance)):
                return {"tray": tray, "hamburg": hamburg, "frenchfries": frenchfries}
        raise SceneRejectedError(
            "could not sample a feasible place_burger_fries layout in 128 attempts"
        )

    @staticmethod
    def _clear(pose_a, pose_b, clearance: float) -> bool:
        return bool(np.linalg.norm(pose_a.p[:2] - pose_b.p[:2]) >= clearance)

    def load_actors(self) -> None:
        layout = self.sample_layout()
        if not self.validate_layout(layout):
            raise SceneRejectedError("place_burger_fries layout rejected by validate_layout")
        self.tray_id = int(np.random.choice([0, 1, 2, 3]))
        self.tray = create_actor(
            scene=self, pose=layout["tray"], modelname="008_tray", convex=True,
            model_id=self.tray_id, scale=(2.0, 2.0, 2.0), is_static=True,
        )
        self.tray.set_mass(0.05)
        # Model 0 is the only hamburg hull that yields a reachable downward
        # grasp from both its flat and its tipped settle poses.  Models 3 and 5
        # fail the grasp pre-flight on every sampled seed, model 4 is
        # unreliable (2 of 3 seeds fail) and model 1 tips upright on 2 of 11
        # seeds into a pose with no reachable grasp; their flat VHACD hulls sit
        # at the curobo near-table singularity with no contact point that
        # resolves to a plan path.  Restricting the selection keeps the grasp
        # deterministic.
        self.object1_id = 0
        # Spawn the foods slightly above the table so the settle phase drops
        # them cleanly; a spawn z flush with the table can tunnel the bottom
        # contact hull through the surface and launch the actor away.
        hamburg_pose = layout["hamburg"]
        hamburg_p = list(hamburg_pose.p)
        hamburg_p[2] += 0.04
        self.hamburg = create_actor(
            scene=self, pose=sapien.Pose(hamburg_p, hamburg_pose.q),
            modelname="006_hamburg", convex=True, model_id=self.object1_id,
        )
        self.hamburg.set_mass(0.05)
        self.object2_id = int(np.random.choice([0, 1]))
        # Spawn the fries slightly above the table so the settle phase drops
        # them cleanly; a spawn z flush with the table can tunnel the bottom
        # contact hull through the surface and launch the actor away.
        fries_pose = layout["frenchfries"]
        fries_p = list(fries_pose.p)
        fries_p[2] += 0.04
        self.frenchfries = create_actor(
            scene=self, pose=sapien.Pose(fries_p, fries_pose.q),
            modelname="005_french-fries", convex=True, model_id=self.object2_id,
        )
        self.frenchfries.set_mass(0.5)
        self.add_prohibit_area(self.tray, padding=0.1)
        self.add_prohibit_area(self.hamburg, padding=0.05)
        self.add_prohibit_area(self.frenchfries, padding=0.05)
        self.record_layout(layout)
        tray0 = self.tray.get_functional_point(0)[:3]
        tray1 = self.tray.get_functional_point(1)[:3]
        self.route_bins = self.classify_routes([
            ("hamburg", np.asarray(layout["hamburg"].p), np.asarray(tray0)),
            ("frenchfries", np.asarray(layout["frenchfries"].p), np.asarray(tray1)),
        ])

    def place_food_left(self, food, point_id: int) -> bool:
        """Transfer one food item to a tray functional point and return home.

        @input: the food actor and the tray functional point index.
        @output: boolean plan success.
        @scenario: Serial left-arm transfer with home reset between items.
        """
        if point_id not in (0, 1):
            raise LeftArmTaskError(f"tray functional point id must be 0 or 1, got {point_id!r}")
        if not self.plan_success:
            return False
        arm_tag = ArmTag("left")
        # Pre-flight the grasp: choose_grasp_pose returns (None, None) when no
        # contact point yields a reachable grasp, and grasp_actor then raises
        # an AssertionError on a None target pose.  Surface the unreachable case
        # as a normal plan failure instead of aborting the episode.
        pre_pose, grasp_pose = self.choose_grasp_pose(food, arm_tag=arm_tag, pre_dis=0.1)
        if pre_pose is None or grasp_pose is None:
            self.plan_success = False
            return False
        self.move(self.grasp_actor(food, arm_tag=arm_tag, pre_grasp_dis=0.1))
        if not self.plan_success:
            return False
        self.move_by_displacement(arm_tag=arm_tag, z=0.1)
        if not self.plan_success:
            return False
        target_pose = self.tray.get_functional_point(point_id)
        self.move(self.place_actor(food, arm_tag=arm_tag, target_pose=target_pose,
                                   functional_point_id=0, constrain="free",
                                   pre_dis=0.1, pre_dis_axis="fp"))
        if not self.plan_success:
            return False
        self.move_by_displacement(arm_tag=arm_tag, z=0.08)
        return self.left_return_home()

    def play_once(self) -> dict:
        self.place_food_left(self.hamburg, 0)
        self.place_food_left(self.frenchfries, 1)
        self.info["info"] = {
            "{A}": f"006_hamburg/base{self.object1_id}",
            "{B}": f"008_tray/base{self.tray_id}",
            "{C}": f"005_french-fries/base{self.object2_id}",
            "{a}": "left",
        }
        return self.info

    def check_success(self) -> bool:
        dis1 = float(np.linalg.norm(
            self.tray.get_functional_point(0, "pose").p[0:2]
            - self.hamburg.get_functional_point(0, "pose").p[0:2]
        ))
        dis2 = float(np.linalg.norm(
            self.tray.get_functional_point(1, "pose").p[0:2]
            - self.frenchfries.get_functional_point(0, "pose").p[0:2]
        ))
        threshold = 0.08
        return dis1 < threshold and dis2 < threshold and self.is_left_gripper_open() \
            and self.right_arm_stationary()
