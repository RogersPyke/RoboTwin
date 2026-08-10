"""LEFT_TASK_DESIGN:
Source task: place_dual_shoes
Single-arm semantic change: source grasps the two shoes concurrently with the
    dual arms and drops them into the box in one parallel group; the derived
    task fixes the shoe box static in the left workspace and serially places
    the left shoe at functional point zero and the right shoe at functional
    point one with the left arm, returning home between placements.
Left workspace manifest: place_dual_shoes, version 0
Actors and clearance: shoe_box (static, min 0.10 m), left_shoe/right_shoe
    (dynamic, min 0.10 m each).
Expert sequence: left grasp left shoe, lift z=0.15, align into box functional
    point zero, return home; left grasp right shoe, align into functional
    point one, return home; brief settling delay.
Success predicate: replace the source hardcoded world target with a
    shoe-box-relative predicate: each shoe at its box functional point, shoe
    orientation aligned with the box orientation, shoe height equal to box
    height plus 0.01 m; require left open and right home.
Instruction change: {A}=shoe, {B}=shoe box, {a}=left; wording says the left
    arm puts both shoes into the stationary box.
Pilot evidence: central-cam 30-seed pilot 0.27 (8/30 success), manifest hash
    5ab2644b7fc17bd9
"""

from __future__ import annotations

import numpy as np

from .left_task_base import LeftTaskBase, SceneRejectedError
from .left_task_manifests import get_manifest
from ..utils import *  # noqa: F401,F403


class PlaceDualShoesLeftImpl(LeftTaskBase):
    """Left-arm-only two-shoes-into-box task with a stationary shoe box.

    Both shoes are transferred serially by the left arm into the two box
    functional points.  The success predicate is expressed relative to the box
    pose rather than in hardcoded world coordinates.
    """

    def __init__(self) -> None:
        super().__init__()
        self.manifest = get_manifest("place_dual_shoes")

    def sample_layout(self) -> dict[str, object]:
        specs = {name: self.manifest.actor_specs[name] for name in
                 ("shoe_box", "left_shoe", "right_shoe")}
        clearance = max(spec.minimum_clearance_m for spec in specs.values())
        for _ in range(128):
            shoe_box = self.sample_spec_pose(specs["shoe_box"])
            left_shoe = self.sample_spec_pose(specs["left_shoe"])
            right_shoe = self.sample_spec_pose(specs["right_shoe"])
            if (self._clear(left_shoe, right_shoe, clearance)
                    and self._clear(left_shoe, shoe_box, clearance)
                    and self._clear(right_shoe, shoe_box, clearance)):
                return {"shoe_box": shoe_box, "left_shoe": left_shoe,
                        "right_shoe": right_shoe}
        raise SceneRejectedError(
            "could not sample a feasible place_dual_shoes layout in 128 attempts"
        )

    @staticmethod
    def _clear(pose_a, pose_b, clearance: float) -> bool:
        return bool(np.linalg.norm(pose_a.p[:2] - pose_b.p[:2]) >= clearance)

    def load_actors(self) -> None:
        layout = self.sample_layout()
        if not self.validate_layout(layout):
            raise SceneRejectedError("place_dual_shoes layout rejected by validate_layout")
        # Box origin sits on the table top; the shoes rest just above it.
        layout["shoe_box"].p[2] = 0.74
        self.shoe_box = create_actor(
            scene=self, pose=layout["shoe_box"], modelname="007_shoe-box",
            convex=True, is_static=True,
        )
        # Model 4 is the only 041_shoe hull that settles stably across the
        # whole left-workspace spawn grid at every allowed yaw.  The standing
        # shoes sit on a curved sole, so most models topple or rock during the
        # settle phase (UnStableError) instead of coming to rest; model 0 is
        # stable in the left band but topples in the right band at yaw above
        # ~0.25.  Restricting the selection keeps the spawn deterministic.
        self.shoe_id = 4
        self.left_shoe = create_actor(
            scene=self, pose=layout["left_shoe"], modelname="041_shoe",
            convex=True, model_id=self.shoe_id,
        )
        self.right_shoe = create_actor(
            scene=self, pose=layout["right_shoe"], modelname="041_shoe",
            convex=True, model_id=self.shoe_id,
        )
        self.add_prohibit_area(self.left_shoe, padding=0.02)
        self.add_prohibit_area(self.right_shoe, padding=0.02)
        self.add_prohibit_area(self.shoe_box, padding=0.05)
        self.record_layout(layout)
        fp0 = self.shoe_box.get_functional_point(0)[:3]
        fp1 = self.shoe_box.get_functional_point(1)[:3]
        self.route_bins = self.classify_routes([
            ("left_shoe", np.asarray(layout["left_shoe"].p), np.asarray(fp0)),
            ("right_shoe", np.asarray(layout["right_shoe"].p), np.asarray(fp1)),
        ])

    def place_shoe_left(self, shoe, point_id: int) -> bool:
        """Grasp, lift and align one shoe into one box functional point.

        @input: the shoe actor and the target functional point index.
        @output: boolean plan success.
        @scenario: Serial left-arm transfer with home reset between shoes.
        """
        if not self.plan_success:
            return False
        arm_tag = ArmTag("left")
        self.move(self.grasp_actor(shoe, arm_tag=arm_tag, pre_grasp_dis=0.1))
        if not self.plan_success:
            return False
        self.move_by_displacement(arm_tag=arm_tag, z=0.15)
        if not self.plan_success:
            return False
        target_pose = self.shoe_box.get_functional_point(point_id)
        self.move(self.place_actor(shoe, arm_tag=arm_tag, target_pose=target_pose,
                                   functional_point_id=0, pre_dis=0.07,
                                   dis=0.02, constrain="align"))
        if not self.plan_success:
            return False
        self.move_by_displacement(arm_tag=arm_tag, z=0.1, move_axis="arm")
        return self.left_return_home()

    def play_once(self) -> dict:
        self.place_shoe_left(self.left_shoe, 0)
        self.place_shoe_left(self.right_shoe, 1)
        self.delay(3)
        self.info["info"] = {
            "{A}": f"041_shoe/base{self.shoe_id}",
            "{B}": "007_shoe-box/base0",
            "{a}": "left",
        }
        return self.info

    def _shoe_aligned(self, shoe, target_p: np.ndarray) -> bool:
        """Return whether one shoe sits aligned at a box functional point.

        @input: the shoe actor and the three-value functional point position.
        @output: boolean; xy within 0.05 m, orientation within 0.07, height
            within 0.03 m of the box top.
        @scenario: Pose-relative success so a shifted box never fails a shoe.
        """
        pose = shoe.get_pose()
        p = np.asarray(pose.p, dtype=float)
        q = np.asarray(pose.q, dtype=float)
        if q[0] < 0:
            q = -q
        target_q = np.asarray(self.shoe_box.get_pose().q, dtype=float)
        eps = np.array([0.05, 0.05, 0.07, 0.07, 0.07, 0.07])
        box_z = float(self.shoe_box.get_pose().p[2])
        return (
            bool(np.all(abs(p[:2] - target_p[:2]) < eps[:2]))
            and bool(np.all(abs(q - target_q) < eps[-4:]))
            and bool(abs(p[2] - (box_z + 0.01)) < 0.03)
        )

    def check_success(self) -> bool:
        fp0 = np.asarray(self.shoe_box.get_functional_point(0)[:3], dtype=float)
        fp1 = np.asarray(self.shoe_box.get_functional_point(1)[:3], dtype=float)
        return (
            self._shoe_aligned(self.left_shoe, fp0)
            and self._shoe_aligned(self.right_shoe, fp1)
            and self.is_left_gripper_open()
            and self.right_arm_stationary()
        )
