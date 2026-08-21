"""LEFT_TASK_DESIGN:
Source task: place_dual_shoes
Single-arm semantic change: source grasps the two shoes concurrently with the
    dual arms and drops them into the box in one parallel group; the derived
    task fixes the shoe box static in the left workspace and serially places
    the left shoe at functional point zero and the right shoe at functional
    point one with the left arm, returning home between placements.
Left workspace manifest: place_dual_shoes, version 3 (2026-08-14)
Actors and clearance: shoe_box (static, union sweep x in [-0.46,-0.14], y in
    [-0.18,-0.02] so the box x band matches the full shoe x band, footprint-aware
    clearance min 0.13 m), left_shoe/right_shoe (dynamic, min 0.10 m each)
    sharing the same reachable envelope x in [-0.46,-0.14] at y in [0.13,0.14],
    yaw up to 0.50.  A physical-reset grasp+lift probe showed the left arm can
    lift a shoe only on the y=0.13 row (two islands x[-0.46,-0.40] and
    x[-0.20,-0.14]) or the y=0.14 row (one continuous band x[-0.44,-0.18]); the
    impl snaps y to one verified row and samples x inside that row's lift-OK
    set so every spawned shoe is both graspable and liftable, then enforces the
    pairwise/basket clearance.  The box y half-extent keeps its footprint top at
    y<=0.11, below the shoe rows' y=0.13.
Expert sequence: left grasp left shoe, lift z=0.15, align into box functional
    point zero, return home; left grasp right shoe, align into functional
    point one, return home; brief settling delay.
Success predicate: replace the source hardcoded world target with a
    shoe-box-relative predicate: each shoe at its box functional point, shoe
    orientation aligned with the box orientation, shoe height equal to box
    height plus 0.01 m; require left open and right home.
Instruction change: {A}=shoe, {B}=shoe box, {a}=left; wording says the left
    arm puts both shoes into the stationary box.
Pilot evidence: central-cam 30-seed pilot on version 0 0.27 (8/30 success),
    manifest hash 5ab2644b7fc17bd9; version 2 (shared reachable envelope)
    50-seed pilot 0.40 (20/50 success), manifest hash 735e2fa6c8f00a5b; version
    3 (union box x sweep) 50-seed pilot 0.36 (18/50 success), manifest hash
    ce583ff0cacdbd38; the 12 setup failures are UnStableError 041_shoe topples
    across the wider envelope, not GPU contamination, and the pilot still
    clears the 10/50 bar.
"""

from __future__ import annotations

import numpy as np

from ..left_task_base import CENTERED_BATCH_VARIANTS, LeftTaskBase, SceneInitRejectError
from ..left_task_manifests import get_manifest
from ...utils import *  # noqa: F401,F403

# ---------------------------------------------------------------------------
# LEGACY_RANDOMIZATION_PARAM
# The randomization protocol used before the current version 2 design.
# Intentionally unused: kept as a declared header constant so the previous
# geometry is reproducible and comparable.
#   Version 0 (original geometry, manifest 5ab2644b7fc17bd9):
#     - left_shoe:  x in [-0.44, -0.40], y in [0.13, 0.14], yaw in [0, 0.50]
#     - right_shoe: x in [-0.20, -0.16], y in [0.13, 0.14], yaw in [0, 0.50]
#     - shoe_box:   static x in [-0.44, -0.16], y in [-0.18, -0.02], yaw 0
LEGACY_RANDOMIZATION_PARAM: dict[str, object] = {
    "shoe_box_x": (-0.44, -0.16),
    "shoe_box_y": (-0.18, -0.02),
    "shoe_box_yaw_rad": (0.0, 0.00),
    "left_shoe_x": (-0.44, -0.40),
    "right_shoe_x": (-0.20, -0.16),
    "shoe_y": (0.13, 0.14),
    "shoe_yaw_rad": (0.0, 0.50),
    "shoe_clearance_m": 0.10,
    "manifest_hash": "5ab2644b7fc17bd9",
}


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
        # Version 2: the manifest declares both shoes in the shared reachable
        # envelope x[-0.46,-0.14], but a physical-reset grasp+lift probe showed
        # the left arm can lift a shoe only on two discrete y rows, with the
        # lift-OK x band depending on the row:
        #   y=0.13: two islands x[-0.46,-0.40] and x[-0.20,-0.14]
        #   y=0.14: one continuous band x[-0.44,-0.18]
        # Snap y to one verified row, then sample each shoe inside that row's
        # union of lift-OK bands so both shoes share the whole appearance range
        # (maximal overlap) while staying graspable AND liftable.  The 0.10 m
        # pairwise/basket clearance keeps the two shoes from never overlapping.
        centered_batch = self.camera_variant in CENTERED_BATCH_VARIANTS
        centered_x_offset = 0.30 if centered_batch else 0.0
        shoe_y = float(np.random.choice([0.13, 0.14]))
        lift_ok_bands = (
            ((-0.46 + centered_x_offset, -0.40 + centered_x_offset),
             (-0.20 + centered_x_offset, -0.14 + centered_x_offset))
            if shoe_y < 0.135
            else ((-0.44 + centered_x_offset, -0.18 + centered_x_offset),)
        )

        def _shoe_pose(spec) -> sapien.Pose:
            base = self.sample_spec_pose(spec)
            lo, hi = lift_ok_bands[int(np.random.rand() * len(lift_ok_bands))]
            p = np.asarray(base.p, dtype=float)
            p[0] = float(np.random.uniform(lo, hi))
            p[1] = shoe_y
            return sapien.Pose(p, base.q)

        for _ in range(256):
            shoe_box = self.sample_spec_pose(specs["shoe_box"])
            if centered_batch:
                # Every actor starts from the identical centred-batch envelope.
                # This placement resolver only rejects/remaps physically invalid
                # candidates: the static box must remain below the shoes' stable
                # row so its rim cannot intersect either curved sole.
                p = np.asarray(shoe_box.p, dtype=float)
                p[1] = float(np.random.uniform(-0.15, -0.02))
                shoe_box = sapien.Pose(p, shoe_box.q)
            left_shoe = _shoe_pose(specs["left_shoe"])
            right_shoe = _shoe_pose(specs["right_shoe"])
            if (self._clear(left_shoe, right_shoe, clearance)
                    and self._clear(left_shoe, shoe_box, clearance)
                    and self._clear(right_shoe, shoe_box, clearance)):
                return {"shoe_box": shoe_box, "left_shoe": left_shoe,
                        "right_shoe": right_shoe}
        raise SceneInitRejectError(
            "could not sample a feasible place_dual_shoes layout in 256 attempts"
        )

    @staticmethod
    def _clear(pose_a, pose_b, clearance: float) -> bool:
        return bool(np.linalg.norm(pose_a.p[:2] - pose_b.p[:2]) >= clearance)

    def load_actors(self) -> None:
        layout = self.sample_layout()
        if not self.validate_layout(layout):
            raise SceneInitRejectError("place_dual_shoes layout rejected by validate_layout")
        # Box origin sits exactly on the table top (0.741); the static box
        # never settles, so the old 0.74 left it permanently 1 mm buried.
        layout["shoe_box"].p[2] = 0.741
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
