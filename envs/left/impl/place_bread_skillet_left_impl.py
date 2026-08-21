"""LEFT_TASK_DESIGN:
Source task: place_bread_skillet
Single-arm semantic change: source puts bread and skillet on opposite sides
    and grasps both concurrently with dual arms; the derived task initialises
    the skillet as a static target directly in the left workspace and performs
    only a left bread grasp-lift-place-release-retreat sequence.
Left workspace manifest: place_bread_skillet, version 3 (2026-08-14)
Actors and clearance: bread (dynamic, x in [-0.44,-0.34], y in [-0.18,0.04],
    yaw up to 0.50), skillet (static, union sweep x in [-0.44,-0.12], y in
    [-0.24,0.14] covering the whole family appearance envelope so the skillet
    overlaps the bread band, spawned at table-contact height with no raise,
    footprint-aware clearance min 0.17 m).  The 0.17 m skillet clearance keeps
    the bread centre outside the 106_skillet hull.  Version 4 removed the
    historical +0.025 m raise: a static skillet never settles, so the raise
    left it floating ~2 cm above the table in every frame; the success height
    gates were recalibrated to contact instead.
Expert sequence: left grasp bread, lift z=0.10, place at skillet functional
    point zero, release; a post-place retreat that cannot plan is skipped and
    the arm returns home.
Success predicate: preserve the bread-to-skillet functional-point distance and
    height tests; add left-open and right-home checks.  No skillet-moving
    helper exists and the skillet is never moved.
Instruction change: {A}=skillet, {B}=bread, {a}=left; wording says the left arm
    places the bread onto the skillet and must not claim the skillet moves.
Pilot evidence: central-cam 30-seed pilot on version 0 0.47 (14/30 success),
    manifest hash a8a91465b6cfbc70; version 2 (widened bread pickup band)
    50-seed pilot 0.36 (18/50 success), manifest hash bad1d4f25f906b53;
    version 3 (union skillet sweep) 50-seed pilot 0.38 (19/50 success),
    manifest hash e0f8b156b7b9c742
"""

from __future__ import annotations

import numpy as np

from ..left_task_base import LeftTaskBase, SceneRejectedError
from ..left_task_manifests import get_manifest
from ...utils import *  # noqa: F401,F403

# ---------------------------------------------------------------------------
# LEGACY_RANDOMIZATION_PARAM
# The randomization protocol used before the current version 2 design.
# Intentionally unused: kept as a declared header constant so the previous
# geometry is reproducible and comparable.
#   Version 0 (original source-task geometry, manifest a8a91465b6cfbc70):
#     - bread pickup: x in [-0.42, -0.36], y in [-0.15, 0.02], yaw in [0, 0.50]
#     - skillet:      static x in [-0.26, -0.12], y in [-0.24, 0.14],
#                     yaw in [0, 0.50], raised +0.025 m, clearance 0.10 m
LEGACY_RANDOMIZATION_PARAM: dict[str, object] = {
    "pickup_bread_x": (-0.42, -0.36),
    "pickup_bread_y": (-0.15, 0.02),
    "pickup_bread_yaw_rad": (0.0, 0.50),
    "bread_min_clearance_m": 0.10,
    "skillet_x": (-0.26, -0.12),
    "skillet_y": (-0.24, 0.14),
    "skillet_yaw_rad": (0.0, 0.50),
    "skillet_clearance_m": 0.10,
    "skillet_raised_z_m": 0.025,
    "manifest_hash": "a8a91465b6cfbc70",
}

# Nominal table surface (rand_pose's default zlim and the created table height).
TABLE_SURFACE_Z = 0.741
# Contact-height success floor.  Measured with the shipped hulls at the
# manifest quat and z=0.741: the skillet functional point sits at
# 0.740-0.759 m across model_ids 0-3 and bread settling inside the pan rests
# at 0.748-0.783 m, so the old stock 0.76 gate only passed because the
# skillet was floated +0.025 m.  These gates now just catch gross failures
# (fallen through the table); the tight 0.035 m functional-point xy gate is
# what actually pins the bread inside the pan.
SUCCESS_MIN_HEIGHT_Z = 0.73


class PlaceBreadSkilletLeftImpl(LeftTaskBase):
    """Left-arm-only bread-on-skillet task.

    The skillet is a static target initialised in the left workspace before the
    episode begins; only the bread is sampled and transported by the left arm.
    """

    def __init__(self) -> None:
        super().__init__()
        self.manifest = get_manifest("place_bread_skillet")

    def sample_layout(self) -> dict[str, object]:
        bread_spec = self.manifest.actor_specs["bread"]
        skillet_spec = self.manifest.actor_specs["skillet"]
        clearance = max(bread_spec.minimum_clearance_m, skillet_spec.minimum_clearance_m)
        for _ in range(128):
            bread = self.sample_spec_pose(bread_spec)
            skillet = self.sample_spec_pose(skillet_spec)
            if float(np.linalg.norm(bread.p[:2] - skillet.p[:2])) >= clearance:
                return {"bread": bread, "skillet": skillet}
        raise SceneRejectedError(
            "could not sample a feasible place_bread_skillet layout in 128 attempts"
        )

    def load_actors(self) -> None:
        layout = self.sample_layout()
        if not self.validate_layout(layout):
            raise SceneRejectedError("place_bread_skillet layout rejected by validate_layout")
        bread_id_list = [0, 1, 3, 5, 6]
        self.bread_id = int(np.random.choice(bread_id_list))
        self.bread = create_actor(
            self, pose=layout["bread"], modelname="075_bread",
            model_id=self.bread_id, convex=True,
        )
        # Keep the bread at its default mass; a near-zero mass makes the
        # gripper closing impulse launch the bread away on a single arm.
        # self.bread.set_mass(0.001)
        skillet_id_list = [0, 1, 2, 3]
        self.skillet_id = int(np.random.choice(skillet_id_list))
        # Spawn the static skillet at table-contact height: with the manifest
        # quat the shipped 106_skillet hulls put the pan bottom at -2.5..0 mm
        # for z=0.741, so the origin needs no offset.  A static actor never
        # settles, so any raise would float in every collected frame (the
        # historical +0.025 m raise did exactly that).
        self.skillet = create_actor(
            self, pose=layout["skillet"], modelname="106_skillet",
            model_id=self.skillet_id, convex=True, is_static=True,
        )
        self.skillet.set_mass(0.01)
        self.add_prohibit_area(self.bread, padding=0.03)
        self.add_prohibit_area(self.skillet, padding=0.05)
        self.record_layout(layout)
        skillet_point = self.skillet.get_functional_point(0)[:3]
        self.route_bins = self.classify_routes(
            [("bread", np.asarray(layout["bread"].p), np.asarray(skillet_point))]
        )

    def place_bread_on_skillet_left(self) -> bool:
        """Complete expert sequence: left grasp, lift, place, release, retreat."""
        if not self.plan_success:
            return False
        arm_tag = ArmTag("left")
        self.move(self.grasp_actor(self.bread, arm_tag=arm_tag, pre_grasp_dis=0.07))
        if not self.plan_success:
            return False
        self.move_by_displacement(arm_tag=arm_tag, z=0.1)
        if not self.plan_success:
            return False
        target_pose = self.skillet.get_functional_point(0)
        self.move(self.place_actor(self.bread, arm_tag=arm_tag, target_pose=target_pose,
                                   constrain="free", pre_dis=0.05, dis=0.05))
        if not self.plan_success:
            return False
        self.move(self.open_gripper(arm_tag))
        self.move_by_displacement(arm_tag=arm_tag, z=0.1, move_axis="arm")
        if not self.plan_success:
            self.plan_success = True
        return self.left_return_home()

    def play_once(self) -> dict:
        self.place_bread_on_skillet_left()
        self.info["info"] = {
            "{A}": f"106_skillet/base{self.skillet_id}",
            "{B}": f"075_bread/base{self.bread_id}",
            "{a}": "left",
        }
        return self.info

    def check_success(self) -> bool:
        target_pose = self.skillet.get_functional_point(0)
        bread_pose = self.bread.get_pose().p
        return (
            bool(np.all(abs(target_pose[:2] - bread_pose[:2]) < [0.035, 0.035]))
            and bool(target_pose[2] > SUCCESS_MIN_HEIGHT_Z + self.table_z_bias)
            and bool(bread_pose[2] > SUCCESS_MIN_HEIGHT_Z + self.table_z_bias)
            and self.is_left_gripper_open()
            and self.right_arm_stationary()
        )
