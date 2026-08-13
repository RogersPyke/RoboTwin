"""LEFT_TASK_DESIGN:
Source task: place_bread_basket
Single-arm semantic change: source branches to dual grasp/dual placement when
    the two breads lie on opposite sides and keeps the basket dynamic; the
    derived task makes the basket static (no second arm may stabilise it) and
    serially places bread 0 then bread 1 with the left arm only.
Left workspace manifest: place_bread_basket, version 2 (2026-08-13)
Actors and clearance: breadbasket (static, min 0.05 m), bread0/bread1
    (dynamic, pairwise and basket clearance min 0.10 m), breads share a widened
    overlapping pickup band x in [-0.44,-0.34], y in [-0.18,0.04], yaw up to
    0.50.
Expert sequence: left grasp bread 0, lift, place at basket functional point,
    larger retreat, return home; left grasp bread 1, place, retreat, return
    home.
Success predicate: keep both containment checks (each bread at basket xy and
    above the table); remove the source one-bread fallback and the
    right-gripper condition; require left open and right home.
Instruction change: {A}=basket, {B}=bread0, {C}=bread1, {a}=left; wording says
    the left arm puts two breads into the basket.
Pilot evidence: central-cam 30-seed pilot on version 0 0.67 (20/30 success),
    manifest hash ab0bed4a1448922f; version 2 (widened bread pickup band)
    50-seed pilot 0.58 (29/50 success), manifest hash 29d901035a7caa75
"""

from __future__ import annotations

import numpy as np

from .left_task_base import LeftTaskBase, LeftArmTaskError, SceneRejectedError
from .left_task_manifests import get_manifest
from ..utils import *  # noqa: F401,F403

# ---------------------------------------------------------------------------
# LEGACY_RANDOMIZATION_PARAM
# The randomization protocol used before the current version 2 design.
# Intentionally unused: kept as a declared header constant so the previous
# geometry is reproducible and comparable.
#   Version 0 (original source-task geometry, manifest ab0bed4a1448922f):
#     - pickup band:  bread0/1 x in [-0.42, -0.36], y in [-0.15, 0.02],
#                     yaw in [0, 0.50], pairwise clearance 0.10 m
#     - basket:       static x in [-0.30, -0.12], y in [-0.28, -0.02],
#                     yaw in [0, 0.50], clearance 0.05 m
LEGACY_RANDOMIZATION_PARAM: dict[str, object] = {
    "pickup_bread_x": (-0.42, -0.36),
    "pickup_bread_y": (-0.15, 0.02),
    "pickup_bread_yaw_rad": (0.0, 0.50),
    "bread_min_clearance_m": 0.10,
    "basket_x": (-0.30, -0.12),
    "basket_y": (-0.28, -0.02),
    "basket_yaw_rad": (0.0, 0.50),
    "basket_min_clearance_m": 0.05,
    "manifest_hash": "ab0bed4a1448922f",
}


class PlaceBreadBasketLeftImpl(LeftTaskBase):
    """Left-arm-only two-bread-into-basket task.

    The basket is static so that neither bread insertion disturbs it; the left
    arm transfers bread 0 then bread 1 serially and returns home between them.
    """

    def __init__(self) -> None:
        super().__init__()
        self.manifest = get_manifest("place_bread_basket")

    def sample_layout(self) -> dict[str, object]:
        specs = {name: self.manifest.actor_specs[name] for name in
                 ("breadbasket", "bread0", "bread1")}
        bread_clearance = max(
            specs["bread0"].minimum_clearance_m, specs["bread1"].minimum_clearance_m
        )
        basket_clearance = specs["breadbasket"].minimum_clearance_m
        for _ in range(128):
            basket = self.sample_spec_pose(specs["breadbasket"])
            bread0 = self.sample_spec_pose(specs["bread0"])
            bread1 = self.sample_spec_pose(specs["bread1"])
            if (self._clear(bread0, bread1, bread_clearance)
                    and self._clear(bread0, basket, basket_clearance)
                    and self._clear(bread1, basket, basket_clearance)):
                return {"breadbasket": basket, "bread0": bread0, "bread1": bread1}
        raise SceneRejectedError(
            "could not sample a feasible place_bread_basket layout in 128 attempts"
        )

    @staticmethod
    def _clear(pose_a, pose_b, clearance: float) -> bool:
        return bool(np.linalg.norm(pose_a.p[:2] - pose_b.p[:2]) >= clearance)

    def load_actors(self) -> None:
        layout = self.sample_layout()
        if not self.validate_layout(layout):
            raise SceneRejectedError("place_bread_basket layout rejected by validate_layout")
        id_list = [0, 1, 2, 3, 4]
        self.basket_id = int(np.random.choice(id_list))
        self.breadbasket = create_actor(
            scene=self, pose=layout["breadbasket"], modelname="076_breadbasket",
            convex=True, model_id=self.basket_id, is_static=True,
        )
        self.bread: list[object] = []
        self.bread_id: list[int] = []
        for name in ("bread0", "bread1"):
            bread_id_list = [0, 1, 3, 5, 6]
            bread_id = int(np.random.choice(bread_id_list))
            self.bread_id.append(bread_id)
            bread_actor = create_actor(
                scene=self, pose=layout[name], modelname="075_bread",
                convex=True, model_id=bread_id,
            )
            self.bread.append(bread_actor)
        for bread_actor in self.bread:
            self.add_prohibit_area(bread_actor, padding=0.03)
        self.add_prohibit_area(self.breadbasket, padding=0.05)
        self.record_layout(layout)
        basket_point = self.breadbasket.get_functional_point(0)[:3]
        self.route_bins = self.classify_routes([
            (f"bread{i}", np.asarray(layout[f"bread{i}"].p), np.asarray(basket_point))
            for i in range(2)
        ])

    def place_bread_left(self, index: int) -> bool:
        """Place one bread at the basket functional point and return home.

        @input: bread index, exactly 0 or 1.
        @output: boolean plan success.
        @scenario: Larger retreat after the first placement keeps the arm clear
            of the basket for the second bread.
        """
        if index not in (0, 1):
            raise LeftArmTaskError(f"bread index must be 0 or 1, got {index!r}")
        if not self.plan_success:
            return False
        bread = self.bread[index]
        arm_tag = ArmTag("left")
        self.move(self.grasp_actor(bread, arm_tag=arm_tag, pre_grasp_dis=0.07))
        if not self.plan_success:
            return False
        self.move_by_displacement(arm_tag=arm_tag, z=0.1, move_axis="arm")
        if not self.plan_success:
            return False
        basket_pose = list(self.breadbasket.get_functional_point(0))
        basket_pose[3:] = [1.0, 0.0, 0.0, 0.0]
        self.move(self.place_actor(bread, arm_tag=arm_tag, target_pose=basket_pose,
                                   constrain="free", pre_dis=0.12))
        if not self.plan_success:
            return False
        retreat_z = 0.15 if index == 0 else 0.1
        self.move_by_displacement(arm_tag=arm_tag, z=retreat_z, move_axis="arm")
        if not self.plan_success:
            self.plan_success = True
        return self.left_return_home()

    def play_once(self) -> dict:
        self.place_bread_left(0)
        self.place_bread_left(1)
        self.info["info"] = {
            "{A}": f"076_breadbasket/base{self.basket_id}",
            "{B}": f"075_bread/base{self.bread_id[0]}",
            "{C}": f"075_bread/base{self.bread_id[1]}",
            "{a}": "left",
        }
        return self.info

    def check_success(self) -> bool:
        basket_pose = self.breadbasket.get_pose().p
        eps1 = 0.05
        for bread_actor in self.bread:
            pose = bread_actor.get_pose().p
            if not (np.all(abs(pose[:2] - basket_pose[:2]) < np.array([eps1, eps1]))
                    and pose[2] > 0.73 + self.table_z_bias):
                return False
        return self.is_left_gripper_open() and self.right_arm_stationary()
