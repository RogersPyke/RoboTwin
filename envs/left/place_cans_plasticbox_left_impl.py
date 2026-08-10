"""LEFT_TASK_DESIGN:
Source task: place_cans_plasticbox
Single-arm semantic change: source grasps can 1 with the left arm and can 2
    with the right arm concurrently and drops both into the plastic box; the
    derived task fixes the plastic box static in the left workspace and
    serially transfers can 1 to functional point one and can 2 to functional
    point zero with the left arm, returning home between transfers.
Left workspace manifest: place_cans_plasticbox, version 0
Actors and clearance: plasticbox (static, min 0.08 m), can1/can2 (dynamic,
    min 0.08 m each).
Expert sequence: left grasp can 1, lift z=0.20, place at plastic box
    functional point one, lift, return home; left grasp can 2, place at
    functional point zero, lift, return home.
Success predicate: preserve both nearest-functional-point containment checks
    (0.04 m) and the left-open condition; replace the right-gripper condition
    with right-home immobility.
Instruction change: {A}=can 1, {B}=plastic box, {C}=can 2, {a}=left; wording
    says the left arm drops both cans into the stationary plastic box.
Pilot evidence: central-cam 30-seed pilot 0.40 (12/30 success), manifest hash
    0ed2a154b3a88e40
"""

from __future__ import annotations

import numpy as np

from .left_task_base import LeftTaskBase, SceneRejectedError
from .left_task_manifests import get_manifest
from ..utils import *  # noqa: F401,F403


class PlaceCansPlasticboxLeftImpl(LeftTaskBase):
    """Left-arm-only two-cans-into-plastic-box task.

    The plastic box is static in the left workspace; the left arm serially
    transfers can 1 to functional point one and can 2 to functional point
    zero.
    """

    def __init__(self) -> None:
        super().__init__()
        self.manifest = get_manifest("place_cans_plasticbox")

    def sample_layout(self) -> dict[str, object]:
        specs = {name: self.manifest.actor_specs[name] for name in
                 ("plasticbox", "can1", "can2")}
        clearance = max(spec.minimum_clearance_m for spec in specs.values())
        for _ in range(128):
            plasticbox = self.sample_spec_pose(specs["plasticbox"])
            can1 = self.sample_spec_pose(specs["can1"])
            can2 = self.sample_spec_pose(specs["can2"])
            if (self._clear(can1, can2, clearance)
                    and self._clear(can1, plasticbox, clearance)
                    and self._clear(can2, plasticbox, clearance)):
                return {"plasticbox": plasticbox, "can1": can1, "can2": can2}
        raise SceneRejectedError(
            "could not sample a feasible place_cans_plasticbox layout in 128 attempts"
        )

    @staticmethod
    def _clear(pose_a, pose_b, clearance: float) -> bool:
        return bool(np.linalg.norm(pose_a.p[:2] - pose_b.p[:2]) >= clearance)

    def load_actors(self) -> None:
        layout = self.sample_layout()
        if not self.validate_layout(layout):
            raise SceneRejectedError(
                "place_cans_plasticbox layout rejected by validate_layout"
            )
        self.plasticbox_id = int(np.random.choice([3, 5]))
        self.plasticbox = create_actor(
            scene=self, pose=layout["plasticbox"], modelname="062_plasticbox",
            convex=True, model_id=self.plasticbox_id, is_static=True,
        )
        self.plasticbox.set_mass(0.05)
        self.object1_id = int(np.random.choice([0, 1, 2, 3, 5, 6]))
        self.can1 = create_actor(
            scene=self, pose=layout["can1"], modelname="071_can",
            convex=True, model_id=self.object1_id,
        )
        self.can1.set_mass(0.05)
        self.object2_id = int(np.random.choice([0, 1, 2, 3, 5, 6]))
        self.can2 = create_actor(
            scene=self, pose=layout["can2"], modelname="071_can",
            convex=True, model_id=self.object2_id,
        )
        self.can2.set_mass(0.05)
        self.add_prohibit_area(self.plasticbox, padding=0.1)
        self.add_prohibit_area(self.can1, padding=0.05)
        self.add_prohibit_area(self.can2, padding=0.05)
        self.record_layout(layout)
        fp0 = self.plasticbox.get_functional_point(0)[:3]
        fp1 = self.plasticbox.get_functional_point(1)[:3]
        self.route_bins = self.classify_routes([
            ("can1", np.asarray(layout["can1"].p), np.asarray(fp1)),
            ("can2", np.asarray(layout["can2"].p), np.asarray(fp0)),
        ])

    def place_can_left(self, can, point_id: int) -> bool:
        """Grasp, lift and place one can at one plastic box functional point.

        @input: the can actor and the target functional point index.
        @output: boolean plan success.
        @scenario: Serial left-arm transfer with home reset between cans.
        """
        if not self.plan_success:
            return False
        arm_tag = ArmTag("left")
        self.move(self.grasp_actor(can, arm_tag=arm_tag, pre_grasp_dis=0.1))
        if not self.plan_success:
            return False
        # Iteration-4 pilot: lifting z=0.20 right after the grasp pushes the
        # left EEF past its ~1.03 m ceiling (grasp already sits at z~0.93 when
        # the can spawns near the left base); z=0.08 keeps the lift reachable.
        self.move_by_displacement(arm_tag=arm_tag, z=0.08)
        if not self.plan_success:
            return False
        target_pose = self.plasticbox.get_functional_point(point_id)
        self.move(self.place_actor(can, arm_tag=arm_tag, target_pose=target_pose,
                                   constrain="free", pre_dis=0.1))
        if not self.plan_success:
            return False
        self.move_by_displacement(arm_tag=arm_tag, z=0.08)
        return self.left_return_home()

    def play_once(self) -> dict:
        self.place_can_left(self.can1, 1)
        self.place_can_left(self.can2, 0)
        self.info["info"] = {
            "{A}": f"071_can/base{self.object1_id}",
            "{B}": f"062_plasticbox/base{self.plasticbox_id}",
            "{C}": f"071_can/base{self.object2_id}",
            "{a}": "left",
        }
        return self.info

    def check_success(self) -> bool:
        fp0 = self.plasticbox.get_functional_point(0)[0:2]
        fp1 = self.plasticbox.get_functional_point(1)[0:2]
        dis1 = min(
            float(np.linalg.norm(self.can1.get_pose().p[0:2] - fp0)),
            float(np.linalg.norm(self.can1.get_pose().p[0:2] - fp1)),
        )
        dis2 = min(
            float(np.linalg.norm(self.can2.get_pose().p[0:2] - fp0)),
            float(np.linalg.norm(self.can2.get_pose().p[0:2] - fp1)),
        )
        threshold = 0.04
        return dis1 < threshold and dis2 < threshold \
            and self.is_left_gripper_open() and self.right_arm_stationary()
