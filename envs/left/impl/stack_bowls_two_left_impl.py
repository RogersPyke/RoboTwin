"""LEFT_TASK_DESIGN:
Source task: stack_bowls_two
Single-arm semantic change: source picks each bowl with whichever arm matches
    its x sign and switches arms between placements; the derived task samples
    both bowls in one left workspace, moves the lower bowl onto a left target
    and stacks the upper bowl onto it, all with the left arm only, verifying
    that the lower bowl has settled before the top-down upper grasp.
Left workspace manifest: stack_bowls_two, version 2 (2026-08-13)
Actors and clearance: bowl1/bowl2 (dynamic, both sharing the identical sweep
    x in [-0.50,-0.05], y in [-0.15,0.15], zero yaw, pairwise min 0.13 m),
    lower bowl target randomized in x in [-0.34,-0.26] at y in [-0.14,-0.06]
    and z 0.76 m.
Expert sequence: left grasp lower bowl, lift z=0.10, align to left target,
    lift, settle delay; left grasp upper bowl with a top-down grasp, stack
    onto the lower bowl, lift, return home.
Success predicate: preserve the source two-height stacking geometry (aligned
    xy within 0.04 m, heights at 0.74/0.77 m plus table bias within 0.02 m);
    require left open and right home.
Instruction change: {A}=lower bowl, {B}=upper bowl, {a}..{b}=left; wording
    says the left arm stacks the upper bowl onto the lower bowl.
Pilot evidence: central-cam 30-seed pilot on version 0 0.57 (17/30 success),
    manifest hash ace5714486a162e4; version 2 (randomized stack target)
    50-seed pilot 0.56 (28/50 success), manifest hash ace5714486a162e4
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
#   Version 0 (original source-task geometry, manifest ace5714486a162e4):
#     - bowl1/bowl2: x in [-0.50, -0.05], y in [-0.15, 0.15], yaw 0,
#                    clearance 0.13 m
#     - lower bowl target: fixed at x=-0.30, y=-0.10 (z 0.76 m)
LEGACY_RANDOMIZATION_PARAM: dict[str, object] = {
    "bowl_x": (-0.50, -0.05),
    "bowl_y": (-0.15, 0.15),
    "bowl_yaw_rad": (0.0, 0.00),
    "bowl_clearance_m": 0.13,
    "target_x": (-0.30, -0.30),
    "target_y": (-0.10, -0.10),
    "manifest_hash": "ace5714486a162e4",
}


class StackBowlsTwoLeftImpl(LeftTaskBase):
    """Left-arm-only two-bowl stacking task.

    The left arm moves the lower bowl onto a left-workspace target, waits for
    it to settle, then stacks the upper bowl onto the lower bowl.
    """

    def __init__(self) -> None:
        super().__init__()
        self.manifest = get_manifest("stack_bowls_two")
        self.quat_of_target_pose = [0.0, 0.707, 0.707, 0.0]

    def sample_layout(self) -> dict[str, object]:
        specs = {
            name: self.manifest.actor_specs[name]
            for name in ("bowl1", "bowl2")
        }
        clearance = max(spec.minimum_clearance_m for spec in specs.values())
        # Version 2: the stack target is no longer fixed at (-0.30,-0.10);
        # sample it inside a conservative reachable band (SR was 0.57 with the
        # fixed target, so keep the sweep modest) and share the sampled value
        # with load_actors through ``self.stack_target_xy``.
        self.stack_target_xy = np.array([
            float(np.random.uniform(-0.18, 0.18)),
            float(np.random.uniform(-0.12, 0.12)),
        ], dtype=float)
        target = self.stack_target_xy
        for _ in range(128):
            bowl1 = self.sample_spec_pose(specs["bowl1"])
            bowl2 = self.sample_spec_pose(specs["bowl2"])
            if float(np.linalg.norm(bowl1.p[:2] - bowl2.p[:2])) < clearance:
                continue
            if float(np.linalg.norm(bowl1.p[:2] - target)) < 0.13:
                continue
            if float(np.linalg.norm(bowl2.p[:2] - target)) < 0.13:
                continue
            return {"bowl1": bowl1, "bowl2": bowl2}
        raise SceneRejectedError(
            "could not sample a feasible stack_bowls_two layout in 128 attempts"
        )

    def load_actors(self) -> None:
        layout = self.sample_layout()
        if not self.validate_layout(layout):
            raise SceneRejectedError("stack_bowls_two layout rejected by validate_layout")
        poses = [layout["bowl1"], layout["bowl2"]]
        poses = sorted(poses, key=lambda pose: float(pose.p[1]))

        def create_bowl(bowl_pose):
            return create_actor(scene=self, pose=bowl_pose, modelname="002_bowl",
                                model_id=3, convex=True)

        self.bowl1 = create_bowl(poses[0])
        self.bowl2 = create_bowl(poses[1])
        self.add_prohibit_area(self.bowl1, padding=0.07)
        self.add_prohibit_area(self.bowl2, padding=0.07)
        self.record_layout({"bowl1": poses[0], "bowl2": poses[1]})
        self.bowl1_target_pose = np.array([
            float(self.stack_target_xy[0]), float(self.stack_target_xy[1]), 0.76,
        ], dtype=float)
        target = self.bowl1_target_pose[:3]
        self.route_bins = self.classify_routes([
            ("bowl1", np.asarray(poses[0].p), np.asarray(target)),
            ("bowl2", np.asarray(poses[1].p),
             np.asarray(target) + np.array([0.0, 0.0, 0.05])),
        ])

    def move_bowl(self, actor, target_pose: np.ndarray) -> bool:
        """Grasp one bowl with the left arm and align it onto a target pose.

        @input: the bowl actor and a three-value target position; the target
            quaternion is appended by this routine.
        @output: boolean plan success.
        @scenario: Shared left grasp (top-down, contact point zero), lift,
            aligned placement and lift-back.
        """
        if not self.plan_success:
            return False
        arm_tag = ArmTag("left")
        self.move(self.grasp_actor(actor, arm_tag=arm_tag, contact_point_id=0,
                                   pre_grasp_dis=0.1))
        if not self.plan_success:
            return False
        self.move_by_displacement(arm_tag=arm_tag, z=0.1)
        if not self.plan_success:
            return False
        target_pose_7 = target_pose.tolist() + self.quat_of_target_pose
        self.move(self.place_actor(actor, target_pose=target_pose_7, arm_tag=arm_tag,
                                   functional_point_id=0, pre_dis=0.09, dis=0.0,
                                   constrain="free"))
        if not self.plan_success:
            return False
        self.move_by_displacement(arm_tag=arm_tag, z=0.09)
        return self.plan_success

    def play_once(self) -> dict:
        self.move_bowl(self.bowl1, self.bowl1_target_pose)
        self.delay(2)
        stack_pose = np.asarray(self.bowl1.get_pose().p, dtype=float) + np.array([0, 0, 0.05])
        self.move_bowl(self.bowl2, stack_pose)
        self.info["info"] = {
            "{A}": "002_bowl/base3",
            "{B}": "002_bowl/base3",
            "{a}": "left",
            "{b}": "left",
        }
        return self.info

    def check_success(self) -> bool:
        bowl1_pose = self.bowl1.get_pose().p
        bowl2_pose = self.bowl2.get_pose().p
        bowl1_pose, bowl2_pose = sorted([bowl1_pose, bowl2_pose], key=lambda p: p[2])
        target_height = [0.74 + self.table_z_bias, 0.77 + self.table_z_bias]
        eps = 0.02
        eps2 = 0.04
        return (
            bool(np.all(abs(bowl1_pose[:2] - bowl2_pose[:2]) < eps2))
            and bool(np.all(np.array([bowl1_pose[2], bowl2_pose[2]]) - target_height < eps))
            and self.is_left_gripper_open()
            and self.right_arm_stationary()
        )
