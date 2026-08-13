"""LEFT_TASK_DESIGN:
Source task: stack_blocks_two
Single-arm semantic change: source picks each block with whichever arm matches
    its x sign and hands the second block to the opposite arm between
    placements; the derived task samples both blocks in one left workspace,
    places block 1 on a left target and stacks block 2 onto block 1, all with
    the left arm only, waiting for block 1 to settle before stacking.
Left workspace manifest: stack_blocks_two, version 2 (2026-08-13)
Actors and clearance: block1/block2 (dynamic boxes, both sharing the identical
    sweep x in [-0.45,-0.10], y in [-0.08,0.05], yaw up to 0.60, pairwise min
    0.10 m), block 1 target randomized in x in [-0.34,-0.26] at y in
    [-0.16,-0.10].
Expert sequence: left pick block 1, place at left target, settle delay; left
    pick block 2, align to block 1 functional point one, release, return home.
Success predicate: preserve the source stacked-geometry test (block 2 on
    block 1 within [0.025, 0.025, 0.012]); require left open and right home.
    Block 1 is not frozen; unstable seeds are rejected by the layout sampler.
Instruction change: {A}=red block, {B}=green block, {a}..{b}=left; wording
    says the left arm stacks the green block onto the red block.
Pilot evidence: central-cam 30-seed pilot on version 0 1.00 (30/30 success),
    manifest hash 2585a2c9d73c5073; version 2 (randomized stack target)
    50-seed pilot 1.00 (50/50 success), manifest hash 2585a2c9d73c5073
"""

from __future__ import annotations

import numpy as np

from .left_task_base import LeftTaskBase, SceneRejectedError
from .left_task_manifests import get_manifest
from ..utils import *  # noqa: F401,F403

# ---------------------------------------------------------------------------
# LEGACY_RANDOMIZATION_PARAM
# The randomization protocol used before the current version 2 design.
# Intentionally unused: kept as a declared header constant so the previous
# geometry is reproducible and comparable.
#   Version 0 (original source-task geometry, manifest 2585a2c9d73c5073):
#     - block1/block2: x in [-0.45, -0.10], y in [-0.08, 0.05],
#                      yaw in [0, 0.60], clearance 0.10 m
#     - block 1 target: fixed at x=-0.30, y=-0.13 (z 0.75 + table bias)
LEGACY_RANDOMIZATION_PARAM: dict[str, object] = {
    "block_x": (-0.45, -0.10),
    "block_y": (-0.08, 0.05),
    "block_yaw_rad": (0.0, 0.60),
    "block_clearance_m": 0.10,
    "target_x": (-0.30, -0.30),
    "target_y": (-0.13, -0.13),
    "manifest_hash": "2585a2c9d73c5073",
}


class StackBlocksTwoLeftImpl(LeftTaskBase):
    """Left-arm-only two-block stacking task.

    The left arm places block 1 onto a left-workspace target, waits for it to
    settle, then aligns block 2 onto block 1's functional point one.
    """

    def __init__(self) -> None:
        super().__init__()
        self.manifest = get_manifest("stack_blocks_two")
        self.block_half_size = 0.025

    def sample_layout(self) -> dict[str, object]:
        specs = {
            name: self.manifest.actor_specs[name]
            for name in ("block1", "block2")
        }
        clearance = max(spec.minimum_clearance_m for spec in specs.values())
        # Version 2: the stack target is no longer fixed at (-0.30,-0.13);
        # sample it inside a conservative reachable band (SR was 1.00 with the
        # fixed target, so keep the sweep modest) and share the sampled value
        # with load_actors through ``self.stack_target_xy``.
        self.stack_target_xy = np.array([
            float(np.random.uniform(-0.34, -0.26)),
            float(np.random.uniform(-0.16, -0.10)),
        ], dtype=float)
        target = self.stack_target_xy
        for _ in range(128):
            block1 = self.sample_spec_pose(specs["block1"])
            block2 = self.sample_spec_pose(specs["block2"])
            for pose in (block1, block2):
                pose.p[2] = 0.741 + self.block_half_size
            if not self._clear(block1, block2, clearance):
                continue
            if float(np.linalg.norm(block1.p[:2] - target)) < 0.15:
                continue
            if float(np.linalg.norm(block2.p[:2] - target)) < 0.15:
                continue
            return {"block1": block1, "block2": block2}
        raise SceneRejectedError(
            "could not sample a feasible stack_blocks_two layout in 128 attempts"
        )

    @staticmethod
    def _clear(pose_a, pose_b, clearance: float) -> bool:
        return bool(np.linalg.norm(pose_a.p[:2] - pose_b.p[:2]) >= clearance)

    def load_actors(self) -> None:
        layout = self.sample_layout()
        if not self.validate_layout(layout):
            raise SceneRejectedError("stack_blocks_two layout rejected by validate_layout")
        color_lst = [(1, 0, 0), (0, 1, 0)]

        def create_block(block_pose, color):
            half = (self.block_half_size,) * 3
            return create_box(scene=self, pose=block_pose, half_size=half,
                              color=color, name="box")

        self.block1 = create_block(layout["block1"], color_lst[0])
        self.block2 = create_block(layout["block2"], color_lst[1])
        self.add_prohibit_area(self.block1, padding=0.07)
        self.add_prohibit_area(self.block2, padding=0.07)
        self.block1_target_pose = [
            float(self.stack_target_xy[0]), float(self.stack_target_xy[1]),
            0.75 + self.table_z_bias, 0, 1, 0, 0,
        ]
        self.record_layout(layout)
        self.route_bins = self.classify_routes([
            ("block1", np.asarray(layout["block1"].p), np.asarray(self.block1_target_pose[:3])),
            ("block2", np.asarray(layout["block2"].p),
             np.asarray(self.block1_target_pose[:3]) + np.array([0.0, 0.0, 0.05])),
        ])

    def pick_and_place_block(self, block) -> bool:
        """Pick one block with the left arm and place it onto the stack target.

        @input: the block actor.
        @output: boolean plan success.
        @scenario: Shared left grasp, lift, aligned placement and lift-back.
        """
        if not self.plan_success:
            return False
        arm_tag = ArmTag("left")
        self.move(self.grasp_actor(block, arm_tag=arm_tag, pre_grasp_dis=0.09))
        if not self.plan_success:
            return False
        self.move_by_displacement(arm_tag=arm_tag, z=0.07)
        if not self.plan_success:
            return False
        if getattr(self, "last_actor", None) is None:
            target_pose = self.block1_target_pose
        else:
            target_pose = self.last_actor.get_functional_point(1)
        self.move(self.place_actor(block, target_pose=target_pose, arm_tag=arm_tag,
                                   functional_point_id=0, pre_dis=0.05, dis=0.0,
                                   pre_dis_axis="fp"))
        if not self.plan_success:
            return False
        self.move_by_displacement(arm_tag=arm_tag, z=0.07)
        self.last_actor = block
        return self.plan_success

    def play_once(self) -> dict:
        self.last_actor = None
        self.pick_and_place_block(self.block1)
        self.delay(2)
        self.pick_and_place_block(self.block2)
        self.info["info"] = {
            "{A}": "red block",
            "{B}": "green block",
            "{a}": "left",
            "{b}": "left",
        }
        return self.info

    def check_success(self) -> bool:
        block1_pose = self.block1.get_pose().p
        block2_pose = self.block2.get_pose().p
        eps = [0.025, 0.025, 0.012]
        return (
            bool(np.all(abs(block2_pose - np.array(
                block1_pose[:2].tolist() + [block1_pose[2] + 0.05]
            )) < eps))
            and self.is_left_gripper_open()
            and self.right_arm_stationary()
        )
