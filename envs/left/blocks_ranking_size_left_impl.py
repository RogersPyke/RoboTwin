"""LEFT_TASK_DESIGN:
Source task: blocks_ranking_size
Single-arm semantic change: source chooses each block's arm from its x sign and
    switches arms with ``back_to_origin(opposite)``; the derived task samples
    all three blocks in one left workspace and ranks them serially with the
    left arm only in the source semantic order small, medium, large.
Left workspace manifest: blocks_ranking_size, version 2 (2026-08-13)
Actors and clearance: block1/2/3 (dynamic boxes, half sizes
    [0.030-0.033, 0.024-0.027, 0.018-0.021], pairwise min 0.10 m), shared
    pickup band x in [-0.45,-0.16], y in [-0.18,0.08], yaw up to 0.75; ordered
    target line translated by line_x in [-0.45,-0.36] so the ranking can sit
    close to or far from the arm, with per-slot x offsets
    [0,0.02]/[0.09,0.11]/[0.18,0.20] at a shared y in [-0.20,-0.10].
Expert sequence: left pick small block, place, return home; left pick medium,
    place, return home; left pick large, place, return home.
Success predicate: preserve the source geometry ordering relation; remove the
    right-gripper-open condition; require left open and right home.
Instruction change: {A}=large block, {B}=medium block, {C}=small block, {a}..{c}
    =left; wording explicitly describes size ordering, not colour.
Pilot evidence: central-cam 30-seed pilot on version 0 0.77 (23/30 success),
    manifest hash 01eab448b0621e63; version 2 (line-translated targets) 50-seed
    pilot 0.54 (27/50 success), manifest hash ed8fb45f24857693
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
#   Version 0 (original source-task geometry, manifest 01eab448b0621e63):
#     - pickup band:  x in [-0.45, -0.22], y in [-0.08, 0.05], yaw in [0, 0.60]
#     - pairwise min clearance: 0.10 m
#     - target line:  large x in [-0.42, -0.40], medium x in [-0.33, -0.31],
#                     small x in [-0.24, -0.22], shared y in [-0.20, -0.10]
LEGACY_RANDOMIZATION_PARAM: dict[str, object] = {
    "pickup_x": (-0.45, -0.22),
    "pickup_y": (-0.08, 0.05),
    "pickup_yaw_rad": (0.0, 0.60),
    "min_clearance_m": 0.10,
    "target_large_x": (-0.42, -0.40),
    "target_medium_x": (-0.33, -0.31),
    "target_small_x": (-0.24, -0.22),
    "target_y": (-0.20, -0.10),
    "manifest_hash": "01eab448b0621e63",
}


class BlocksRankingSizeLeftImpl(LeftTaskBase):
    """Left-arm-only large/medium/small block ranking task.

    Three different-sized cubes are moved by the left arm onto an ordered
    target line with large x < medium x < small x at a common y, following the
    source execution order small, medium, large.
    """

    def __init__(self) -> None:
        super().__init__()
        self.manifest = get_manifest("blocks_ranking_size")
        self.halfsize_lst: list[float] | None = None

    def sample_layout(self) -> dict[str, object]:
        specs = {
            name: self.manifest.actor_specs[name]
            for name in ("block1", "block2", "block3")
        }
        clearance = max(spec.minimum_clearance_m for spec in specs.values())
        assert self.halfsize_lst is not None, "halfsize_lst must be sampled before layout"
        for _ in range(128):
            block1 = self.sample_spec_pose(specs["block1"])
            block2 = self.sample_spec_pose(specs["block2"])
            block3 = self.sample_spec_pose(specs["block3"])
            for pose, half in zip((block1, block2, block3), self.halfsize_lst):
                pose.p[2] = 0.741 + float(half)
            if (self._pairwise_clear(block1, block2, clearance)
                    and self._pairwise_clear(block2, block3, clearance)
                    and self._pairwise_clear(block1, block3, clearance)):
                return {"block1": block1, "block2": block2, "block3": block3}
        raise SceneRejectedError(
            "could not sample a feasible blocks_ranking_size layout in 128 attempts"
        )

    @staticmethod
    def _pairwise_clear(pose_a, pose_b, clearance: float) -> bool:
        return bool(np.linalg.norm(pose_a.p[:2] - pose_b.p[:2]) >= clearance)

    def load_actors(self) -> None:
        self.halfsize_lst = [
            float(np.random.uniform(0.03, 0.033)),
            float(np.random.uniform(0.024, 0.027)),
            float(np.random.uniform(0.018, 0.021)),
        ]
        layout = self.sample_layout()
        if not self.validate_layout(layout):
            raise SceneRejectedError("blocks_ranking_size layout rejected by validate_layout")
        color_lst = [(np.random.random(), np.random.random(), np.random.random()) for _ in range(3)]

        def create_block(block_pose, half, color):
            half_size = (half, half, half)
            return create_box(scene=self, pose=block_pose, half_size=half_size,
                              color=color, name="box")

        self.block1 = create_block(layout["block1"], self.halfsize_lst[0], color_lst[0])
        self.block2 = create_block(layout["block2"], self.halfsize_lst[1], color_lst[1])
        self.block3 = create_block(layout["block3"], self.halfsize_lst[2], color_lst[2])
        self.add_prohibit_area(self.block1, padding=0.1)
        self.add_prohibit_area(self.block2, padding=0.1)
        self.add_prohibit_area(self.block3, padding=0.1)
        # Ordered target line inside the left workspace: large < medium < small x.
        # The whole line translates via ``line_x`` so the ranking can be
        # positioned close to or far from the arm, and every slot keeps a
        # per-slot x band.  The relative offsets keep each adjacent worst-case
        # extreme 0.11 m apart (< 0.13 m, so the check_success eps=[0.13, 0.03]
        # predicate always holds for exactly-placed blocks) and keep the
        # smallest centre gap 0.07 m above the 0.066 m largest combined radius,
        # so the placed blocks never overlap; large<medium<small is guaranteed
        # by the slot ordering.  y_pose is shared across the three slots so the
        # 0.03 y tolerance is trivially met, and the full line stays inside the
        # pickup x band.
        line_x = np.random.uniform(-0.45, -0.36)
        y_pose = np.random.uniform(-0.2, -0.1)
        self.block1_target_pose = [
            np.random.uniform(line_x, line_x + 0.02), y_pose, 0.74 + self.table_z_bias,
        ] + [0, 1, 0, 0]
        self.block2_target_pose = [
            np.random.uniform(line_x + 0.09, line_x + 0.11), y_pose, 0.74 + self.table_z_bias,
        ] + [0, 1, 0, 0]
        self.block3_target_pose = [
            np.random.uniform(line_x + 0.18, line_x + 0.20), y_pose, 0.74 + self.table_z_bias,
        ] + [0, 1, 0, 0]
        self.record_layout(layout)
        targets = [self.block1_target_pose, self.block2_target_pose, self.block3_target_pose]
        self.route_bins = self.classify_routes([
            (f"block{i + 1}", np.asarray(layout[f"block{i + 1}"].p), np.asarray(targets[i][:3]))
            for i in range(3)
        ])

    def place_ranked_block(self, block, target_pose: list) -> bool:
        """Pick one block with the left arm, place it on the line and go home."""
        if not self.plan_success:
            return False
        placed = self.left_pick_place(
            block, target_pose, pre_grasp_dis=0.09, lift_z=0.07,
            pre_dis=0.09, dis=0.02, constrain="align",
        )
        if not placed:
            return False
        return self.left_return_home()

    def play_once(self) -> dict:
        # Source semantic order: small, medium, large.
        self.place_ranked_block(self.block3, self.block3_target_pose)
        self.place_ranked_block(self.block2, self.block2_target_pose)
        self.place_ranked_block(self.block1, self.block1_target_pose)
        self.info["info"] = {
            "{A}": "large block",
            "{B}": "medium block",
            "{C}": "small block",
            "{a}": "left",
            "{b}": "left",
            "{c}": "left",
        }
        return self.info

    def check_success(self) -> bool:
        block1_pose = self.block1.get_pose().p
        block2_pose = self.block2.get_pose().p
        block3_pose = self.block3.get_pose().p
        eps = [0.13, 0.03]
        return (
            bool(np.all(abs(block1_pose[:2] - block2_pose[:2]) < eps))
            and bool(np.all(abs(block2_pose[:2] - block3_pose[:2]) < eps))
            and bool(block1_pose[0] < block2_pose[0])
            and bool(block2_pose[0] < block3_pose[0])
            and self.is_left_gripper_open()
            and self.right_arm_stationary()
        )
