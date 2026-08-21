"""LEFT_TASK_DESIGN:
Source task: blocks_ranking_size
Single-arm semantic change: source chooses each block's arm from its x sign and
    switches arms with ``back_to_origin(opposite)``; the derived task samples
    all three blocks in one left workspace and ranks them serially with the
    left arm only in the source semantic order small, medium, large.
Left workspace manifest: blocks_ranking_size, version 3 (2026-08-21)
Actors and clearance: block1/2/3 (dynamic boxes, half sizes
    [0.030-0.033, 0.024-0.027, 0.018-0.021], pairwise min 0.10 m), shared
    pickup band x in [-0.45,-0.16], y in [-0.18,0.08], yaw up to 0.75;
    ordered target line sampled as three consecutive slots on one straight
    line in the xoy plane whose angle with the x axis stays within +-15 deg,
    with the two adjacent gaps drawn independently from [0.10,0.16] m, and
    every slot inside the same envelope as the blocks' own initial
    randomization band (the centred-wide batch envelope when that variant
    is active).
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

from ..left_task_base import LeftTaskBase, SceneInitRejectError
from ..left_task_manifests import get_manifest
from ...utils import *  # noqa: F401,F403

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
#   Version 2 (code-level retune of the version 0 geometry): an x-parallel
#     target line at line_x in [-0.20,-0.16], shared y in [-0.15,0.15], with
#     per-slot x offsets [0,0.01]/[0.14,0.15]/[0.28,0.29].  Superseded by
#     version 3's angled-line randomization.
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
    "version_2": {
        "target_line_x": (-0.20, -0.16),
        "target_y": (-0.15, 0.15),
        "target_slot_offsets": [(0.0, 0.01), (0.14, 0.15), (0.28, 0.29)],
    },
}

# Version 3 (2026-08-21) target-line randomization: three consecutive slots
# on one straight line in the xoy plane.  The line's angle with the x axis is
# uniform in +-15 deg (so large < medium < small in x always holds), each
# adjacent gap is drawn independently from TARGET_GAP_RANGE_M, and every slot
# stays inside the blocks' own initial-randomization envelope.  The gap bounds
# exist because the smallest gap must keep the worst-case adjacent placed
# footprints clear (0.033 + 0.027 m block halves plus the expert's ~0.02 m
# placement drift per slot), while the largest keeps the full <= 0.32 m span
# fitting inside the envelope diagonally even at the 15 deg extreme.
# Tightened from the original +-45 deg to +-15 deg on 2026-08-22.
TARGET_LINE_MAX_ANGLE_RAD: float = float(np.deg2rad(15.0))
# Success-check slope bound: a line steeper than this cannot come
# from a +-15 deg sample plus placement noise (1.5x slack on the
# tangent, ~21.9 deg).
TARGET_LINE_REJECT_SLOPE: float = 1.5 * float(np.tan(TARGET_LINE_MAX_ANGLE_RAD))
TARGET_GAP_RANGE_M: tuple[float, float] = (0.10, 0.16)


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
        raise SceneInitRejectError(
            "could not sample a feasible blocks_ranking_size layout in 128 attempts"
        )

    @staticmethod
    def _pairwise_clear(pose_a, pose_b, clearance: float) -> bool:
        return bool(np.linalg.norm(pose_a.p[:2] - pose_b.p[:2]) >= clearance)

    def _sample_ranked_targets(self) -> list[list[float]]:
        """Sample the three ordered target slots on one random straight line.

        The line lies in the xoy plane with its angle to the x axis uniform in
        +-15 deg, the two adjacent gaps are drawn independently from
        ``TARGET_GAP_RANGE_M``, and the first slot is placed so the whole
        span stays inside the blocks' initial-randomization envelope (the
        manifest band, or the centred-wide batch envelope when that variant
        replaced it).
        """
        spec = self.manifest.actor_specs["block1"]
        (x_lo, x_hi), (y_lo, y_hi) = spec.workspace.x, spec.workspace.y
        theta = np.random.uniform(-TARGET_LINE_MAX_ANGLE_RAD, TARGET_LINE_MAX_ANGLE_RAD)
        direction = np.array([np.cos(theta), np.sin(theta)])
        gaps = np.random.uniform(*TARGET_GAP_RANGE_M, size=2)
        span = float(np.sum(gaps))
        # The start interval is shifted by the span's y extent so both line
        # endpoints stay inside the envelope whether the line tilts up or down.
        start = np.array([
            np.random.uniform(x_lo, x_hi - span * direction[0]),
            np.random.uniform(y_lo - min(span * direction[1], 0.0),
                              y_hi - max(span * direction[1], 0.0)),
        ])
        z = 0.74 + self.table_z_bias
        return [
            [float(c[0]), float(c[1]), z, 0.0, 1.0, 0.0, 0.0]
            for c in (start, start + direction * gaps[0], start + direction * span)
        ]

    def load_actors(self) -> None:
        self.halfsize_lst = [
            float(np.random.uniform(0.03, 0.033)),
            float(np.random.uniform(0.024, 0.027)),
            float(np.random.uniform(0.018, 0.021)),
        ]
        layout = self.sample_layout()
        if not self.validate_layout(layout):
            raise SceneInitRejectError("blocks_ranking_size layout rejected by validate_layout")
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
        # Three consecutive target slots on one randomly angled line (see
        # TARGET_GAP_RANGE_M above for the gap-bound rationale).
        (self.block1_target_pose,
         self.block2_target_pose,
         self.block3_target_pose) = self._sample_ranked_targets()
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
        return (
            self._blocks_on_ordered_line(block1_pose, block2_pose, block3_pose)
            and self.is_left_gripper_open()
            and self.right_arm_stationary()
        )

    @staticmethod
    def _blocks_on_ordered_line(p1, p2, p3) -> bool:
        """Check three blocks sit on one x-ordered line near the x axis.

        The tilted target lines replace the old shared-y relation, so success
        now means: strictly increasing x (guaranteed by the +-15 deg line
        sampling), the middle block within one placement tolerance of the
        end-to-end line, and each adjacent gap inside the sampled band with
        drift slack.
        """
        if not (p1[0] < p2[0] < p3[0]):
            return False
        axis = p3[:2] - p1[:2]
        # A line drifting past ~21.9 deg from the x axis cannot come
        # from a +-15 deg sample plus placement noise; reject it as a
        # failure.
        if abs(axis[1]) > TARGET_LINE_REJECT_SLOPE * abs(axis[0]):
            return False
        axis_len = np.linalg.norm(axis)
        perp = abs(axis[0] * (p2[1] - p1[1]) - axis[1] * (p2[0] - p1[0])) / axis_len
        gaps_ok = all(
            TARGET_GAP_RANGE_M[0] - 0.04
            <= float(np.linalg.norm(b[:2] - a[:2]))
            <= TARGET_GAP_RANGE_M[1] + 0.06
            for a, b in ((p1, p2), (p2, p3))
        )
        return bool(perp < 0.03 and gaps_ok)
