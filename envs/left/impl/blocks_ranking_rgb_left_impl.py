"""LEFT_TASK_DESIGN:
Source task: blocks_ranking_rgb
Single-arm semantic change: source chooses each block's arm from its x sign and
    switches arms with ``back_to_origin(opposite)``; the derived task samples
    all three blocks in one left workspace and ranks them serially with the
    left arm only, returning home between blocks.
Left workspace manifest: blocks_ranking_rgb, version 3 (2026-08-21)
Actors and clearance: block1/2/3 (dynamic coloured boxes, pairwise min
    0.10 m), shared pickup band x in [-0.45,-0.16], y in [-0.18,0.08],
    yaw up to 0.75; ordered target line sampled as three consecutive
    slots on one straight line in the xoy plane whose angle with the x
    axis stays within +-45 deg, with the two adjacent gaps drawn
    independently from [0.10,0.16] m, and every slot inside the same
    envelope as the blocks' own initial randomization band (the
    centred-wide batch envelope when that variant is active).
Expert sequence: left pick red, place on line, return home; left pick green,
    place, return home; left pick blue, place, return home.
Success predicate: preserve source ordered-x and common-y ranking relation;
    remove the right-gripper-open condition; require left open and right home.
Instruction change: {A}=red block, {B}=green block, {C}=blue block, {a}..{c}
    =left; wording says the left arm ranks three coloured blocks.
Pilot evidence: central-cam 50-seed pilots on version 1 (fixed-band targets)
    0.54 (27/50) and on version 2 (line-translated targets) 0.60 (30/50),
    manifest hash 3b292b9e1d535999
"""

from __future__ import annotations

import numpy as np

from ..left_task_base import LeftTaskBase, SceneInitRejectError
from ..left_task_manifests import get_manifest
from ...utils import *  # noqa: F401,F403

# ---------------------------------------------------------------------------
# LEGACY_RANDOMIZATION_PARAM
# The randomization protocols used before the current version 2 design.
# Intentionally unused: kept as a declared header constant so the previous
# geometry is reproducible and comparable.
#   Version 0 (original source-task geometry, manifest 308a70af1adf07a2):
#     - pickup band:  x in [-0.45, -0.22], y in [-0.08, 0.05], yaw in [0, 0.60]
#     - pairwise min clearance: 0.10 m
#     - target line:  red x in [-0.42, -0.40], green x in [-0.33, -0.31],
#                     blue x in [-0.24, -0.22], shared y in [-0.20, -0.10]
#   Version 1 (2026-08-13 interim widening, manifest 3b292b9e1d535999): the
#     pickup band covered the full left workspace; the target line widened to
#     red [-0.42,-0.39]/green [-0.34,-0.31]/blue [-0.26,-0.23] at shared y in
#     [-0.18,-0.08].  Superseded by version 2's line_x translation.
#   Version 2 (code-level retune of the version 1 geometry): an x-parallel
#     target line at line_x in [-0.20,-0.16], shared y in [-0.15,0.15], with
#     per-slot x offsets [0,0.01]/[0.14,0.15]/[0.28,0.29].  Superseded by
#     version 3's angled-line randomization.
LEGACY_RANDOMIZATION_PARAM: dict[str, object] = {
    "pickup_x": (-0.45, -0.22),
    "pickup_y": (-0.08, 0.05),
    "pickup_yaw_rad": (0.0, 0.60),
    "min_clearance_m": 0.10,
    "target_red_x": (-0.42, -0.40),
    "target_green_x": (-0.33, -0.31),
    "target_blue_x": (-0.24, -0.22),
    "target_y": (-0.20, -0.10),
    "manifest_hash": "308a70af1adf07a2",
    "version_1": {
        "pickup_x": (-0.45, -0.16),
        "pickup_y": (-0.18, 0.08),
        "pickup_yaw_rad": (0.0, 0.75),
        "min_clearance_m": 0.10,
        "target_red_x": (-0.42, -0.39),
        "target_green_x": (-0.34, -0.31),
        "target_blue_x": (-0.26, -0.23),
        "target_y": (-0.18, -0.08),
        "manifest_hash": "3b292b9e1d535999",
    },
    "version_2": {
        "target_line_x": (-0.20, -0.16),
        "target_y": (-0.15, 0.15),
        "target_slot_offsets": [(0.0, 0.01), (0.14, 0.15), (0.28, 0.29)],
    },
}

# Version 3 (2026-08-21) target-line randomization: three consecutive slots
# on one straight line in the xoy plane.  The line's angle with the x axis is
# uniform in +-45 deg (so block1 < block2 < block3 in x always holds), each
# adjacent gap is drawn independently from TARGET_GAP_RANGE_M, and every slot
# stays inside the blocks' own initial-randomization envelope.  The gap bounds
# exist because the smallest gap must keep the worst-case adjacent placed
# footprints clear (2 * 0.025 m block halves + the expert's ~0.02 m placement
# drift per slot), while the largest keeps the full <= 0.32 m span fitting
# inside the envelope diagonally even at the 45 deg extreme.
TARGET_LINE_MAX_ANGLE_RAD: float = float(np.deg2rad(45.0))
TARGET_GAP_RANGE_M: tuple[float, float] = (0.10, 0.16)


class BlocksRankingRgbLeftImpl(LeftTaskBase):
    """Left-arm-only red/green/blue block ranking task.

    Three coloured cubes are moved by the left arm onto an ordered target line
    whose x positions increase from red to green to blue at a common y.
    """

    def __init__(self) -> None:
        super().__init__()
        self.manifest = get_manifest("blocks_ranking_rgb")

    def sample_layout(self) -> dict[str, object]:
        specs = {
            name: self.manifest.actor_specs[name]
            for name in ("block1", "block2", "block3")
        }
        clearance = max(spec.minimum_clearance_m for spec in specs.values())
        for _ in range(128):
            block1 = self.sample_spec_pose(specs["block1"])
            block2 = self.sample_spec_pose(specs["block2"])
            block3 = self.sample_spec_pose(specs["block3"])
            if (self._pairwise_clear(block1, block2, clearance)
                    and self._pairwise_clear(block2, block3, clearance)
                    and self._pairwise_clear(block1, block3, clearance)):
                return {"block1": block1, "block2": block2, "block3": block3}
        raise SceneInitRejectError(
            "could not sample a feasible blocks_ranking_rgb layout in 128 attempts"
        )

    @staticmethod
    def _pairwise_clear(pose_a, pose_b, clearance: float) -> bool:
        return bool(np.linalg.norm(pose_a.p[:2] - pose_b.p[:2]) >= clearance)

    def _sample_ranked_targets(self) -> list[list[float]]:
        """Sample the three ordered target slots on one random straight line.

        The line lies in the xoy plane with its angle to the x axis uniform in
        +-45 deg, the two adjacent gaps are drawn independently from
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
        layout = self.sample_layout()
        if not self.validate_layout(layout):
            raise SceneInitRejectError("blocks_ranking_rgb layout rejected by validate_layout")
        size = np.random.uniform(0.015, 0.025)
        half_size = (size, size, size)
        # Spawn each block with its bottom exactly on the table (the box origin
        # is its geometric centre), mirroring blocks_ranking_size.  The old
        # fixed z=0.765 hovered small blocks up to 9 mm at spawn; the settle
        # loop hid it from observations, but exact contact keeps the spawn
        # geometry honest for both variants.
        for block in layout.values():
            block.p[2] = 0.741 + size
        self.block1 = create_box(
            scene=self, pose=layout["block1"], half_size=half_size,
            color=(1, 0, 0), name="box",
        )
        self.block2 = create_box(
            scene=self, pose=layout["block2"], half_size=half_size,
            color=(0, 1, 0), name="box",
        )
        self.block3 = create_box(
            scene=self, pose=layout["block3"], half_size=half_size,
            color=(0, 0, 1), name="box",
        )
        self.add_prohibit_area(self.block1, padding=0.05)
        self.add_prohibit_area(self.block2, padding=0.05)
        self.add_prohibit_area(self.block3, padding=0.05)
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
        self.place_ranked_block(self.block1, self.block1_target_pose)
        self.place_ranked_block(self.block2, self.block2_target_pose)
        self.place_ranked_block(self.block3, self.block3_target_pose)
        self.info["info"] = {
            "{A}": "red block",
            "{B}": "green block",
            "{C}": "blue block",
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
        now means: strictly increasing x (guaranteed by the +-45 deg line
        sampling), the middle block within one placement tolerance of the
        end-to-end line, and each adjacent gap inside the sampled band with
        drift slack.
        """
        if not (p1[0] < p2[0] < p3[0]):
            return False
        axis = p3[:2] - p1[:2]
        # A line drifting past ~56 deg from the x axis cannot come from a
        # +-45 deg sample plus placement noise; reject it as a failure.
        if abs(axis[1]) > 1.5 * abs(axis[0]):
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
