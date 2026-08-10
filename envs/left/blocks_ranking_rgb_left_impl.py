"""LEFT_TASK_DESIGN:
Source task: blocks_ranking_rgb
Single-arm semantic change: source chooses each block's arm from its x sign and
    switches arms with ``back_to_origin(opposite)``; the derived task samples
    all three blocks in one left workspace and ranks them serially with the
    left arm only, returning home between blocks.
Left workspace manifest: blocks_ranking_rgb, version 0
Actors and clearance: block1/2/3 (dynamic coloured boxes, pairwise min
    0.10 m), ordered left-arm target line in x
    [-0.38,-0.28] at a shared y in [-0.20,-0.10].
Expert sequence: left pick red, place on line, return home; left pick green,
    place, return home; left pick blue, place, return home.
Success predicate: preserve source ordered-x and common-y ranking relation;
    remove the right-gripper-open condition; require left open and right home.
Instruction change: {A}=red block, {B}=green block, {C}=blue block, {a}..{c}
    =left; wording says the left arm ranks three coloured blocks.
Pilot evidence: central-cam 30-seed pilot 0.80 (24/30 success), manifest hash
    308a70af1adf07a2
"""

from __future__ import annotations

import numpy as np

from .left_task_base import LeftTaskBase, SceneRejectedError
from .left_task_manifests import get_manifest
from ..utils import *  # noqa: F401,F403


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
            for pose in (block1, block2, block3):
                pose.p[2] = 0.765
            if (self._pairwise_clear(block1, block2, clearance)
                    and self._pairwise_clear(block2, block3, clearance)
                    and self._pairwise_clear(block1, block3, clearance)):
                return {"block1": block1, "block2": block2, "block3": block3}
        raise SceneRejectedError(
            "could not sample a feasible blocks_ranking_rgb layout in 128 attempts"
        )

    @staticmethod
    def _pairwise_clear(pose_a, pose_b, clearance: float) -> bool:
        return bool(np.linalg.norm(pose_a.p[:2] - pose_b.p[:2]) >= clearance)

    def load_actors(self) -> None:
        layout = self.sample_layout()
        if not self.validate_layout(layout):
            raise SceneRejectedError("blocks_ranking_rgb layout rejected by validate_layout")
        size = np.random.uniform(0.015, 0.025)
        half_size = (size, size, size)
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
        # Ordered target line inside the left workspace: red < green < blue x.
        # Adjacent slots are 0.09 m apart so a post-release slide of a few
        # centimetres cannot reorder the blocks (see the leave-a-drop retreat).
        y_pose = np.random.uniform(-0.2, -0.1)
        self.block1_target_pose = [
            np.random.uniform(-0.42, -0.40), y_pose, 0.74 + self.table_z_bias,
        ] + [0, 1, 0, 0]
        self.block2_target_pose = [
            np.random.uniform(-0.33, -0.31), y_pose, 0.74 + self.table_z_bias,
        ] + [0, 1, 0, 0]
        self.block3_target_pose = [
            np.random.uniform(-0.24, -0.22), y_pose, 0.74 + self.table_z_bias,
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
        eps = [0.13, 0.03]
        return (
            bool(np.all(abs(block1_pose[:2] - block2_pose[:2]) < eps))
            and bool(np.all(abs(block2_pose[:2] - block3_pose[:2]) < eps))
            and bool(block1_pose[0] < block2_pose[0])
            and bool(block2_pose[0] < block3_pose[0])
            and self.is_left_gripper_open()
            and self.right_arm_stationary()
        )
