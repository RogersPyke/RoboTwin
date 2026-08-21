"""RoboTwin entry point for blocks_ranking_size_left_oppo_cam.

The semantic implementation lives in blocks_ranking_size_left_impl.  This module only selects
the head-camera configuration for the variant; it contains no actor or expert
algorithm.
"""

from __future__ import annotations

from typing import Literal

from ..impl.blocks_ranking_size_left_impl import BlocksRankingSizeLeftImpl


class blocks_ranking_size_left_oppo_cam(BlocksRankingSizeLeftImpl):
    camera_variant: Literal["left_oppo_cam"] = "left_oppo_cam"
