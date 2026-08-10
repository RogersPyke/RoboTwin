"""RoboTwin entry point for blocks_ranking_rgb_left_central_wide_cam.

The semantic implementation lives in blocks_ranking_rgb_left_impl.  This module only selects
the head-camera configuration for the variant; it contains no actor or expert
algorithm.
"""

from __future__ import annotations

from typing import Literal

from .blocks_ranking_rgb_left_impl import BlocksRankingRgbLeftImpl


class blocks_ranking_rgb_left_central_wide_cam(BlocksRankingRgbLeftImpl):
    camera_variant: Literal["central_wide_cam"] = "central_wide_cam"
