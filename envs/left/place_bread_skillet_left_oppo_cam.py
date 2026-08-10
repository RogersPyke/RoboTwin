"""RoboTwin entry point for place_bread_skillet_left_oppo_cam.

The semantic implementation lives in place_bread_skillet_left_impl.  This module only selects
the head-camera configuration for the variant; it contains no actor or expert
algorithm.
"""

from __future__ import annotations

from typing import Literal

from .place_bread_skillet_left_impl import PlaceBreadSkilletLeftImpl


class place_bread_skillet_left_oppo_cam(PlaceBreadSkilletLeftImpl):
    camera_variant: Literal["left_oppo_cam"] = "left_oppo_cam"
