"""RoboTwin entry point for place_dual_shoes_left.

The semantic implementation lives in place_dual_shoes_left_impl.  This module only selects
the head-camera configuration for the variant; it contains no actor or expert
algorithm.
"""

from __future__ import annotations

from typing import Literal

from .place_dual_shoes_left_impl import PlaceDualShoesLeftImpl


class place_dual_shoes_left(PlaceDualShoesLeftImpl):
    camera_variant: Literal["central_cam"] = "central_cam"
