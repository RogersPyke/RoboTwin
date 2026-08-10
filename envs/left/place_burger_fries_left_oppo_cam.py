"""RoboTwin entry point for place_burger_fries_left_oppo_cam.

The semantic implementation lives in place_burger_fries_left_impl.  This module only selects
the head-camera configuration for the variant; it contains no actor or expert
algorithm.
"""

from __future__ import annotations

from typing import Literal

from .place_burger_fries_left_impl import PlaceBurgerFriesLeftImpl


class place_burger_fries_left_oppo_cam(PlaceBurgerFriesLeftImpl):
    camera_variant: Literal["left_oppo_cam"] = "left_oppo_cam"
