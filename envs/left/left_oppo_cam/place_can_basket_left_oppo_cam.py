"""RoboTwin entry point for place_can_basket_left_oppo_cam.

The semantic implementation lives in place_can_basket_left_impl.  This module only selects
the head-camera configuration for the variant; it contains no actor or expert
algorithm.
"""

from __future__ import annotations

from typing import Literal

from ..impl.place_can_basket_left_impl import PlaceCanBasketLeftImpl


class place_can_basket_left_oppo_cam(PlaceCanBasketLeftImpl):
    camera_variant: Literal["left_oppo_cam"] = "left_oppo_cam"
