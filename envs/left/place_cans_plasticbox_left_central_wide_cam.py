"""RoboTwin entry point for place_cans_plasticbox_left_central_wide_cam.

The semantic implementation lives in place_cans_plasticbox_left_impl.  This module only selects
the head-camera configuration for the variant; it contains no actor or expert
algorithm.
"""

from __future__ import annotations

from typing import Literal

from .place_cans_plasticbox_left_impl import PlaceCansPlasticboxLeftImpl


class place_cans_plasticbox_left_central_wide_cam(PlaceCansPlasticboxLeftImpl):
    camera_variant: Literal["central_wide_cam"] = "central_wide_cam"
