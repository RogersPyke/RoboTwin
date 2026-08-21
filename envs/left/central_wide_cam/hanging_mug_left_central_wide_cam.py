"""RoboTwin entry point for hanging_mug_left_central_wide_cam.

The semantic implementation lives in hanging_mug_left_impl.  This module only selects
the head-camera configuration for the variant; it contains no actor or expert
algorithm.
"""

from __future__ import annotations

from typing import Literal

from ..impl.hanging_mug_left_impl import HangingMugLeftImpl


class hanging_mug_left_central_wide_cam(HangingMugLeftImpl):
    camera_variant: Literal["central_wide_cam"] = "central_wide_cam"
