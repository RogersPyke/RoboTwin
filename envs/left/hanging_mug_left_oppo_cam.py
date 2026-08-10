"""RoboTwin entry point for hanging_mug_left_oppo_cam.

The semantic implementation lives in hanging_mug_left_impl.  This module only selects
the head-camera configuration for the variant; it contains no actor or expert
algorithm.
"""

from __future__ import annotations

from typing import Literal

from .hanging_mug_left_impl import HangingMugLeftImpl


class hanging_mug_left_oppo_cam(HangingMugLeftImpl):
    camera_variant: Literal["left_oppo_cam"] = "left_oppo_cam"
