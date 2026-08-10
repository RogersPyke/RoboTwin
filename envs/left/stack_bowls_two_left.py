"""RoboTwin entry point for stack_bowls_two_left.

The semantic implementation lives in stack_bowls_two_left_impl.  This module only selects
the head-camera configuration for the variant; it contains no actor or expert
algorithm.
"""

from __future__ import annotations

from typing import Literal

from .stack_bowls_two_left_impl import StackBowlsTwoLeftImpl


class stack_bowls_two_left(StackBowlsTwoLeftImpl):
    camera_variant: Literal["central_cam"] = "central_cam"
