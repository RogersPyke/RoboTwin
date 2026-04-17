#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wrapper base classes for ACT/DP/TinyVLA training and evaluation.

This module provides:
- BaseTrWrapper: Abstract base class for training wrappers
- BaseEvWrapper: Abstract base class for evaluation wrappers
- config_util: Common configuration utilities
- checkpoint_util: Model-specific checkpoint handling utilities

Usage:
    from wrapper_base import BaseTrWrapper, BaseEvWrapper
    from wrapper_base.config_util import load_unified_config, merge_global_and_model_config
    from wrapper_base.checkpoint_util import extract_step_from_ckpt_name, package_checkpoints_for_eval
"""

from .tr_wrapper_base import BaseTrWrapper
from .ev_wrapper_base import BaseEvWrapper

__all__ = [
    "BaseTrWrapper",
    "BaseEvWrapper",
]
