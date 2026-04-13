"""
Purpose: Utility functions and constants for perturbation mixin.
Dependencies: logging
Usage:
    - Import segment validation functions
    - Import default configuration constants
    - Used by _pert_mixin.py

@input:  segment configuration dict
@output: validated configuration or raise error
@scenario: Validate and merge segment configurations for perturbation tasks.

Design Philosophy:
    - Each segment MUST explicitly specify 'enabled' key
    - Missing 'enabled' key raises ValueError with clear error message
    - Other parameters are optional and merged with defaults
    - No silent defaults for 'enabled' to force explicit configuration

Segment Configuration Parameters:
    Required:
        - enabled: bool, must be True or False explicitly

    Position/Orientation Jitter (applied to target pose):
        - xy_jitter: float, max random offset in meters for x and y axes
                    Example: 0.010 means each axis gets random offset in [-0.010, 0.010]
        - yaw_jitter_deg: float, max random yaw rotation in degrees
                         Example: 8.0 means random rotation in [-8.0, 8.0] degrees

    Waypoint Chain Parameters (for trajectory planning):
        - waypoint_xy_radius: float, max lateral distance from direct path in meters (default: 0.08)
                             Example: 0.08 means waypoints can deviate up to 8cm sideways
        - waypoint_z_jitter: float, max z offset for waypoints in meters (default: 0.05)
        - orientation_jitter_deg: float, max orientation change for waypoints in degrees (default: 10.0)

    RRT Anchor Parameters (alternative trajectory generation):
        - rrt_lateral_xy: float, max lateral offset for RRT anchor in meters (default: 0.10)
        - rrt_z_jitter: float, max z jitter for RRT anchor in meters (default: 0.04)

    Common Anchor Parameters (shared by both strategies):
        - anchor_ratio_min: float, min ratio along path for anchor point (default: 0.25)
        - anchor_ratio_max: float, max ratio along path for anchor point (default: 0.75)

    Planning Parameters:
        - candidate_trials: int, number of candidate paths to generate (default: 6)
        - fallback_to_direct: bool, use direct path if all candidates fail (default: True)
"""

import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


# ==============================================================================
# Default Segment Configuration Constants
# ==============================================================================

DEFAULT_SEGMENT_PARAMS: Dict[str, Any] = {
    # Position/Orientation Jitter
    "xy_jitter": 0.0,
    "yaw_jitter_deg": 0.0,
    # Common Anchor Parameters (shared by both strategies)
    "anchor_ratio_min": 0.25,
    "anchor_ratio_max": 0.75,
    # Waypoint Chain Parameters
    "waypoint_xy_radius": 0.08,
    "waypoint_z_jitter": 0.05,
    "orientation_jitter_deg": 10.0,
    # RRT Anchor Parameters
    "rrt_lateral_xy": 0.10,
    "rrt_z_jitter": 0.04,
    # Planning Parameters
    "candidate_trials": 6,
    "fallback_to_direct": True,
}


# ==============================================================================
# Validation Functions
# ==============================================================================


def validate_segment_config(
    segment_cfg: Optional[Dict[str, Any]],
    segment_name: str,
    task_name: str,
    global_defaults: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Validate a single segment configuration.

    @input:
        segment_cfg: Dict, segment configuration or None
        segment_name: str, name for error messages (e.g., "grasp_segment[0]")
        task_name: str, task class name for error context
        global_defaults: Dict, global defaults from config file (optional)
    @output:
        Dict, validated and merged configuration with all required keys
    @scenario:
        Validate that 'enabled' key exists, merge with defaults for missing keys.
        Raise ValueError if segment_cfg is None or missing 'enabled'.

    @param segment_cfg: Configuration dict for a single segment
    @param segment_name: Human-readable segment name for error messages
    @param task_name: Task class name for error context
    @param global_defaults: Global defaults from config file (lower priority than segment_cfg)

    Merge Priority (highest to lowest):
        1. segment_cfg (task file explicit config)
        2. global_defaults (config file defaults)
        3. DEFAULT_SEGMENT_PARAMS (hardcoded defaults)

    Example valid input:
        {"enabled": True, "xy_jitter": 0.010}
    Example invalid input (will raise):
        None  -> ValueError
        {}    -> ValueError (missing 'enabled')
        {"xy_jitter": 0.01} -> ValueError (missing 'enabled')
    """
    if segment_cfg is None:
        error_msg = (
            f"[{task_name}] Segment '{segment_name}' configuration is None. "
            f"Each segment MUST specify 'enabled' key explicitly. "
            f"Set 'enabled: True' or 'enabled: False'."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    if "enabled" not in segment_cfg:
        error_msg = (
            f"[{task_name}] Segment '{segment_name}' missing required key 'enabled'. "
            f"Each segment MUST specify 'enabled: True' or 'enabled: False'. "
            f"Provided keys: {list(segment_cfg.keys())}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Merge: hardcoded defaults < config file defaults < segment config
    merged = dict(DEFAULT_SEGMENT_PARAMS)
    if global_defaults:
        merged.update(global_defaults)
    merged.update(segment_cfg)

    if not isinstance(merged["enabled"], bool):
        error_msg = (
            f"[{task_name}] Segment '{segment_name}' has invalid 'enabled' type. "
            f"Expected bool, got {type(merged['enabled']).__name__}."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.debug(
        f"[{task_name}] Segment '{segment_name}' validated: enabled={merged['enabled']}"
    )
    return merged


def validate_segments_list(
    segments: Optional[List[Dict[str, Any]]],
    action_type: str,
    task_name: str,
    expected_count: int = 2,
    global_defaults: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Validate a list of segment configurations (for grasp/place actions).

    @input:
        segments: List[Dict], list of segment configurations
        action_type: str, "grasp" or "place" for error messages
        task_name: str, task name for error context
        expected_count: int, expected number of segments (default 2)
        global_defaults: Dict, global defaults from config file (optional)
    @output:
        List[Dict], validated segment configurations
    @scenario:
        Validate each segment in the list, ensure count matches expected.

    @param segments: List of segment configuration dicts
    @param action_type: Type of action ("grasp" or "place")
    @param task_name: Task class name for error context
    @param expected_count: Expected number of segments in the list
    @param global_defaults: Global defaults from config file

    Note:
        - grasp_actor has 2 move segments: approach + descent
        - place_actor has 2 move segments: approach + descent
    """
    if segments is None:
        error_msg = (
            f"[{task_name}] {action_type.upper()}_SEGMENTS is None. "
            f"Must provide a list of {expected_count} segment configurations."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    if not isinstance(segments, list):
        error_msg = (
            f"[{task_name}] {action_type.upper()}_SEGMENTS must be a list, "
            f"got {type(segments).__name__}."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    if len(segments) != expected_count:
        error_msg = (
            f"[{task_name}] {action_type.upper()}_SEGMENTS has {len(segments)} segments, "
            f"expected {expected_count}. "
            f"Each {action_type}_actor has exactly {expected_count} move segments."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    validated = []
    for idx, seg in enumerate(segments):
        seg_name = f"{action_type}_segment[{idx}]"
        validated.append(
            validate_segment_config(seg, seg_name, task_name, global_defaults)
        )

    logger.debug(f"[{task_name}] Validated {len(validated)} segments for {action_type}")
    return validated


def validate_single_segment(
    segment: Optional[Dict[str, Any]],
    action_type: str,
    task_name: str,
    global_defaults: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Validate a single segment configuration (for move_by_displacement, back_to_origin).

    @input:
        segment: Dict, segment configuration
        action_type: str, action type name for error messages
        task_name: str, task name for error context
        global_defaults: Dict, global defaults from config file (optional)
    @output:
        Dict, validated segment configuration
    @scenario:
        Validate single segment for simple movement actions.

    @param segment: Segment configuration dict
    @param action_type: Type of action (e.g., "move", "lift", "retreat")
    @param task_name: Task class name for error context
    @param global_defaults: Global defaults from config file
    """
    seg_name = f"{action_type}_segment"
    return validate_segment_config(segment, seg_name, task_name, global_defaults)


def merge_segment_config(
    base_cfg: Dict[str, Any],
    override_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Merge two segment configurations with override taking precedence.

    @input:
        base_cfg: Dict, base configuration
        override_cfg: Dict, override configuration
    @output:
        Dict, merged configuration
    @scenario:
        Merge configurations when multiple layers need to be combined.
    """
    result = dict(base_cfg)
    result.update(override_cfg)
    return result
