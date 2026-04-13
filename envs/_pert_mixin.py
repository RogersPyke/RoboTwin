"""
Purpose: Perturbation mixin for *_pert tasks with segment-level control.
Dependencies:
    - logging, math, numpy, transforms3d
    - envs._pert_utils (validation functions)
    - envs._base_task (parent class)
    - envs.utils.action (Action class)

Usage Example:
    # In task file (e.g., hanging_mug_pert.py):

    class hanging_mug_pert(PerturbationMixin, hanging_mug):

        def _grasp_mug_initial(self, arm_tag):
            return self._wrap_grasp(
                actor=self.mug, arm_tag=arm_tag, pre_grasp_dis=0.05,
                segments=[
                    {"enabled": True, "xy_jitter": 0.010, "yaw_jitter_deg": 8.0},
                    {"enabled": True, "xy_jitter": 0.002, "yaw_jitter_deg": 2.0},
                ]
            )

        def _lift(self, arm_tag, z):
            return self._wrap_move(
                arm_tag=arm_tag, z=z,
                segment={"enabled": True, "xy_jitter": 0.008}
            )

        def play_once(self):
            self.move(self._grasp_mug_initial("left"))
            self.move(self._lift("left", z=0.1))
            # ... more actions

@input: kwargs from collect_data.py with perturbation config
@output: Augmented trajectories with segment-level perturbation
@scenario: Generate diverse training data for imitation learning.

Design Philosophy:
    - Segment-level perturbation control
    - Each segment MUST specify 'enabled' key explicitly
    - Semantic wrapper functions in task files provide clear configuration
    - No silent defaults - explicit configuration required
"""

import math
import logging
from copy import deepcopy
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import transforms3d as t3d

from ._pert_utils import (
    validate_segment_config,
    validate_segments_list,
    validate_single_segment,
    DEFAULT_SEGMENT_PARAMS,
)

logger = logging.getLogger(__name__)


class PerturbationMixin:
    """
    Mixin class for perturbation tasks with segment-level control.

    This mixin overrides action generation methods to apply perturbation
    based on segment configurations. Task classes should define semantic
    wrapper functions that call _wrap_grasp, _wrap_place, _wrap_move.

    Key Design:
        - grasp_actor: 2 move segments (approach + descent)
        - place_actor: 2 move segments (approach + descent)
        - move_by_displacement: 1 move segment
        - back_to_origin: 1 move segment

    Each segment requires explicit 'enabled' key in configuration.

    Configuration Priority:
        1. Segment config passed to wrapper function (highest)
        2. No fallback to global config - must be explicit
    """

    # ==========================================================================
    # Configuration Loading
    # ==========================================================================

    def _load_pert_cfg(self, kwargs: dict) -> None:
        """
        Load perturbation configuration from kwargs.

        @input:
            kwargs: dict, configuration from collect_data.py
        @output: None (sets instance attributes)
        @scenario:
            Initialize perturbation settings and default segment params.
            Note: END_RESET_TO_INIT is handled by _base_task._init_task_env_.

        @param kwargs: Configuration dictionary from task config file
        """
        cfg = kwargs.get("perturbation", {}) or {}

        self._pert_enabled = bool(cfg.get("enabled", True))

        # Load default segment parameters from config file
        defaults = cfg.get("defaults", {}) or {}
        self._pert_defaults = {
            "xy_jitter": defaults.get("xy_jitter", 0.0),
            "yaw_jitter_deg": defaults.get("yaw_jitter_deg", 0.0),
            "anchor_ratio_min": defaults.get("anchor_ratio_min", 0.25),
            "anchor_ratio_max": defaults.get("anchor_ratio_max", 0.75),
            "waypoint_xy_radius": defaults.get("waypoint_xy_radius", 0.08),
            "waypoint_z_jitter": defaults.get("waypoint_z_jitter", 0.05),
            "orientation_jitter_deg": defaults.get("orientation_jitter_deg", 10.0),
            "rrt_lateral_xy": defaults.get("rrt_lateral_xy", 0.10),
            "rrt_z_jitter": defaults.get("rrt_z_jitter", 0.04),
            "candidate_trials": defaults.get("candidate_trials", 6),
            "fallback_to_direct": defaults.get("fallback_to_direct", True),
        }

        self._pert_meta = {
            "enabled": self._pert_enabled,
            "defaults": self._pert_defaults,
        }

        logger.debug(
            f"[{self.__class__.__name__}] Perturbation config loaded: "
            f"enabled={self._pert_enabled}, defaults={self._pert_defaults}"
        )

    def setup_demo(self, *args, **kwargs) -> Any:
        """
        Override setup_demo to load perturbation configuration.

        @input: args, kwargs from parent setup_demo
        @output: Result from parent setup_demo
        @scenario: Initialize perturbation before task setup
        """
        self._load_pert_cfg(kwargs)
        return super().setup_demo(*args, **kwargs)

    # ==========================================================================
    # Pose Perturbation Utilities
    # ==========================================================================

    def _jitter_xy(self, pose_7d: List[float], jitter: float) -> List[float]:
        """
        Apply XY jitter to a 7D pose.

        @input:
            pose_7d: List[float], [x, y, z, qx, qy, qz, qw]
            jitter: float, max jitter distance in meters
        @output:
            List[float], jittered pose
        @scenario: Add random XY offset for position diversity

        @param pose_7d: 7D pose [position, quaternion]
        @param jitter: Maximum jitter distance in meters (uniform distribution)
        """
        if pose_7d is None or jitter <= 0:
            return pose_7d

        pose = deepcopy(pose_7d)
        pose[0] += float(np.random.uniform(-jitter, jitter))
        pose[1] += float(np.random.uniform(-jitter, jitter))
        return pose

    def _jitter_yaw(self, pose_7d: List[float], jitter_deg: float) -> List[float]:
        """
        Apply yaw jitter to a 7D pose.

        @input:
            pose_7d: List[float], [x, y, z, qx, qy, qz, qw]
            jitter_deg: float, max yaw jitter in degrees
        @output:
            List[float], jittered pose
        @scenario: Add random yaw rotation for orientation diversity

        @param pose_7d: 7D pose [position, quaternion]
        @param jitter_deg: Maximum yaw jitter in degrees (uniform distribution)
        """
        if pose_7d is None or jitter_deg <= 0:
            return pose_7d

        pose = np.array(deepcopy(pose_7d), dtype=np.float64)
        yaw = float(np.random.uniform(-jitter_deg, jitter_deg)) * math.pi / 180.0
        q_orig = pose[3:7]
        q_yaw = t3d.euler.euler2quat(0.0, 0.0, yaw, axes="sxyz")
        q_new = t3d.quaternions.qmult(q_yaw, q_orig)
        q_new = q_new / np.linalg.norm(q_new)
        pose[3:7] = q_new
        return pose.tolist()

    def _apply_segment_perturbation(
        self,
        pose_7d: List[float],
        segment_cfg: Dict[str, Any],
    ) -> List[float]:
        """
        Apply perturbation to a pose based on segment configuration.

        @input:
            pose_7d: List[float], target pose
            segment_cfg: Dict, validated segment configuration
        @output:
            List[float], perturbed pose
        @scenario: Apply XY and yaw perturbation if segment is enabled

        @param pose_7d: 7D target pose
        @param segment_cfg: Validated segment configuration dict
        """
        if pose_7d is None:
            return pose_7d

        if not segment_cfg.get("enabled", False):
            logger.debug(f"Segment disabled, skipping perturbation")
            return pose_7d

        pose = deepcopy(pose_7d)

        xy_jitter = segment_cfg.get("xy_jitter", 0)
        if xy_jitter > 0:
            pose[0] += float(np.random.uniform(-xy_jitter, xy_jitter))
            pose[1] += float(np.random.uniform(-xy_jitter, xy_jitter))

        yaw_jitter_deg = segment_cfg.get("yaw_jitter_deg", 0)
        if yaw_jitter_deg > 0:
            pose = self._jitter_yaw(pose, yaw_jitter_deg)

        logger.debug(
            f"Applied perturbation: xy_jitter={xy_jitter}, yaw_jitter_deg={yaw_jitter_deg}"
        )
        return pose

    # ==========================================================================
    # Trajectory Augmentation Utilities
    # ==========================================================================

    def _normalize_pose_7d(self, pose_7d: Any) -> Optional[List[float]]:
        """
        Normalize pose to 7D list format.

        @input:
            pose_7d: Any, pose in various formats (sapien.Pose, list, np.ndarray)
        @output:
            List[float] or None, normalized [x, y, z, qx, qy, qz, qw]
        @scenario: Convert pose to standard format for processing
        """
        if pose_7d is None:
            return None
        if hasattr(pose_7d, "p") and hasattr(pose_7d, "q"):
            return pose_7d.p.tolist() + pose_7d.q.tolist()
        return np.array(deepcopy(pose_7d), dtype=np.float64).tolist()

    def _apply_orientation_noise(
        self,
        pose_7d: List[float],
        jitter_deg: float,
    ) -> List[float]:
        """
        Apply random orientation noise to a pose.

        @input:
            pose_7d: List[float], 7D pose
            jitter_deg: float, max jitter in degrees for each Euler angle
        @output:
            List[float], pose with orientation noise
        @scenario: Add random RPY noise for waypoint orientation diversity
        """
        if pose_7d is None or jitter_deg <= 0:
            return pose_7d

        pose = np.array(deepcopy(pose_7d), dtype=np.float64)
        rpy = np.random.uniform(-jitter_deg, jitter_deg, size=3) * math.pi / 180.0
        q_delta = t3d.euler.euler2quat(
            float(rpy[0]), float(rpy[1]), float(rpy[2]), axes="sxyz"
        )
        q_new = t3d.quaternions.qmult(q_delta, pose[3:7])
        q_new = q_new / np.linalg.norm(q_new)
        pose[3:7] = q_new
        return pose.tolist()

    def _build_waypoint_chain_candidates(
        self,
        start_pose: List[float],
        target_pose: List[float],
        segment_cfg: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Build waypoint chain candidates for trajectory augmentation.

        @input:
            start_pose: List[float], current end-effector pose
            target_pose: List[float], target pose
            segment_cfg: Dict, segment configuration
        @output:
            List[Dict], list of candidate waypoint chains
        @scenario: Generate diverse waypoint paths between start and target

        @param start_pose: Starting pose for trajectory
        @param target_pose: Target pose for trajectory
        @param segment_cfg: Segment configuration with waypoint parameters
        """
        candidates = []

        xy_radius = segment_cfg.get("waypoint_xy_radius", 0.08)
        z_jitter = segment_cfg.get("waypoint_z_jitter", 0.05)
        orientation_jitter = segment_cfg.get("orientation_jitter_deg", 10.0)
        ratio_min = segment_cfg.get("anchor_ratio_min", 0.25)
        ratio_max = segment_cfg.get("anchor_ratio_max", 0.75)
        trials = segment_cfg.get("candidate_trials", 6)

        start_xyz = np.array(start_pose[:3], dtype=np.float64)
        target_xyz = np.array(target_pose[:3], dtype=np.float64)
        direct_vec = target_xyz - start_xyz

        for _ in range(trials):
            ratio = float(np.random.uniform(ratio_min, ratio_max))
            base_xyz = start_xyz + ratio * direct_vec
            offset = np.array(
                [
                    np.random.uniform(-xy_radius, xy_radius),
                    np.random.uniform(-xy_radius, xy_radius),
                    np.random.uniform(-z_jitter, z_jitter),
                ],
                dtype=np.float64,
            )
            waypoint = deepcopy(target_pose)
            waypoint[:3] = (base_xyz + offset).tolist()
            waypoint = self._apply_orientation_noise(waypoint, orientation_jitter)

            candidates.append(
                {
                    "strategy": "waypoint_chain",
                    "waypoints": [waypoint],
                }
            )

        return candidates

    def _build_rrt_guided_candidates(
        self,
        start_pose: List[float],
        target_pose: List[float],
        direct_result: Optional[Dict],
        segment_cfg: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Build RRT-guided candidates for trajectory augmentation.

        @input:
            start_pose: List[float], current end-effector pose
            target_pose: List[float], target pose
            direct_result: Dict or None, result from direct path planning
            segment_cfg: Dict, segment configuration
        @output:
            List[Dict], list of RRT-guided candidates
        @scenario: Generate lateral offset waypoints based on direct path
        """
        if direct_result is None or direct_result.get("status") != "Success":
            return []

        candidates = []

        lateral_xy = segment_cfg.get("rrt_lateral_xy", 0.10)
        z_jitter = segment_cfg.get("rrt_z_jitter", 0.04)
        orientation_jitter = segment_cfg.get("orientation_jitter_deg", 10.0)
        ratio_min = segment_cfg.get("anchor_ratio_min", 0.25)
        ratio_max = segment_cfg.get("anchor_ratio_max", 0.75)
        trials = segment_cfg.get("candidate_trials", 6)

        step_count = max(1, int(direct_result["position"].shape[0]))
        path_scale = min(1.8, max(0.6, float(step_count) / 200.0))
        lateral_xy = lateral_xy * path_scale

        start_xyz = np.array(start_pose[:3], dtype=np.float64)
        target_xyz = np.array(target_pose[:3], dtype=np.float64)
        direct_vec = target_xyz - start_xyz
        horizontal_vec = np.array([direct_vec[0], direct_vec[1], 0.0], dtype=np.float64)
        norm_xy = float(np.linalg.norm(horizontal_vec))

        if norm_xy < 1e-8:
            rand = np.random.uniform(-1.0, 1.0, size=2)
            horizontal_vec = np.array([rand[0], rand[1], 0.0], dtype=np.float64)
            norm_xy = float(np.linalg.norm(horizontal_vec))
            if norm_xy < 1e-8:
                horizontal_vec = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                norm_xy = 1.0

        horizontal_vec /= norm_xy
        orthogonal_vec = np.array(
            [-horizontal_vec[1], horizontal_vec[0], 0.0], dtype=np.float64
        )

        for _ in range(trials):
            ratio = float(np.random.uniform(ratio_min, ratio_max))
            lateral = float(np.random.uniform(-lateral_xy, lateral_xy))
            anchor = start_xyz + ratio * direct_vec + lateral * orthogonal_vec
            anchor[2] += float(np.random.uniform(-z_jitter, z_jitter))

            waypoint = deepcopy(target_pose)
            waypoint[:3] = anchor.tolist()
            waypoint = self._apply_orientation_noise(waypoint, orientation_jitter)

            candidates.append(
                {
                    "strategy": "rrt_guided",
                    "waypoints": [waypoint],
                }
            )

        return candidates

    def _plan_single_segment(
        self,
        arm_tag: str,
        pose_7d: List[float],
        constraint_pose: Optional[List[float]],
        last_full_qpos: Optional[np.ndarray] = None,
        last_arm_qpos: Optional[np.ndarray] = None,
    ) -> Optional[Dict]:
        """
        Plan a single trajectory segment.

        @input:
            arm_tag: str, "left" or "right"
            pose_7d: List[float], target pose
            constraint_pose: List[float] or None, constraint for movement
            last_full_qpos: np.ndarray or None, previous full qpos
            last_arm_qpos: np.ndarray or None, previous arm qpos
        @output:
            Dict or None, planning result with position and velocity
        @scenario: Call robot planner for single segment
        """
        if arm_tag == "left":
            plan_fn = self.robot.left_plan_path
        else:
            plan_fn = self.robot.right_plan_path

        return plan_fn(
            pose_7d,
            constraint_pose=constraint_pose,
            last_full_qpos=last_full_qpos,
            last_arm_qpos=last_arm_qpos,
        )

    def _merge_segment_results(
        self,
        segment_results: List[Dict],
    ) -> Optional[Dict]:
        """
        Merge multiple segment results into a single trajectory.

        @input:
            segment_results: List[Dict], results from each segment
        @output:
            Dict or None, merged trajectory
        @scenario: Combine waypoint segments into complete trajectory
        """
        if not segment_results:
            return None

        position_list = []
        velocity_list = []

        for idx, result in enumerate(segment_results):
            position = result["position"]
            velocity = result["velocity"]

            if idx > 0 and position.shape[0] > 0:
                position = position[1:]
                velocity = velocity[1:]

            if position.shape[0] == 0:
                continue

            position_list.append(position)
            velocity_list.append(velocity)

        if not position_list:
            return None

        return {
            "status": "Success",
            "position": np.vstack(position_list),
            "velocity": np.vstack(velocity_list),
        }

    def _try_plan_waypoint_chain(
        self,
        arm_tag: str,
        target_pose: List[float],
        waypoints: List[List[float]],
        constraint_pose: Optional[List[float]],
    ) -> Optional[Dict]:
        """
        Try to plan a trajectory through waypoints.

        @input:
            arm_tag: str, "left" or "right"
            target_pose: List[float], final target pose
            waypoints: List[List[float]], intermediate waypoints
            constraint_pose: List[float] or None, movement constraint
        @output:
            Dict or None, merged trajectory result
        @scenario: Plan through all waypoints to target
        """
        if arm_tag == "left":
            now_full_qpos = self.robot.left_entity.get_qpos()
        else:
            now_full_qpos = self.robot.right_entity.get_qpos()

        segments = []
        now_arm_qpos = None

        for pose in list(waypoints) + [target_pose]:
            result = self._plan_single_segment(
                arm_tag=arm_tag,
                pose_7d=pose,
                constraint_pose=constraint_pose,
                last_full_qpos=now_full_qpos if len(segments) == 0 else None,
                last_arm_qpos=None if len(segments) == 0 else now_arm_qpos,
            )

            if result is None or result.get("status") != "Success":
                return None

            segments.append(result)
            now_arm_qpos = result["position"][-1]

        return self._merge_segment_results(segments)

    def _plan_augmented_result(
        self,
        arm_tag: str,
        target_pose: List[float],
        constraint_pose: Optional[List[float]],
        segment_cfg: Dict[str, Any],
    ) -> Optional[Dict]:
        """
        Plan augmented trajectory based on segment configuration.

        @input:
            arm_tag: str, "left" or "right"
            target_pose: List[float], target pose
            constraint_pose: List[float] or None, movement constraint
            segment_cfg: Dict, validated segment configuration
        @output:
            Dict or None, augmented trajectory result
        @scenario: Generate diverse trajectory through waypoints
        """
        start_pose = self._normalize_pose_7d(self.get_arm_pose(arm_tag))
        target_pose = self._normalize_pose_7d(target_pose)

        if start_pose is None or target_pose is None:
            return None

        if arm_tag == "left":
            now_full_qpos = self.robot.left_entity.get_qpos()
            direct_result = self.robot.left_plan_path(
                target_pose,
                constraint_pose=constraint_pose,
                last_full_qpos=now_full_qpos,
            )
        else:
            now_full_qpos = self.robot.right_entity.get_qpos()
            direct_result = self.robot.right_plan_path(
                target_pose,
                constraint_pose=constraint_pose,
                last_full_qpos=now_full_qpos,
            )

        if not segment_cfg.get("enabled", False):
            return direct_result

        candidates = []
        candidates.extend(
            self._build_waypoint_chain_candidates(start_pose, target_pose, segment_cfg)
        )
        candidates.extend(
            self._build_rrt_guided_candidates(
                start_pose, target_pose, direct_result, segment_cfg
            )
        )

        np.random.shuffle(candidates)

        for candidate in candidates:
            merged = self._try_plan_waypoint_chain(
                arm_tag=arm_tag,
                target_pose=target_pose,
                waypoints=candidate["waypoints"],
                constraint_pose=constraint_pose,
            )

            if merged is None:
                continue

            self._pert_meta["last_augmentation"] = {
                "strategy": candidate["strategy"],
                "waypoint_count": int(len(candidate["waypoints"])),
            }
            return merged

        if segment_cfg.get("fallback_to_direct", True):
            self._pert_meta["last_augmentation"] = {
                "strategy": "direct_fallback",
                "waypoint_count": 0,
            }
            return direct_result

        return None

    # ==========================================================================
    # Semantic Wrapper Functions
    # ==========================================================================

    def _wrap_grasp(
        self,
        actor,
        arm_tag,
        segments: List[Dict[str, Any]],
        **kwargs,
    ):
        """
        Wrap grasp_actor with segment-level perturbation configuration.

        @input:
            actor: Actor, object to grasp
            arm_tag: str or ArmTag, which arm to use
            segments: List[Dict], exactly 2 segment configurations
            **kwargs: Additional args for grasp_actor
        @output:
            Tuple[ArmTag, List[Action]], actions with perturbation config
        @scenario:
            Apply perturbation to grasp actions based on segment configs.
            segments[0] = approach phase
            segments[1] = descent phase (constrained)

        @param segments: List of exactly 2 segment configuration dicts.
            Each dict MUST contain 'enabled' key.

        Example:
            segments=[
                {"enabled": True, "xy_jitter": 0.010, "yaw_jitter_deg": 8.0},
                {"enabled": True, "xy_jitter": 0.002, "yaw_jitter_deg": 2.0},
            ]
        """
        task_name = self.__class__.__name__
        validated_segments = validate_segments_list(
            segments,
            "grasp",
            task_name,
            expected_count=2,
            global_defaults=getattr(self, "_pert_defaults", None),
        )

        arm, actions = super().grasp_actor(actor, arm_tag=arm_tag, **kwargs)

        if not self._pert_enabled or not self.need_plan:
            return arm, actions

        if not actions:
            return arm, actions

        move_idx = 0
        for action in actions:
            if action.action == "move":
                if move_idx < len(validated_segments):
                    seg_cfg = validated_segments[move_idx]
                    action.args["segment_cfg"] = seg_cfg
                    logger.debug(
                        f"[{task_name}] grasp segment[{move_idx}] config attached"
                    )
                move_idx += 1

        return arm, actions

    def _wrap_place(
        self,
        actor,
        arm_tag,
        target_pose,
        segments: List[Dict[str, Any]],
        **kwargs,
    ):
        """
        Wrap place_actor with segment-level perturbation configuration.

        @input:
            actor: Actor, object to place
            arm_tag: str or ArmTag, which arm to use
            target_pose: List[float], target pose for placement
            segments: List[Dict], exactly 2 segment configurations
            **kwargs: Additional args for place_actor
        @output:
            Tuple[ArmTag, List[Action]], actions with perturbation config
        @scenario:
            Apply perturbation to place actions based on segment configs.
            segments[0] = approach phase
            segments[1] = descent phase

        @param segments: List of exactly 2 segment configuration dicts.
            Each dict MUST contain 'enabled' key.

        Example:
            segments=[
                {"enabled": True, "xy_jitter": 0.012, "yaw_jitter_deg": 10.0},
                {"enabled": True, "xy_jitter": 0.003, "yaw_jitter_deg": 3.0},
            ]
        """
        task_name = self.__class__.__name__
        validated_segments = validate_segments_list(
            segments,
            "place",
            task_name,
            expected_count=2,
            global_defaults=getattr(self, "_pert_defaults", None),
        )

        arm, actions = super().place_actor(actor, arm_tag, target_pose, **kwargs)

        if not self._pert_enabled or not self.need_plan:
            return arm, actions

        if not actions:
            return arm, actions

        move_idx = 0
        for action in actions:
            if action.action == "move":
                if move_idx < len(validated_segments):
                    seg_cfg = validated_segments[move_idx]
                    action.args["segment_cfg"] = seg_cfg
                    logger.debug(
                        f"[{task_name}] place segment[{move_idx}] config attached"
                    )
                move_idx += 1

        return arm, actions

    def _wrap_move(
        self,
        arm_tag,
        segment: Dict[str, Any],
        **kwargs,
    ):
        """
        Wrap move_by_displacement with segment perturbation configuration.

        @input:
            arm_tag: str or ArmTag, which arm to use
            segment: Dict, single segment configuration
            **kwargs: Additional args for move_by_displacement (x, y, z, etc.)
        @output:
            Tuple[ArmTag, List[Action]], actions with perturbation config
        @scenario:
            Apply perturbation to single-segment movement actions.

        @param segment: Segment configuration dict.
            MUST contain 'enabled' key.

        Example:
            segment={"enabled": True, "xy_jitter": 0.008, "yaw_jitter_deg": 6.0}
        """
        task_name = self.__class__.__name__
        validated_segment = validate_single_segment(
            segment,
            "move",
            task_name,
            global_defaults=getattr(self, "_pert_defaults", None),
        )

        arm, actions = super().move_by_displacement(arm_tag, **kwargs)

        if not self._pert_enabled or not self.need_plan:
            return arm, actions

        if not actions:
            return arm, actions

        for action in actions:
            if action.action == "move":
                action.args["segment_cfg"] = validated_segment
                logger.debug(f"[{task_name}] move segment config attached")

        return arm, actions

    def _wrap_back_to_origin(
        self,
        arm_tag,
        segment: Dict[str, Any],
    ):
        """
        Wrap back_to_origin with segment perturbation configuration.

        @input:
            arm_tag: str or ArmTag, which arm to use
            segment: Dict, single segment configuration
        @output:
            Tuple[ArmTag, List[Action]], actions with perturbation config
        @scenario:
            Apply perturbation to return-to-origin movement.

        @param segment: Segment configuration dict.
            MUST contain 'enabled' key.

        Example:
            segment={"enabled": True, "xy_jitter": 0.010, "yaw_jitter_deg": 8.0}
        """
        task_name = self.__class__.__name__
        validated_segment = validate_single_segment(
            segment,
            "back_to_origin",
            task_name,
            global_defaults=getattr(self, "_pert_defaults", None),
        )

        arm, actions = super().back_to_origin(arm_tag)

        if not self._pert_enabled or not self.need_plan:
            return arm, actions

        if not actions:
            return arm, actions

        for action in actions:
            if action.action == "move":
                action.args["segment_cfg"] = validated_segment
                logger.debug(f"[{task_name}] back_to_origin segment config attached")

        return arm, actions

    # ==========================================================================
    # Override move_to_pose for Trajectory Augmentation
    # ==========================================================================

    def left_move_to_pose(
        self,
        pose,
        constraint_pose=None,
        segment_cfg: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Override left_move_to_pose to apply trajectory augmentation.

        @input:
            pose: target pose
            constraint_pose: constraint for movement
            segment_cfg: Dict or None, segment configuration from wrapper
            **kwargs: Additional arguments
        @output:
            Trajectory result or None
        @scenario:
            If segment_cfg is provided and enabled, augment trajectory.
            Otherwise, use direct planning.
        """
        if segment_cfg is None or not segment_cfg.get("enabled", False):
            return super().left_move_to_pose(
                pose=pose, constraint_pose=constraint_pose, **kwargs
            )

        if not self.plan_success:
            return None

        target_pose = self._normalize_pose_7d(pose)
        if target_pose is None:
            self.plan_success = False
            return None

        result = self._plan_augmented_result(
            arm_tag="left",
            target_pose=target_pose,
            constraint_pose=constraint_pose,
            segment_cfg=segment_cfg,
        )

        if result is None or result.get("status") != "Success":
            self.plan_success = False
            return None

        self.left_joint_path.append(deepcopy(result))
        return result

    def right_move_to_pose(
        self,
        pose,
        constraint_pose=None,
        segment_cfg: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Override right_move_to_pose to apply trajectory augmentation.

        @input:
            pose: target pose
            constraint_pose: constraint for movement
            segment_cfg: Dict or None, segment configuration from wrapper
            **kwargs: Additional arguments
        @output:
            Trajectory result or None
        @scenario:
            If segment_cfg is provided and enabled, augment trajectory.
            Otherwise, use direct planning.
        """
        if segment_cfg is None or not segment_cfg.get("enabled", False):
            return super().right_move_to_pose(
                pose=pose, constraint_pose=constraint_pose, **kwargs
            )

        if not self.plan_success:
            return None

        target_pose = self._normalize_pose_7d(pose)
        if target_pose is None:
            self.plan_success = False
            return None

        result = self._plan_augmented_result(
            arm_tag="right",
            target_pose=target_pose,
            constraint_pose=constraint_pose,
            segment_cfg=segment_cfg,
        )

        if result is None or result.get("status") != "Success":
            self.plan_success = False
            return None

        self.right_joint_path.append(deepcopy(result))
        return result

    # ==========================================================================
    # Override move() to pass segment_cfg to move_to_pose
    # ==========================================================================

    def move(
        self,
        actions_by_arm1,
        actions_by_arm2=None,
        save_freq=-1,
    ):
        """
        Override move to extract segment_cfg and pass to move_to_pose.

        @input:
            actions_by_arm1: Tuple[ArmTag, List[Action]]
            actions_by_arm2: Tuple[ArmTag, List[Action]] or None
            save_freq: int, save frequency
        @output:
            bool, success status
        @scenario:
            Execute actions while passing segment configuration to move_to_pose.
        """
        from .utils.action import ArmTag

        if self.plan_success is False:
            return False

        def get_actions(actions, arm_tag: ArmTag) -> list:
            if actions[1] is None:
                if actions[0][0] == arm_tag:
                    return actions[0][1]
                else:
                    return []
            else:
                if actions[0][0] == actions[0][1]:
                    raise ValueError("")
                if actions[0][0] == arm_tag:
                    return actions[0][1]
                else:
                    return actions[1][1]

        actions = [actions_by_arm1, actions_by_arm2]
        left_actions = get_actions(actions, "left")
        right_actions = get_actions(actions, "right")

        max_len = max(len(left_actions), len(right_actions))
        left_actions += [None] * (max_len - len(left_actions))
        right_actions += [None] * (max_len - len(right_actions))

        for left, right in zip(left_actions, right_actions):
            if (left is not None and left.arm_tag != "left") or (
                right is not None and right.arm_tag != "right"
            ):
                raise ValueError(
                    f"Invalid arm tag: {left.arm_tag if left else None} or "
                    f"{right.arm_tag if right else None}. Must be 'left' or 'right'."
                )

            if (
                left is not None
                and left.action == "move"
                and right is not None
                and right.action == "move"
            ):
                self.together_move_to_pose(
                    left_target_pose=left.target_pose,
                    right_target_pose=right.target_pose,
                    left_constraint_pose=left.args.get("constraint_pose"),
                    right_constraint_pose=right.args.get("constraint_pose"),
                )
                if self.plan_success is False:
                    return False
                continue
            else:
                control_seq = {
                    "left_arm": None,
                    "left_gripper": None,
                    "right_arm": None,
                    "right_gripper": None,
                }

                if left is not None:
                    if left.action == "move":
                        control_seq["left_arm"] = self.left_move_to_pose(
                            pose=left.target_pose,
                            constraint_pose=left.args.get("constraint_pose"),
                            segment_cfg=left.args.get("segment_cfg"),
                        )
                    else:
                        control_seq["left_gripper"] = self.set_gripper(
                            left_pos=left.target_gripper_pos, set_tag="left"
                        )
                    if self.plan_success is False:
                        return False

                if right is not None:
                    if right.action == "move":
                        control_seq["right_arm"] = self.right_move_to_pose(
                            pose=right.target_pose,
                            constraint_pose=right.args.get("constraint_pose"),
                            segment_cfg=right.args.get("segment_cfg"),
                        )
                    else:
                        control_seq["right_gripper"] = self.set_gripper(
                            right_pos=right.target_gripper_pos, set_tag="right"
                        )
                    if self.plan_success is False:
                        return False

            self.take_dense_action(control_seq)

        return True
