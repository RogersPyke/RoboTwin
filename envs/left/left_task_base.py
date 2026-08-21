"""Shared base for left-arm-only RoboTwin task families.

Every derived task family in this repository follows one contract:

* The active arm is always the left Piper mounted at x=-0.30 m.  The right
  Piper at x=+0.30 m stays at its captured home qpos for the whole episode.
* The native qpos action has the fixed ordering
  ``[left_joint_0..5, left_gripper, right_joint_0..5, right_gripper]`` and
  ``LEFT_QPOS_DIM = 7`` is the projection boundary.
* Expert code may only command ``ArmTag("left")``.  A full-width policy action
  is permitted as input, but before every simulator step the suffix ``[7:]``
  is overwritten with the right home qpos captured after setup.
* ``take_action`` accepts only ``action_type == "qpos"``; ``ee`` control is
  rejected for these tasks.
* Every accepted episode appends one JSONL audit record.
* Each ``central_cam``/``left_oppo_cam`` pair shares one immutable
  ``TaskManifest``; camera is the only intended variation between the two
  variants.

``CameraVariant`` is exactly ``"central_cam"`` or ``"left_oppo_cam"``.
``CENTRAL_CAM`` uses ``None`` pose values to mean "preserve the asset-configured
``head_camera`` exactly".  ``LEFT_OPPO_CAM`` overwrites only ``position``,
``forward`` and ``left`` of that ``head_camera``.
"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import sapien

from .._base_task import Base_Task
from ..utils import *  # noqa: F401,F403  (ArmTag, Action, rand_pose, create_actor, ...)
from ..utils import ArmTag, Actor

ArmName = Literal["left"]
CameraVariant = Literal[
    "central_cam", "left_oppo_cam", "central_wide_cam",
    "cen_arm_right_wide_cam", "cen_arm_near_side_cam", "cen_arm_side_cam",
    "cen_arm_front_cam", "cen_arm_top_cam",
]

# Camera variants that share the centred-arm batch behaviour: the left Piper
# base is compensated back to x=0, every actor samples the shared centred-wide
# workspace and the idle right articulation is moved out of view.
CENTERED_BATCH_VARIANTS = (
    "cen_arm_right_wide_cam", "cen_arm_near_side_cam", "cen_arm_side_cam",
    "cen_arm_front_cam", "cen_arm_top_cam",
)

# Native qpos ordering: [left_joint_0..5, left_gripper, right_joint_0..5, right_gripper]
LEFT_QPOS_DIM = 7
NATIVE_QPOS_DIM = LEFT_QPOS_DIM * 2
# Any right-arm deviation beyond this (radians) invalidates the episode.
# The value is set from physics, not from an ideal of zero: the idle right arm
# is never commanded during a left-arm episode, so its real joints hold the
# home drive target with the controller's steady-state lag.  Measured across
# left-arm pilots this lag peaks near 4e-5 rad (one Piper elbow joint).  A
# tolerance of 1e-3 keeps that physical lag legal while still flagging any
# genuine right-arm motion, which would exceed it by orders of magnitude.
RIGHT_ARM_TOLERANCE = 1e-3
# Curobo builds its planning frame from the configured dual-arm separation, so
# that value must stay at the validated 0.60 m. After planners are initialized,
# the idle right articulation alone is moved to this world-x coordinate; it is
# consequently absent from the table and D435-W frame without changing the
# active left arm's planning coordinates.
CENTERED_BATCH_ARM_SEPARATION_M = 0.60
RIGHT_ARM_HIDDEN_WORLD_X_M = 5.0


@dataclass(frozen=True)
class CameraSpec:
    """One named head-camera configuration for a left-arm variant.

    ``None`` pose values preserve the asset-configured ``head_camera`` exactly.
    ``head_camera_type`` is a RoboTwin ``_camera_config.yml`` camera type; when
    set it replaces ``camera.head_camera_type`` for the variant (e.g. the wide
    ``D435-W``, fovy 60, versus the default ``D435``, fovy 37).
    """

    name: CameraVariant
    position: tuple[float, float, float] | None
    forward: tuple[float, float, float] | None
    left: tuple[float, float, float] | None
    head_camera_type: str | None = None


CENTRAL_CAM = CameraSpec("central_cam", None, None, None)
LEFT_OPPO_CAM = CameraSpec(
    "left_oppo_cam",
    (-0.30, 0.45, 1.35),
    (0.0, -0.6, -0.8),
    (1.0, 0.0, 0.0),
)
# The asset-configured central head camera pose (position (-0.032, -0.45,
# 1.35), forward (0, 0.6, -0.8), left (-1, 0, 0) in
# assets/embodiments/piper/config.yml) shifted 0.30 m to the +X side, with the
# wider head camera: same 320x240 sensor, fovy widened from 37 to 60 degrees.
CENTRAL_WIDE_CAM = CameraSpec(
    "central_wide_cam",
    (0.268, -0.45, 1.35),
    (0.0, 0.6, -0.8),
    (-1.0, 0.0, 0.0),
    head_camera_type="D435-W",
)
# Wide version of the default head camera shifted 0.30 m to the +X (right)
# side for the centred-arm batch: same pose as ``CENTRAL_WIDE_CAM`` — the
# asset-configured central head camera (position (-0.032, -0.45, 1.35),
# forward (0, 0.6, -0.8), left (-1, 0, 0)) with x_cam = x_cam + 0.30 — paired
# with the centred-arm workspace instead of the native left-arm mount.  Same
# 320x240 sensor, fovy widened from 37 to 60 degrees (D435-W).
CEN_ARM_RIGHT_WIDE_CAM = CameraSpec(
    "cen_arm_right_wide_cam",
    (0.268, -0.45, 1.35),
    (0.0, 0.6, -0.8),
    (-1.0, 0.0, 0.0),
    head_camera_type="D435-W",
)
# Near side view of the centred-arm batch.  O is the centre of the shared
# actor randomization envelope (CENTERED_WIDE_WORKSPACE, x/y both centred on
# the origin); C is the active left Piper base at (0, -0.45, 0.75) after the
# centred-arm compensation.  The camera sits at C shifted +X by 0.20 m and
# raised to z = 1.35 m, i.e. (0.20, -0.45, 1.35), and looks at the point
# (0, 0, 0.90) on the workspace axis — sight-line distance
# sqrt(0.20^2 + 0.45^2 + 0.45^2) = 0.6671 m with a 42.42-degree depression
# angle; the horizontal projection of the sight-line still passes through O
# at planar |OC| = sqrt(0.20^2 + 0.45^2) = 0.4924 m.  ``left`` is the
# horizontal ẑ×forward direction, so the image horizon is level (up has +Z
# and +Y components).  Wide sensor, inherited from the centred batch.
CEN_ARM_NEAR_SIDE_CAM = CameraSpec(
    "cen_arm_near_side_cam",
    (0.20, -0.45, 1.35),
    (-0.299813, 0.674579, -0.674579),
    (-0.913812, -0.406138, 0.0),
    head_camera_type="D435-W",
)

# Far side view of the centred-arm batch: the camera sits at
# (0.45, -0.45, 0.95) — diagonal (+X, -Y) corner outside the shared actor
# envelope — and looks horizontally (0-degree depression) at the point
# (0, 0, 0.95), the workspace centre at camera height, so the horizontal
# projection of the sight-line passes through O at planar distance
# sqrt(0.45^2 + 0.45^2) = 0.6364 m.  forward is the unit vector from the
# camera to the sight point, (-0.707107, 0.707107, 0); ``left`` is the
# horizontal ẑ×forward direction (-0.707107, -0.707107, 0), so the image
# horizon is level and coincides with the sight-line (up = +Z).  Wide sensor,
# inherited from the centred batch.
CEN_ARM_SIDE_CAM = CameraSpec(
    "cen_arm_side_cam",
    (0.45, -0.45, 0.95),
    (-0.707107, 0.707107, 0.0),
    (-0.707107, -0.707107, 0.0),
    head_camera_type="D435-W",
)

# Front view of the centred-arm batch: centred on x at (0, -0.30, 0.80),
# aiming along +Y with a 1:1.732 vertical:horizontal ratio — arctan(1/1.732)
# = 30 degrees of depression, forward (0, 0.866025, -0.5).  That sight-line
# meets the 0.74 m tabletop at (0, -0.196, 0.74), just past the workspace's
# -y edge (-0.15).  Reference geometry: the compensated left Piper base sits
# at (0, -0.45, 0.75); the envelope spans x(-0.20, 0.20), y(-0.15, 0.15).
CEN_ARM_FRONT_CAM = CameraSpec(
    "cen_arm_front_cam",
    (0.0, -0.30, 0.8),
    (0.0, 0.866025, -0.5),
    (-1.0, 0.0, 0.0),
    head_camera_type="D435-W",
)

# Top-down view of the centred-arm batch: the camera sits 0.76 m above the
# 0.74 m tabletop at (0, -0.10, 1.50) — directly over the workspace — with
# forward (0, 0, -1), a 90-degree depression angle perpendicular to the
# tabletop.  ``left`` stays (-1, 0, 0) as in the front/right views, so
# image-up is world +Y (the far side of the table) and the compensated left
# Piper base at (0, -0.45, 0.75) sits at the bottom of the frame.  With the
# D435-W fovy 60 the 0.76 m height covers a 0.877 m y extent (the frame
# spans y in [-0.54, 0.34], clearing the shared envelope's y(-0.15, 0.15)
# with margin on both edges) and a 1.17 m x extent (x in [-0.69, 0.49]).
CEN_ARM_TOP_CAM = CameraSpec(
    "cen_arm_top_cam",
    (0.0, -0.10, 1.50),
    (0.0, 0.0, -1.0),
    (-1.0, 0.0, 0.0),
    head_camera_type="D435-W",
)

CAMERA_SPECS: Mapping[str, CameraSpec] = {
    "central_cam": CENTRAL_CAM,
    "left_oppo_cam": LEFT_OPPO_CAM,
    "central_wide_cam": CENTRAL_WIDE_CAM,
    "cen_arm_right_wide_cam": CEN_ARM_RIGHT_WIDE_CAM,
    "cen_arm_near_side_cam": CEN_ARM_NEAR_SIDE_CAM,
    "cen_arm_side_cam": CEN_ARM_SIDE_CAM,
    "cen_arm_front_cam": CEN_ARM_FRONT_CAM,
    "cen_arm_top_cam": CEN_ARM_TOP_CAM,
}


class SceneInitRejectError(RuntimeError):
    """Raised when a sampled actor layout is infeasible for the left arm."""


class RightArmMotionError(RuntimeError):
    """Raised when the idle right arm deviates from its captured home qpos."""


class LeftArmTaskError(ValueError):
    """Raised for any contract violation inside a left-arm-only task."""


@dataclass(frozen=True)
class XYYawRange:
    x: tuple[float, float]
    y: tuple[float, float]
    yaw_rad: tuple[float, float]


@dataclass(frozen=True)
class ActorSamplingSpec:
    name: str
    workspace: XYYawRange
    minimum_clearance_m: float
    static: bool
    base_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class TaskManifest:
    semantic_task: str
    actor_specs: Mapping[str, ActorSamplingSpec]
    route_bin_count: int
    pilot_seed_count: int
    min_success_rate: float
    min_crossing_pair_rate: float


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, sapien.Pose):
        return {"p": value.p.tolist(), "q": value.q.tolist()}
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


class LeftTaskBase(Base_Task):
    """Base class implementing the left-arm safety and audit contract.

    Semantic task classes inherit this class and provide ``sample_layout`` /
    ``load_actors`` / ``play_once`` / ``check_success``.  Each wrapper module
    sets ``camera_variant`` to the exact named head-camera configuration it
    exports.
    """

    manifest: TaskManifest
    camera_variant: CameraVariant

    # ------------------------------------------------------------------ setup

    def setup_demo(self, **kwargs: Any) -> None:
        """Initialize the left-arm task environment.

        @input: the existing RoboTwin setup kwargs, including
            ``left_embodiment_config``, ``right_embodiment_config``, ``seed``,
            ``task_name`` and ``domain_randomization``.
        @output: None; stores the arm side, configures the camera, initializes
            the scene and captures the right home qpos after homing.
        @scenario: Enforce left-arm-only and fixed-right-arm behavior for every
            derived task variant.
        """
        task_kwargs = self.configure_camera(kwargs)
        self.camera_variant = task_kwargs.get("camera_variant", self.camera_variant)
        self.arm_side = "left"
        self.current_seed = kwargs.get("seed")
        self.attempt = int(getattr(self, "attempt", 0)) + 1
        self.right_home_qpos = None
        self.right_home_real_qpos = None
        self.right_max_abs_deviation = 0.0
        self.route_bins: list[int] = []
        self.sampled_layout_summary: dict[str, list[float]] = {}
        self.expert_task_success = False
        self.audit_path = str(kwargs.get("audit_path") or "")
        super()._init_task_env_(**task_kwargs)
        if self.camera_variant in CENTERED_BATCH_VARIANTS:
            self.move_right_arm_out_of_view()
        self.right_home_qpos = self.capture_right_home_qpos()
        self.right_home_real_qpos = np.asarray(
            self.robot.get_right_arm_real_jointState(), dtype=np.float32
        )
        if not self.audit_path:
            self.audit_path = os.path.join(self.save_dir, "audit.jsonl")

    def configure_camera(self, kwargs: Mapping[str, Any]) -> dict[str, Any]:
        """Return a deep-copied kwargs map for the selected camera variant.

        @input: the original setup kwargs mapping (never mutated); the
            wrapper-selected ``CameraVariant`` is read from ``self.camera_variant``.
        @output: a deep copy that embeds the exact variant name under the
            ``camera_variant`` key; ``central_cam`` leaves all camera fields
            unchanged, ``left_oppo_cam`` overwrites only ``head_camera`` pose,
            ``central_wide_cam`` keeps the central pose but selects the wide
            ``D435-W`` head camera type.
        @scenario: Keep camera as the single intended variation within a family
            and never infer the variant from the task-name suffix.
        """
        if not isinstance(kwargs, Mapping):
            raise LeftArmTaskError("setup_demo kwargs must be a mapping")
        if "left_embodiment_config" not in kwargs:
            raise LeftArmTaskError("missing left_embodiment_config in setup kwargs")
        if self.camera_variant not in CAMERA_SPECS:
            raise LeftArmTaskError(f"unknown camera_variant {self.camera_variant!r}")
        task_kwargs = deepcopy(dict(kwargs))
        # Embed the exact named variant so collection audit, config generation
        # and dataset metadata cannot infer it from the task-name suffix.
        task_kwargs["camera_variant"] = self.camera_variant

        camera_cfg = task_kwargs.get("camera")
        if isinstance(camera_cfg, Mapping):
            if not bool(camera_cfg.get("collect_head_camera", True)):
                raise LeftArmTaskError(
                    "left-arm task requires an enabled head camera; "
                    "camera.collect_head_camera is False"
                )
            if bool(camera_cfg.get("collect_wrist_camera", False)):
                raise LeftArmTaskError(
                    "left-arm task requires all wrist cameras disabled; "
                    "camera.collect_wrist_camera is True"
                )

        embodiment = task_kwargs["left_embodiment_config"]
        if self.camera_variant in CENTERED_BATCH_VARIANTS:
            embodiment_types = task_kwargs.get("embodiment")
            if (not isinstance(embodiment_types, list) or len(embodiment_types) != 3
                    or not all(isinstance(item, str) for item in embodiment_types[:2])):
                raise LeftArmTaskError(
                    "centred-arm batch requires a two-Piper embodiment specification"
                )
            # Robot applies -embodiment_dis / 2 to the left mount. Compensate
            # that entire offset so the active left Piper base remains at x=0;
            # the right articulation is moved out of view after Curobo setup.
            robot_pose = embodiment.get("robot_pose")
            if not isinstance(robot_pose, list) or not robot_pose or len(robot_pose[0]) != 7:
                raise LeftArmTaskError("left_embodiment_config has no valid robot_pose")
            robot_pose[0][0] = float(robot_pose[0][0]) + CENTERED_BATCH_ARM_SEPARATION_M / 2.0
            from .left_task_manifests import centered_wide_workspace_manifest
            self.manifest = centered_wide_workspace_manifest(self.manifest)
        camera_list = embodiment.get("static_camera_list")
        if not isinstance(camera_list, list):
            raise LeftArmTaskError("left_embodiment_config has no static_camera_list")
        head_cameras = [c for c in camera_list if c.get("name") == "head_camera"]
        if len(head_cameras) != 1:
            raise ValueError(
                "Piper configuration must have exactly one enabled head_camera; "
                f"found {len(head_cameras)}"
            )
        spec = CAMERA_SPECS[self.camera_variant]
        if spec.position is not None:
            head_cameras[0].update(
                {
                    "position": list(spec.position),
                    "forward": list(spec.forward),
                    "left": list(spec.left),
                }
            )
        if spec.head_camera_type is not None:
            if not isinstance(camera_cfg, Mapping):
                raise LeftArmTaskError(
                    f"camera variant {self.camera_variant!r} requires a camera "
                    "config to select its head camera type"
                )
            camera_cfg["head_camera_type"] = spec.head_camera_type
        return task_kwargs

    def move_right_arm_out_of_view(self) -> None:
        """Move only the idle right scene articulation outside the camera frame."""
        right_pose = self.robot.right_entity.get_root_pose()
        right_position = np.asarray(right_pose.p, dtype=np.float32).copy()
        right_position[0] = RIGHT_ARM_HIDDEN_WORLD_X_M
        self.robot.right_entity.set_root_pose(sapien.Pose(right_position, right_pose.q))
        applied_pose = self.robot.right_entity.get_root_pose()
        applied_x = float(np.asarray(applied_pose.p, dtype=np.float32)[0])
        if not np.isclose(applied_x, RIGHT_ARM_HIDDEN_WORLD_X_M, atol=1e-4):
            raise LeftArmTaskError(
                "failed to move the idle right arm outside the centered-camera frame: "
                f"expected x={RIGHT_ARM_HIDDEN_WORLD_X_M}, got x={applied_x}"
            )

    # ---------------------------------------------------------------- layout

    def sample_layout(self) -> dict[str, sapien.Pose]:
        """Sample one deterministic actor layout inside the left workspace.

        @input: None; uses only the seeded ``numpy.random`` stream.
        @output: mapping from stable actor names to ``sapien.Pose``.
        @scenario: Rejection-sampling helper for the semantic ``load_actors``.
        """
        raise NotImplementedError("sample_layout must be implemented by the semantic task")

    def validate_layout(self, layout: Mapping[str, sapien.Pose]) -> bool:
        """Return whether a sampled layout is legal for robot control.

        @input: the sampled name-to-pose layout.
        @output: boolean legality; pure with regard to robot control.
        @scenario: Allow a semantic task to reject a layout before acting.
        """
        return True

    def sample_spec_pose(self, spec: ActorSamplingSpec) -> sapien.Pose:
        """Sample one pose from an actor sampling spec.

        @input: an immutable actor sampling spec.
        @output: a ``sapien.Pose`` drawn from the spec workspace.
        @scenario: Reuse the manifest-defined ranges in every semantic sampler.
        """
        w = spec.workspace
        lock_yaw_for_stable_static = (
            self.camera_variant in CENTERED_BATCH_VARIANTS and spec.static
        )
        return rand_pose(
            xlim=[float(w.x[0]), float(w.x[1])],
            ylim=[float(w.y[0]), float(w.y[1])],
            qpos=[float(v) for v in spec.base_quat],
            rotate_rand=not lock_yaw_for_stable_static,
            rotate_lim=[0.0, 0.0, float(w.yaw_rad[1])],
        )

    def record_layout(self, layout: Mapping[str, sapien.Pose]) -> None:
        """Store a serializable layout summary and route bins for auditing."""
        summary: dict[str, list[float]] = {}
        for name, pose in layout.items():
            summary[str(name)] = [float(v) for v in pose.p] + [float(v) for v in pose.q]
        self.sampled_layout_summary = summary

    # ------------------------------------------------------------------ moves

    def move(self, actions_by_arm1: tuple, actions_by_arm2: tuple | None = None,
             save_freq: int | None = -1) -> Any:
        """Reject any action that would invoke the inactive right arm.

        @input: one ``(ArmTag("left"), action-list)`` pair and ``None`` as the
            second argument; a right tag, an ``.opposite`` tag or a parallel
            action group is rejected before delegating to ``Base_Task.move``.
        @output: whatever ``Base_Task.move`` returns.
        @scenario: Enforce the left-arm-only expert contract.
        """
        if actions_by_arm1 is not None and actions_by_arm2 is not None:
            raise LeftArmTaskError(
                f"{self.task_name} is left-arm-only; parallel action groups are forbidden"
            )
        for group in (actions_by_arm1, actions_by_arm2):
            if group is None:
                continue
            if not isinstance(group, (tuple, list)) or len(group) != 2:
                raise LeftArmTaskError(
                    "left-arm task expects a single (ArmTag, action-list) pair per arm"
                )
            arm_tag = group[0]
            if isinstance(arm_tag, str):
                arm_tag = ArmTag(arm_tag)
            if not isinstance(arm_tag, ArmTag):
                raise LeftArmTaskError(f"invalid arm tag {group[0]!r}")
            if arm_tag.arm != "left":
                raise LeftArmTaskError(
                    f"{self.task_name} is left-arm-only; received arm tag {arm_tag!r}"
                )
        return super().move(actions_by_arm1, actions_by_arm2, save_freq=save_freq)

    def move_by_displacement(
        self,
        arm_tag: ArmTag,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        quat: list | None = None,
        move_axis: Literal["world", "arm"] = "world",
    ) -> tuple:
        """Displace the left arm, falling back to the other axis on failure.

        @input: the arm tag, the world x/y/z displacement, an optional quat and
            the axis convention ("world" or "arm", as in ``Base_Task``).
        @output: the (ArmTag, action-list) pair that ``Base_Task`` would return.
        @scenario: A displacement can be full-SE(3) infeasible right after a
            grasp or placement, because the executed pose (position-constrained
            planning) may sit at the edge of the reachable workspace.  The two
            axis conventions give different retreat directions; when the
            requested axis fails and the move has a z component the method
            retries with the other axis.  The failed planner output is dropped
            from ``left_joint_path`` so replay parity is preserved.
        """
        before = self.plan_success
        path_len = len(self.left_joint_path) if hasattr(self, "left_joint_path") else 0
        result = super().move_by_displacement(arm_tag, x, y, z, quat, move_axis)
        self.move(result)
        other = "arm" if move_axis == "world" else "world"
        # Replay-only safety: during collection the recorded left_joint_path is
        # authoritative and each move consumes exactly one entry.  When the
        # recorded entry failed (a best-effort retreat that both axes could not
        # plan), firing the cross-axis retry here would consume a SECOND entry
        # belonging to a later move and drift left_cnt past the path end.
        # The retry only makes sense while live-planning (need_plan=True).
        if self.need_plan and z != 0.0 and not self.plan_success and before:
            if hasattr(self, "left_joint_path") and path_len < len(self.left_joint_path):
                del self.left_joint_path[path_len:]
            self.plan_success = True
            retry = super().move_by_displacement(arm_tag, x, y, z, quat, other)
            self.move(retry)
        return self.plan_success

    # ------------------------------------------------------------- right arm

    def capture_right_home_qpos(self) -> np.ndarray:
        """Capture the right home qpos in native drive-target space.

        @input: None; the right arm is home after ``_init_task_env_``.
        @output: np.ndarray of shape (7,), float32.
        @scenario: Provide the projection home values for policy actions.
        """
        return np.asarray(self.robot.get_right_arm_jointState(), dtype=np.float32).copy()

    def project_right_home(self, action: np.ndarray) -> np.ndarray:
        """Return a copied qpos vector whose right suffix is replaced by home.

        @input: action np.ndarray of native width (one-dimensional, 14 values).
        @output: np.ndarray float32 copy with ``[7:]`` set to the captured
            right home qpos.
        @scenario: Freeze the right arm before every simulator step without
            mutating the caller's buffer.
        """
        arr = np.asarray(action)
        if arr.ndim != 1:
            raise LeftArmTaskError(
                "project_right_home expects a one-dimensional native-width qpos vector"
            )
        if arr.shape[0] != NATIVE_QPOS_DIM:
            raise LeftArmTaskError(
                f"native qpos width must be {NATIVE_QPOS_DIM}, got {arr.shape[0]}"
            )
        if self.right_home_qpos is None:
            raise LeftArmTaskError("right home qpos has not been captured yet")
        projected = arr.astype(np.float32).copy()
        projected[LEFT_QPOS_DIM:] = self.right_home_qpos
        return projected

    def take_action(self, action: np.ndarray, action_type: Literal["qpos", "ee"] = "qpos") -> Any:
        """Reject ``ee`` control, project the right arm, then delegate.

        @input: a native-width policy action and its control mode.
        @output: whatever ``Base_Task.take_action`` returns.
        @scenario: Keep the right arm frozen for policy-driven episodes.
        """
        if action_type != "qpos":
            raise LeftArmTaskError(
                f"{self.task_name} rejects {action_type!r} policy control; only qpos is permitted"
            )
        projected = self.project_right_home(action)
        super().take_action(projected, action_type="qpos")
        self.assert_right_arm_stationary()

    def assert_right_arm_stationary(self) -> None:
        """Raise ``RightArmMotionError`` when the right arm deviates from home.

        @input: None; compares the real right joint state to the captured home.
        @output: None; raises when the maximum absolute deviation exceeds the
            contract tolerance.
        @scenario: Convert any idle-arm motion into an episode invalidator.
        """
        if self.right_home_real_qpos is None:
            return
        actual = np.asarray(self.robot.get_right_arm_real_jointState(), dtype=np.float32)
        deviation = float(np.max(np.abs(actual - self.right_home_real_qpos)))
        self.right_max_abs_deviation = max(self.right_max_abs_deviation, deviation)
        if deviation > RIGHT_ARM_TOLERANCE:
            raise RightArmMotionError(
                f"right arm deviated {deviation:.3e} from its home qpos "
                f"(tolerance {RIGHT_ARM_TOLERANCE:.1e})"
            )

    def right_arm_stationary(self) -> bool:
        """Return True when the right arm stayed at its home qpos.

        @input: None.
        @output: boolean immobility result.
        @scenario: Allow ``check_success`` to include the right-stationary test.
        """
        try:
            self.assert_right_arm_stationary()
            return True
        except RightArmMotionError:
            return False

    # ----------------------------------------------------------- expert moves

    def left_pick_place(self, actor: Actor, target_pose: Sequence[float], *,
                        functional_point_id: int = 0, pre_grasp_dis: float = 0.07,
                        lift_z: float = 0.08, pre_dis: float = 0.08, dis: float = 0.02,
                        constrain: str = "free") -> bool:
        """Pick one actor with the left arm and place it at a target pose.

        @input: actor and a seven-value target pose; optional grasp/place
            tuning parameters.
        @output: boolean plan success.
        @scenario: Shared left grasp, vertical clearance, target placement,
            release and vertical retreat routine using guarded moves.
        """
        if not self.plan_success:
            return False
        arm_tag = ArmTag("left")
        self.move(self.grasp_actor(actor, arm_tag=arm_tag, pre_grasp_dis=pre_grasp_dis))
        if not self.plan_success:
            return False
        self.move_by_displacement(arm_tag=arm_tag, z=lift_z)
        if not self.plan_success:
            return False
        self.move(
            self.place_actor(actor, arm_tag=arm_tag, target_pose=target_pose,
                             functional_point_id=functional_point_id, pre_dis=pre_dis,
                             dis=dis, constrain=constrain)
        )
        if not self.plan_success:
            return False
        self.move(self.open_gripper(arm_tag))
        self.move_by_displacement(arm_tag=arm_tag, z=lift_z, move_axis="world")
        return self.plan_success

    def left_return_home(self) -> bool:
        """Return the left arm to its original pose.

        @input: None.
        @output: boolean plan status.
        @scenario: Release-side cleanup after an expert placement.
        """
        if self.plan_success:
            self.move(self.back_to_origin(ArmTag("left")))
        return self.plan_success

    def left_retreat(self, arm_tag: ArmTag, z: float = 0.12) -> bool:
        """Lift the left arm with a position-only constrained world-frame move.

        @input: the arm tag and the world-frame upward displacement.
        @output: boolean plan success.
        @scenario: After releasing an object the exit pose does not need a
            fixed orientation; a position-only constrained lift lets the
            orientation relax to a reachable one, so the retreat plans even
            when the full-SE(3) displacement is infeasible at the workspace
            edge of a placement.
        """
        if not self.plan_success:
            return False
        origin = np.asarray(self.robot.get_left_ee_pose(), dtype=np.float64)
        target = origin.copy()
        target[:3] += np.asarray([0.0, 0.0, z], dtype=np.float64)
        self.move(
            (arm_tag, [Action(arm_tag, "move", target_pose=target.tolist(),
                              constraint_pose=[1.0, 1.0, 1.0, 0.0, 0.0, 0.0])])
        )
        return self.plan_success

    # ------------------------------------------------------------------ audit

    def write_episode_audit(self, status: str, **details: Any) -> None:
        """Append and flush one JSONL audit record.

        @input: status string (``accepted``, ``scene_rejected``,
            ``expert_failed``, ``right_arm_violation`` or
            ``invalid_observation``) plus optional extra detail fields.
        @output: None; writes one line to the audit JSONL path.
        @scenario: Produce a complete, serializable audit trail per attempt.
        """
        try:
            expert_success = bool(self.plan_success and self.check_success())
        except Exception:  # noqa: BLE001 - check_success may not be evaluable mid-rejection
            expert_success = False
        record: dict[str, Any] = {
            "task": self.task_name,
            "camera_variant": self.camera_variant,
            "seed": self.current_seed,
            "attempt": self.attempt,
            "accepted_episode": self.ep_num,
            "sampled_layout": self.sampled_layout_summary,
            "route_bins": self.route_bins,
            "expert_plan_success": bool(self.plan_success),
            "expert_task_success": expert_success,
            "right_home_qpos": (
                [float(v) for v in self.right_home_qpos]
                if self.right_home_qpos is not None else None
            ),
            "right_max_abs_deviation": float(self.right_max_abs_deviation),
            "status": status,
        }
        record.update(details)
        audit_path = self.audit_path or os.path.join(self.save_dir, "audit.jsonl")
        parent = os.path.dirname(os.path.abspath(audit_path))
        os.makedirs(parent, exist_ok=True)
        with open(audit_path, "a", encoding="ascii") as handle:
            handle.write(json.dumps(record, default=_json_default, ensure_ascii=True) + "\n")
            handle.flush()

    # ------------------------------------------------------------- route bins

    def classify_routes(self, routes: Sequence[tuple[str, Sequence[float], Sequence[float]]],
                        bin_count: int | None = None) -> list[int]:
        """Classify pickup-to-target routes into angular direction bins.

        @input: list of ``(name, pickup, target)`` triples and an optional bin
            count (defaults to ``manifest.route_bin_count``).
        @output: list of integer bin indices aligned with ``routes``.
        @scenario: Record route diversity for the manifest crossing gate.
        """
        count = bin_count or getattr(self.manifest, "route_bin_count", 4) or 4
        bins: list[int] = []
        for _name, pickup, target in routes:
            dx = float(target[0]) - float(pickup[0])
            dy = float(target[1]) - float(pickup[1])
            angle = float(np.arctan2(dy, dx)) + float(np.pi)
            bins.append(int((angle / (2.0 * np.pi)) * count) % count)
        return bins
