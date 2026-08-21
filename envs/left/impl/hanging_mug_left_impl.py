"""LEFT_TASK_DESIGN:
Source task: hanging_mug
Single-arm semantic change: source hands the mug from the left arm to the
    right arm at ``middle_pos`` before hanging; the derived task deletes
    ``middle_pos`` and all handover calls and keeps the mug in the left arm.
Left workspace manifest: hanging_mug, version 0
Actors and clearance: mug (dynamic, min 0.10 m), rack (static, min 0.10 m)
Expert sequence: left grasp mug, free-place the mug base below the rack middle
    so the tilted pillar catches it, release, settle, retreat along the arm
    axis (a failed retreat is tolerated).  The rack is rolled about x by 45
    deg so its hook sits at fp0 z ~0.85-0.87, inside the left EEF ceiling.
Success predicate: preserve source rack/mug functional-point geometry and the
    height test; replace ``is_right_gripper_open()`` with left-open plus
    right-home immobility.
Instruction change: {A}=mug asset, {B}=rack asset, {a}=left; wording is
    left-arm-only hang.
Mug model: frozen on model 0 (scale 0.08); the other 039_mug models fall off
    the tilted pillar or make the placement unreachable.
Pilot evidence: central-cam 30-seed pilot 0.80 (24/30 success), manifest hash
    28ee798eb3dfcde4

DEPRECATED (2026-08-09): the task is frozen and no longer pursued.  The only
    reliable hanging geometry is the tilted rack (fp0 world z ~0.85-0.87).
    Under an upright, yaw-only rack (the only orientation allowed for future
    work) the 19 mm rack bar cannot support the 11 cm mug rim: every upright
    hang variant (rim-over-bar, drop-in, handle, custom bars) fails physically,
    and no 039_mug model offers a collision void on its handle.  The task is
    kept registered so manifests, wrappers, and tests stay stable, but every
    instantiation warns and the task must not be used for collection.  See
    docs/theory/viewpoint_experiment_summary.md.
"""

from __future__ import annotations

import warnings

import numpy as np

from ..left_task_base import LeftTaskBase, SceneInitRejectError
from ..left_task_manifests import get_manifest
from ...utils import *  # noqa: F401,F403

_DEPRECATION_WARNING = (
    "HangingMugLeftImpl is DEPRECATED / NotFullyImplemented: an upright "
    "yaw-only rack cannot physically hang the 039_mug (19 mm rack bar vs "
    "11 cm mug rim; all upright variants fail physically).  Do not use this "
    "task for collection or evaluation."
)


class HangingMugLeftImpl(LeftTaskBase):
    """Left-arm-only hang-mug semantic task.

    DEPRECATED: this task is frozen and kept only to stabilise manifests,
    wrappers, and tests.  Every instantiation emits a warning (see
    ``_DEPRECATION_WARNING``); the task must not be collected or evaluated.

    The left arm keeps the mug through a left-side safe transit pose and
    performs the rack functional-point approach itself.  The right arm stays
    at its captured home qpos for the whole episode.
    """

    def __init__(self) -> None:
        super().__init__()
        warnings.warn(_DEPRECATION_WARNING, RuntimeWarning, stacklevel=2)
        self.manifest = get_manifest("hanging_mug")
        self.transit_pos = [-0.30, -0.15, 0.75, 1, 0, 0, 0]

    def sample_layout(self) -> dict[str, object]:
        mug_spec = self.manifest.actor_specs["mug"]
        rack_spec = self.manifest.actor_specs["rack"]
        for _ in range(128):
            mug_pose = self.sample_spec_pose(mug_spec)
            rack_pose = self.sample_spec_pose(rack_spec)
            # The source mug spawns rotated about the x axis; that pose keeps
            # the left EEF able to lift after grasping.  The generic yaw
            # sampler would tilt the mug out of the reachable grasp frame.
            mug_pose.q = np.array([0.707, 0.707, 0.0, 0.0])
            if self._layout_feasible(mug_pose, rack_pose):
                return {"mug": mug_pose, "rack": rack_pose}
        raise SceneInitRejectError(
            "could not sample a feasible hanging_mug layout in 128 attempts"
        )

    def _layout_feasible(self, mug_pose, rack_pose) -> bool:
        clearance = max(
            self.manifest.actor_specs["mug"].minimum_clearance_m,
            self.manifest.actor_specs["rack"].minimum_clearance_m,
        )
        if float(np.linalg.norm(mug_pose.p[:2] - rack_pose.p[:2])) < clearance:
            return False
        # Iteration-4 pilot: the rack is pinned to a single left-reachable spot
        # (rolled 45 deg about x so the hook sits at fp0 z ~0.85-0.87); keep
        # the sampled spot inside that band.
        if not (-0.30 <= float(rack_pose.p[0]) <= -0.28):
            return False
        return True

    def load_actors(self) -> None:
        layout = self.sample_layout()
        if not self.validate_layout(layout):
            raise SceneInitRejectError("hanging_mug layout rejected by validate_layout")
        # Iteration-4 pilot: only model 0 (scale 0.08) is graspable and rides
        # the tilted pillar; the other mug models either fall off the rack or
        # make the placement unreachable, so the family freezes on model 0.
        self.mug_id = 0
        self.mug = create_actor(
            self, pose=layout["mug"], modelname="039_mug", convex=True,
            model_id=self.mug_id,
        )
        self.mug.set_mass(0.05)
        self.rack = create_actor(
            self, pose=layout["rack"], modelname="040_rack", is_static=True, convex=True,
        )
        self.add_prohibit_area(self.mug, padding=0.1)
        self.add_prohibit_area(self.rack, padding=0.1)
        self.record_layout(layout)
        rack_point = self.rack.get_functional_point(0)[:3]
        self.route_bins = self.classify_routes(
            [("mug", np.asarray(layout["mug"].p), np.asarray(rack_point))]
        )

    def hang_mug_left(self) -> bool:
        """Grasp and free-place the mug onto the tilted rack with the left arm."""
        if not self.plan_success:
            return False
        arm_tag = ArmTag("left")
        self.move(self.grasp_actor(self.mug, arm_tag=arm_tag, pre_grasp_dis=0.05))
        if not self.plan_success:
            return False
        # Iteration-4 pilot (B config): free-place the mug base just below the
        # rack middle so the mug drops a few cm and the tilted pillar catches
        # it; mug fp then rests near z=0.95, far above the 0.86 check.  The
        # EE target stays under the left EEF ceiling.
        rack_mid = (np.asarray(self.rack.get_pose().p)
                    + np.asarray(self.rack.get_functional_point(0))[:3]) / 2
        fp_off = (np.asarray(self.mug.get_functional_point(0))[:3]
                  - np.asarray(self.mug.get_pose().p))
        target_pose = [float(rack_mid[0] - fp_off[0]),
                       float(rack_mid[1] - fp_off[1]), 0.77]
        self.move(self.place_actor(self.mug, arm_tag=arm_tag,
                                   target_pose=target_pose,
                                   pre_dis=0.05, dis=0.0, constrain="free"))
        if not self.plan_success:
            return False
        self.move(self.open_gripper(arm_tag))
        for _ in range(120):
            self.scene.step()
        self.move_by_displacement(arm_tag=arm_tag, z=0.1, move_axis="arm")
        # Iteration-4 pilot: the post-release retreat lifts the left EEF along
        # the arm axis near the tilted rack and can sit at the workspace edge;
        # when neither axis can plan, tolerate it (the mug is already hanging
        # and the success predicate only reads the mug/rack geometry plus the
        # gripper and right-arm states).
        if not self.plan_success:
            self.plan_success = True
        return self.plan_success

    def play_once(self) -> dict:
        self.hang_mug_left()
        self.info["info"] = {
            "{A}": f"039_mug/base{self.mug_id}",
            "{B}": "040_rack/base0",
            "{a}": "left",
        }
        return self.info

    def check_success(self) -> bool:
        mug_function_pose = self.mug.get_functional_point(0)[:3]
        rack_pose = self.rack.get_pose().p
        rack_function_pose = self.rack.get_functional_point(0)[:3]
        rack_middle_pose = (rack_pose + rack_function_pose) / 2
        eps = 0.02
        return (
            bool(np.all(abs((mug_function_pose - rack_middle_pose)[:2]) < eps))
            and bool(mug_function_pose[2] > 0.86)
            and self.is_left_gripper_open()
            and self.right_arm_stationary()
        )
