"""LEFT_TASK_DESIGN:
Source task: place_can_basket
Single-arm semantic change: source randomly selects an active arm, places the
    can into the basket, then has the opposite arm grasp and lift the basket;
    the derived task fixes the basket static in the left workspace, drops the
    opposite-arm grasp entirely and narrows the success predicate to a basket
    that stays within +/- 1 cm of its spawn height.
Left workspace manifest: place_can_basket, version 2 (2026-08-13)
Actors and clearance: basket (static, swept slightly wider to x in
    [-0.38,-0.14], y in [-0.30,-0.15], zero yaw, min 0.15 m), can (dynamic,
    x in [-0.25,-0.20], y in [0,0.08], zero yaw, min 0.15 m).  The can band is
    pinned by the lying-can downward-grasp IK reach (y capped at 0.08) and the
    basket yaw is pinned at zero so the calibrated plate offset (+0.05, -0.04)
    stays valid; the widened basket sweep is the safe randomization lever.
Expert sequence: left grasp can, pick the nearer basket functional point,
    place, release, retreat along the arm axis, return home.
Success predicate: preserve basket uprightness, can-in-basket distance and
    can/basket contact plus no-can-table-contact; narrow the basket-height test
    to +/- 1 cm; require left open and right home.  The opposite-arm basket
    grasp and the fallback placement branch are deleted.
Instruction change: {A}=can, {B}=basket, {a}=left; wording says the left arm
    puts the can into the stationary basket and must not claim the basket
    moves.
Pilot evidence: central-cam 30-seed pilot on version 0 0.27 (8/30 success),
    manifest hash c37d5602554d546f; version 2 (widened basket sweep) 50-seed
    pilot 0.28 (14/50 success), manifest hash ad0055fc4f784083
"""

from __future__ import annotations

import numpy as np
import sapien

from .left_task_base import LeftTaskBase, SceneRejectedError
from .left_task_manifests import get_manifest
from ..utils import *  # noqa: F401,F403

# ---------------------------------------------------------------------------
# LEGACY_RANDOMIZATION_PARAM
# The randomization protocol used before the current version 2 design.
# Intentionally unused: kept as a declared header constant so the previous
# geometry is reproducible and comparable.
#   Version 0 (original source-task geometry, manifest c37d5602554d546f):
#     - basket: static x in [-0.36, -0.16], y in [-0.28, -0.15], yaw 0,
#               clearance 0.15 m
#     - can:    x in [-0.25, -0.20], y in [0, 0.08], yaw 0, clearance 0.15 m
LEGACY_RANDOMIZATION_PARAM: dict[str, object] = {
    "basket_x": (-0.36, -0.16),
    "basket_y": (-0.28, -0.15),
    "basket_yaw_rad": (0.0, 0.00),
    "basket_clearance_m": 0.15,
    "can_x": (-0.25, -0.20),
    "can_y": (0.0, 0.08),
    "can_yaw_rad": (0.0, 0.00),
    "can_clearance_m": 0.15,
    "manifest_hash": "c37d5602554d546f",
}


def _rotate_quat(q: list | np.ndarray) -> list:
    """Return the input quaternion rotated by 45 deg about the world x axis.

    @input: a (w, x, y, z) quaternion, conventionally normalised (robot EE
        poses always are).
    @output: the Hamilton product ``qrot * q`` where ``qrot`` is a 45 deg
        rotation about the world x axis, as a Python list.
    @scenario: The downward grasp sits at a curobo workspace singularity: a
        pure vertical lift of the grasped can fails just above the can, but any
        wrist tilt about a world axis makes the same lift reachable.  The
        scipy ``Rotation.from_rotvec(45 deg, x) * q`` composition reduces to
        this left-multiply on a normalised quaternion, kept dependency-free.
    """
    half = float(np.deg2rad(45.0) / 2.0)
    c = float(np.cos(half))
    s = float(np.sin(half))
    qw, qx, qy, qz = c, s, 0.0, 0.0
    rw, rx, ry, rz = (float(v) for v in q)
    return [
        qw * rw - qx * rx - qy * ry - qz * rz,
        qw * rx + qx * rw + qy * rz - qz * ry,
        qw * ry - qx * rz + qy * rw + qz * rx,
        qw * rz + qx * ry - qy * rx + qz * rw,
    ]


def _actor_aabb(actor: Actor) -> tuple[float, float]:
    """Return the lowest and highest world-frame collision vertex z of an actor.

    @input: any RoboTwin ``Actor`` wrapper.
    @output: ``(min_z, max_z)`` over every collision shape.
    @scenario: The VHACD convex hull of the basket is taller than the visual
        mesh, so the interior floor and rim heights are read from the real
        collision geometry.
    """
    entity = actor.actor
    static = entity.find_component_by_type(sapien.physx.PhysxRigidStaticComponent)
    dynamic = entity.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
    comp = static if static is not None else dynamic
    if comp is None:
        return float(actor.get_pose().p[2]), float(actor.get_pose().p[2])
    world = actor.get_pose().to_transformation_matrix()
    lo, hi = float("inf"), -float("inf")
    for shape in comp.collision_shapes:
        local = shape.get_local_pose().to_transformation_matrix()
        verts = np.asarray(shape.get_vertices(), dtype=np.float64)
        verts = verts * np.asarray(shape.get_scale(), dtype=np.float64)
        verts_world = verts @ local[:3, :3].T + local[:3, 3]
        verts_world = verts_world @ world[:3, :3].T + world[:3, 3]
        lo = min(lo, float(verts_world[:, 2].min()))
        hi = max(hi, float(verts_world[:, 2].max()))
    return lo, hi


class PlaceCanBasketLeftImpl(LeftTaskBase):
    """Left-arm-only can-into-basket task with a stationary basket.

    The basket is static; the left arm transports the can to whichever basket
    functional point is nearer the current arm pose.  The success predicate
    verifies the basket never left its spawn height.
    """

    def __init__(self) -> None:
        super().__init__()
        self.manifest = get_manifest("place_can_basket")

    def sample_layout(self) -> dict[str, object]:
        basket_spec = self.manifest.actor_specs["basket"]
        can_spec = self.manifest.actor_specs["can"]
        clearance = max(basket_spec.minimum_clearance_m, can_spec.minimum_clearance_m)
        for _ in range(256):
            basket = self.sample_spec_pose(basket_spec)
            can = self.sample_spec_pose(can_spec)
            if float(np.linalg.norm(can.p[:2] - basket.p[:2])) >= clearance:
                return {"basket": basket, "can": can}
        raise SceneRejectedError(
            "could not sample a feasible place_can_basket layout in 256 attempts"
        )

    def load_actors(self) -> None:
        layout = self.sample_layout()
        if not self.validate_layout(layout):
            raise SceneRejectedError("place_can_basket layout rejected by validate_layout")
        self.basket_name = "110_basket"
        self.basket_id = int(np.random.choice([0, 1]))
        basket_pose = layout["basket"]
        # The basket sits at its natural z (resting on the table).  Raising it
        # pushed the collision rim to z~0.989, above what the held can can
        # clear: the left end effector peaks near z=1.10, the can hangs ~0.12 m
        # below it and the (lying) can is only ~0.026 m tall, so the can base
        # can clear at most z~0.955.  With the basket unraised the rim sits at
        # z~0.914 and the hover target z~1.085 keeps the can base above it.
        self.basket = create_actor(
            scene=self, pose=basket_pose,
            modelname=self.basket_name, model_id=self.basket_id, convex=True,
            is_static=True,
        )
        # Measure the real collision floor and rim: the VHACD convex hull
        # bulges beyond the visual extents, so the interior floor and rim
        # heights are read from the actual collision geometry rather than a
        # fixed half-height.
        self.basket_aabb_lo, self.basket_aabb_hi = _actor_aabb(self.basket)
        self.basket_rim_top = self.basket_aabb_hi
        self.can_name = "071_can"
        # Models 0, 1 and 5 are excluded: their VHACD hulls are not stable
        # resting shapes, so the lying can rolls (13-52 cm for model 1, and up
        # to a 2.9 deg drift for model 5) during the settle phase and trips
        # check_stable (UnStableError).  Models 2, 3 and 6 rest level with
        # under 0.3 cm displacement and hold a grasp.
        self.can_id = int(np.random.choice([2, 3, 6]))
        # The lying can's origin sits at its centre, so rand_pose's z (the
        # table plane, 0.741) sinks the bottom hull through the table and the
        # settle phase pops/tumbles it (UnStableError or a rolling spawn the
        # grasp then misses).  Rest the can's lowest collision vertex exactly
        # on the table plane so it settles onto its side with no impact.
        can_pose = layout["can"]
        self.can = create_actor(
            scene=self, pose=can_pose, modelname=self.can_name,
            model_id=self.can_id, convex=True,
        )
        can_lo, _can_hi = _actor_aabb(self.can)
        if can_lo < 0.741 - 0.001:
            can_p = list(self.can.get_pose().p)
            can_p[2] += 0.741 - can_lo
            self.can.actor.set_pose(
                sapien.Pose(can_p, self.can.get_pose().q)
            )
        self.start_height = float(self.basket.get_pose().p[2])
        self.object_start_height = float(self.can.get_pose().p[2])
        self.basket.set_mass(0.5)
        self.can.set_mass(0.01)
        self.add_prohibit_area(self.can, padding=0.1)
        self.add_prohibit_area(self.basket, padding=0.05)
        self.record_layout(layout)
        basket_point = self.basket.get_functional_point(0)[:3]
        self.route_bins = self.classify_routes(
            [("can", np.asarray(layout["can"].p), np.asarray(basket_point))]
        )

    def select_basket_point(self) -> np.ndarray:
        """Return the nearer basket functional point for the current left pose.

        @input: None; reads the live left arm pose and both basket functional
            points.
        @output: a seven-value target pose copy with a left-facing quaternion.
        @scenario: Mirror the source nearest-point selection without any
            opposite-arm branch.
        """
        arm_pose = self.get_arm_pose(arm_tag=ArmTag("left"))
        f0 = np.array(self.basket.get_functional_point(0))
        f1 = np.array(self.basket.get_functional_point(1))
        if np.linalg.norm(f0[:2] - arm_pose[:2]) < np.linalg.norm(f1[:2] - arm_pose[:2]):
            place_pose = f0
        else:
            place_pose = f1
        place_pose = place_pose.copy()
        place_pose[3:] = (-1, 0, 0, 0)
        return place_pose

    def place_can_left(self) -> bool:
        """Left grasp, tilted carry, plate descend, gentle release, retreat."""
        if not self.plan_success:
            return False
        arm_tag = ArmTag("left")
        self.move(self.grasp_actor(self.can, arm_tag=arm_tag, pre_grasp_dis=0.05))
        if not self.plan_success:
            return False
        # The downward grasp sits at a curobo workspace singularity: a pure
        # vertical lift fails just above the can, but tilting the wrist 45 deg
        # about the world x axis makes the same lift reachable.  Tilt and lift
        # together, then keep the tilted grasp for the whole carry so the held
        # can clears the basket rim while the hand stays inside the reachable
        # ceiling (the can hangs ~0.10 m below the end effector).
        ee = np.asarray(self.get_arm_pose(arm_tag))
        target = np.array([ee[0], ee[1], ee[2] + 0.06, *_rotate_quat(ee[3:])])
        self.move(self.move_to_pose(arm_tag, target))
        if not self.plan_success:
            return False
        ee2 = np.asarray(self.get_arm_pose(arm_tag))
        offset = np.asarray(self.can.get_pose().p[:3]) - ee2[:3]
        # The basket floor is ring-shaped: the VHACD collision pieces form an
        # annulus of plates around an open central void, so a drop over the
        # basket centre falls through to the table.  Target the solid plate
        # cluster offset (+0.05, -0.04) from the basket origin (calibrated
        # against the fixed basket quat and yaw) instead.
        basket_p = np.asarray(self.basket.get_pose().p)
        plate_world = np.array([basket_p[0] + 0.05, basket_p[1] - 0.04])
        hover_ee = np.concatenate(
            [np.array([plate_world[0], plate_world[1], 0.95]) - offset, ee2[3:]]
        )
        self.move(self.move_to_pose(arm_tag, hover_ee))
        if not self.plan_success:
            return False
        # Re-measure the can-to-EE offset after the hover transport: the can
        # can slip in the fingers during the carry, and the descend target must
        # track the can, not the hand.  Descend until the can centre reaches
        # z=0.90, which seats the can bottom on the plate.
        ee3 = np.asarray(self.get_arm_pose(arm_tag))
        offset2 = np.asarray(self.can.get_pose().p[:3]) - ee3[:3]
        drop_ee = np.concatenate(
            [np.array([plate_world[0], plate_world[1], 0.90]) - offset2, ee3[3:]]
        )
        self.move(self.move_to_pose(arm_tag, drop_ee))
        if not self.plan_success:
            return False
        # Release gently: a fully open gripper (pos=1.0) sweeps the fingers
        # past the released can and flings it sideways, while pos=0.7 opens the
        # fingers just past the can diameter so it seats on the plate.  Each
        # gripper action spans 200 physics steps, so the second open(0.7) is a
        # 200-step hold that lets the can settle; a final open(1.0) then makes
        # the success predicate's is_left_gripper_open (val > 0.8) hold.
        self.move(self.open_gripper(arm_tag, pos=0.7))
        self.move(self.open_gripper(arm_tag, pos=0.7))
        self.move(self.open_gripper(arm_tag, pos=1.0))
        # Lift the hand clear of the resting can before returning home: the
        # direct home path sweeps through the basket airspace and drags the
        # settled can out onto the table.  The descend pose sits near the
        # reachable ceiling (EE z ~1.00 of a ~1.10 limit), so a fixed raise is
        # sometimes infeasible; retry with progressively smaller heights.  A
        # raise >= 0.04 lifts the finger pads above the can's top, after which
        # the planner routes the empty hand home clear of the basket.
        ee4 = np.asarray(self.get_arm_pose(arm_tag))
        for raise_h in (0.10, 0.06, 0.04):
            self.plan_success = True
            self.move(
                self.move_to_pose(
                    arm_tag, np.array([ee4[0], ee4[1], ee4[2] + raise_h, *ee4[3:]])
                )
            )
            if self.plan_success:
                break
        home_ok = self.left_return_home()
        # The final release is a pure gripper command and never needs IK
        # planning, but Base_Task.move short-circuits whenever plan_success is
        # False, so a failed home-return plan would leave the just-released can
        # unopened and fail the is_left_gripper_open predicate.  Force the open
        # through and restore the pre-release plan status for the caller.
        was_plan = bool(self.plan_success)
        self.plan_success = True
        self.move(self.open_gripper(arm_tag, pos=1.0))
        released = self.is_left_gripper_open()
        return home_ok and was_plan and released

    def play_once(self) -> dict:
        self.place_can_left()
        self.info["info"] = {
            "{A}": f"{self.can_name}/base{self.can_id}",
            "{B}": f"{self.basket_name}/base{self.basket_id}",
            "{a}": "left",
        }
        return self.info

    def check_success(self) -> bool:
        can_p = self.can.get_pose().p
        basket_p = self.basket.get_pose().p
        basket_axis = (
            self.basket.get_pose().to_transformation_matrix()[:3, :3]
            @ np.array([[0, 1, 0]]).T
        )
        can_contact_table = not self.check_actors_contact(self.can_name, "table")
        can_contact_basket = self.check_actors_contact(self.can_name, self.basket_name)
        # The static basket never moves, so the source L1 distance predicate
        # (|dx|+|dy|+|dz| < 0.15) is unsatisfiable: the can resting on the
        # basket floor sits ~0.14 m above the basket origin and consumes the
        # whole budget in z alone.  Measure the can against the stationary
        # basket horizontally (0.15 m, matching the source clearance) plus a
        # vertical band between the measured basket floor and rim, mirroring
        # the sibling place_cans_plasticbox horizontal-only pattern.
        can_in_basket_horiz = bool(np.linalg.norm(can_p[:2] - basket_p[:2]) < 0.15)
        can_in_basket_vert = bool(
            self.basket_aabb_lo <= can_p[2] < self.basket_rim_top
        )
        return (
            bool(abs(basket_p[2] - self.start_height) < 0.01)
            and bool(np.dot(basket_axis.reshape(3), [0, 0, 1]) > 0.5)
            and can_in_basket_horiz
            and can_in_basket_vert
            and can_contact_table
            and can_contact_basket
            and self.is_left_gripper_open()
            and self.right_arm_stationary()
        )
