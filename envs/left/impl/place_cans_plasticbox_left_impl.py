"""LEFT_TASK_DESIGN:
Source task: place_cans_plasticbox
Single-arm semantic change: source grasps can 1 with the left arm and can 2
    with the right arm concurrently and drops both into the plastic box; the
    derived task fixes the plastic box static in the left workspace and
    serially transfers can 1 to functional point one and can 2 to functional
    point zero with the left arm, returning home between transfers.
Left workspace manifest: place_cans_plasticbox, version 4 (2026-08-22)
Actors and clearance: plasticbox (static, centre band x in [-0.37,-0.15],
    self-symmetric about x=-0.26, y in [-0.28,-0.23], zero yaw), can1
    (dynamic, x in [-0.42,-0.36], y in [-0.15,-0.07], zero yaw) and can2
    (dynamic, x in [-0.16,-0.10], the exact mirror image of can1 about
    x=-0.26, y in [-0.15,-0.07], zero yaw).  The rejection sampler uses
    per-axis AABB footprint clearance (box half-extents (0.078, 0.097), can
    half-extent 0.029, 0.01 m margin) so a can centre never enters the
    062_plasticbox hull, including its corner region.
Expert sequence: left grasp can 1, lift z=0.08, diagonal transit (pre_dis
    0.05/0.03 ladder — 0.1 pushes the pre pose EE past the ~1.03 m ceiling),
    measured descend to the functional point, gentle release (open 0.7 x3
    then 1.0 — a gripped full open sweeps the can out of the box), descending
    raise retries, home; then the same for can 2 at the other functional
    point (floor-level seat; ~6 mm hull clearance to the seated can 1).
Success predicate: preserve both nearest-functional-point containment checks
    (0.04 m) and the left-open condition; replace the right-gripper condition
    with right-home immobility.
Instruction change: {A}=can 1, {B}=plastic box, {C}=can 2, {a}=left; wording
    says the left arm drops both cans into the stationary plastic box.
Pilot evidence: central-cam 30-seed pilot on version 0 0.40 (12/30 success),
    manifest hash 0ed2a154b3a88e40; version 2 (overlapped can bands) 50-seed
    pilot 0.20 (10/50 success), manifest hash 2f9693508acf794b; version 3
    (union plasticbox sweep) 50-seed pilot 0.28 (14/50 success) but only ~0.05
    realized in 200-episode collection (3479 seeds, ~20% UnStableError),
    manifest hash 4e5f2e964176bc6e; version 4 200-seed pilot on the
    cen_arm_near_side_cam wrapper 0.15 (30/200, see
    results/pilot/sym_v4), centered-batch override in
    left_task_manifests.CENTERED_FAMILY_OVERRIDES (box x in [-0.05,0.05],
    can1 x in [-0.20,-0.13], can2 x in [0.13,0.20], mirror-symmetric about
    x=0; mean(can1_x+can2_x)=0.009 over successes), manifest hash
    d13fd6141c9178fe; SQUARE iteration same day — both cans sample one
    y-axis-centred square (0,-0.11) x in [-0.08,0.08], y in [-0.18,-0.04],
    box square (0,-0.25) x in [-0.04,0.04], y in [-0.28,-0.22], all six can
    models restored, clearance margin tightened to 0.015 m — 50-seed pilot
    0.28 (14/50, results/pilot/square_v5; can x mean +0.003, box x mean
    -0.003 over successes); YAW iteration (2026-08-23, version 6) — the box
    centre sweeps x in [-0.10,0.10], y in [-0.28,-0.20] with yaw sampled in
    [-0.60,0.60] rad (impl _sample_box_pose bypasses the centred-batch static
    yaw lock), both cans sample one shared standing band x in [-0.18,0.18],
    y in [-0.24,-0.02] with fixed upright orientation, and each can's
    admissible region is derived from the sampled box pose/geometry with a
    0.05 m safety margin (impl _clear_can_box) so cans appear on any side of
    a randomly oriented box (pilot evidence: results/pilot/yaw_v6) — 200-seed
    pilot 0.47 (93/200; box yaw covers +/-34 deg, can bearings vs box span
    the full circle, min can-to-box-hull distance exactly 0.079 m =
    0.05 margin + can radius, min can-can 0.075 m), centered-batch override
    hash f0e7b96e311fefbb
"""

from __future__ import annotations

import numpy as np
import sapien

from ..left_task_base import LeftTaskBase, SceneInitRejectError
from ..left_task_manifests import get_manifest
from ...utils import *  # noqa: F401,F403

# ---------------------------------------------------------------------------
# LEGACY_RANDOMIZATION_PARAM
# The randomization protocol used before the current version 2 design.
# Intentionally unused: kept as a declared header constant so the previous
# geometry is reproducible and comparable.
#   Version 0 (original source-task geometry, manifest 0ed2a154b3a88e40):
#     - plasticbox: static x in [-0.36, -0.14], y in [-0.28, -0.23], yaw 0,
#                   clearance 0.08 m
#     - can1:       x in [-0.42, -0.36], y in [-0.15, -0.07], yaw 0,
#                   clearance 0.08 m
#     - can2:       x in [-0.16, -0.10], y in [-0.15, -0.07], yaw 0,
#                   clearance 0.08 m
LEGACY_RANDOMIZATION_PARAM: dict[str, object] = {
    "plasticbox_x": (-0.36, -0.14),
    "plasticbox_y": (-0.28, -0.23),
    "plasticbox_yaw_rad": (0.0, 0.00),
    "plasticbox_clearance_m": 0.08,
    "can1_x": (-0.42, -0.36),
    "can1_y": (-0.15, -0.07),
    "can2_x": (-0.16, -0.10),
    "can2_y": (-0.15, -0.07),
    "can_yaw_rad": (0.0, 0.00),
    "can_clearance_m": 0.08,
    "manifest_hash": "0ed2a154b3a88e40",
}

# World-frame collision-footprint half-extents (measured in sim 2026-08-22,
# worst case across model ids 3/5 for the box and 0-6 for the can) used by the
# per-axis footprint clearance in sample_layout.
_BOX_HALF = (0.078, 0.097)
_CAN_HALF = (0.029, 0.029)
# Standing-can half height (collision hull, models 2/3/6) and the table plane.
_CAN_HALF_Z = 0.048
_TABLE_Z = 0.741

# ---------------------------------------------------------------------------
# VERSION 4 CHANGES (2026-08-22) — fixing the ~5% collection rate
#
# Symptom: the cen_arm_near_side_cam 200-episode batch burned 3479 seeds for
# 184 episodes (~5% success).  Three independent causes, all fixed here:
#
# 1. GEOMETRY (cen_arm batch only — see CENTERED_FAMILY_OVERRIDES in
#    envs/left/left_task_manifests.py).  The centred-arm batch replaces every
#    family manifest with one shared envelope x in (-0.20,0.20),
#    y in (-0.15,0.15), which let the box and cans overlap or spawn behind
#    the shoulder.  place_cans_plasticbox now overrides it with an
#    x-mirror-symmetric flanking layout (box centre band x in [-0.05,0.05],
#    can1 x in [-0.20,-0.13], can2 x in [0.13,0.20], all in the front region
#    y in [-0.14,-0.04]); the per-family manifest here (left-arm frame,
#    centred on x=-0.26) got the same flanking structure.
#
# 2. SETTLE STABILITY (~20% of seeds died as UnStableError "071_can"):
#    a) 071_can MODELS 0, 1 AND 5 ARE DISABLED — their VHACD collision
#       hulls are not valid resting shapes, so the can tips or rolls during
#       the settle phase and check_stable rejects the seed.  Measured
#       evidence (place_can_basket pilots): model 1 rolls 13-52 cm, model 5
#       drifts up to 2.9 deg, model 0 also tips; models 2, 3 and 6 rest
#       level with < 0.3 cm displacement and hold a grasp.  The spawn pool
#       is therefore np.random.choice([2, 3, 6]) below — do NOT re-add
#       0/1/5 without re-proving settle stability.  (SQUARE ITERATION,
#       2026-08-22 later the same day: the [2,3,6] restriction was LIFTED —
#       all six models are back in the pool; see the comment at the spawn
#       site in load_actors for why _rest_on_table makes them viable.)
#    b) _rest_on_table: rand_pose spawns the can ORIGIN at the table plane,
#       burying the lower half of the hull; the settle pop then rotates the
#       can past the 3 deg threshold.  The can is lifted so its lowest
#       collision vertex rests exactly on the table.
#    c) sample_layout._clear: per-axis AABB footprint clearance replaces the
#       radial centre-distance test, which let a can sit inside the box
#       hull's corner region (radial distance under-approximates a
#       rectangular footprint at the diagonal).
#
# 3. EXPERT SEQUENCE (place_can_left).  The stock place_actor(pre_dis=0.1)
#    failed planning: its pre pose sits at EE z~1.03, on the left arm's
#    reachable ceiling.  Replaced by the place_can_basket recipe: a 0.05/0.03
#    pre ladder, a descend measured from the live can-to-EE offset, a gentle
#    0.7x3 + 1.0 release (a gripped full-open sweeps the can 3-6 cm out of
#    the box), and descending raise retries for the retreat.  Both cans use
#    the floor-level seat; the drop_in rim-release ladder remains available
#    as a parameter but is not exercised by play_once.
#
# Result: 200-seed pilot 0.15 (30/200, results/pilot/sym_v4); realized
# collection rate ~27% (vs ~5% before).
# ---------------------------------------------------------------------------


def _actor_aabb_z(actor) -> tuple[float, float]:
    """Return the lowest and highest world-frame collision vertex z of an actor.

    @input: any RoboTwin ``Actor`` wrapper.
    @output: ``(min_z, max_z)`` over every collision shape.
    @scenario: Read rim/floor heights from the real collision geometry (the
        VHACD hull differs from the visual mesh), as in place_can_basket.
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


def _rest_on_table(actor) -> None:
    """Lift an actor so its lowest collision vertex sits on the table plane.

    @input: a freshly created dynamic actor whose origin spawns at the table
        plane (rand_pose z), sinking its lower hull through the table.
    @output: None; the actor pose z is corrected in place.
    @scenario: The settle phase otherwise pops the buried hull upward and the
    orientation drift trips check_stable (UnStableError) — the fix proven by
    place_can_basket.
    """
    lo, _hi = _actor_aabb_z(actor)
    if lo < _TABLE_Z - 0.001:
        can_p = list(actor.get_pose().p)
        can_p[2] += _TABLE_Z - lo
        actor.actor.set_pose(sapien.Pose(can_p, actor.get_pose().q))


class PlaceCansPlasticboxLeftImpl(LeftTaskBase):
    """Left-arm-only two-cans-into-plastic-box task.

    The plastic box is static in the left workspace; the left arm serially
    transfers can 1 to functional point one and can 2 to functional point
    zero.
    """

    def __init__(self) -> None:
        super().__init__()
        self.manifest = get_manifest("place_cans_plasticbox")

    def sample_layout(self) -> dict[str, object]:
        specs = {name: self.manifest.actor_specs[name] for name in
                 ("plasticbox", "can1", "can2")}
        for _ in range(128):
            plasticbox, box_yaw = self._sample_box_pose(specs["plasticbox"])
            can1 = self.sample_spec_pose(specs["can1"])
            can2 = self.sample_spec_pose(specs["can2"])
            if (self._clear(can1, can2, _CAN_HALF, _CAN_HALF)
                    and self._clear_can_box(can1, plasticbox, box_yaw)
                    and self._clear_can_box(can2, plasticbox, box_yaw)):
                return {"plasticbox": plasticbox, "can1": can1, "can2": can2}
        raise SceneInitRejectError(
            "could not sample a feasible place_cans_plasticbox layout in 128 attempts"
        )

    def _sample_box_pose(self, spec):
        """Sample the plasticbox centre and yaw explicitly.

        @input: the plasticbox ``ActorSamplingSpec``.
        @output: ``(sapien.Pose, yaw_rad)``; the yaw is composed onto the spec
            base quaternion as a world-z rotation and returned so the can/box
            clearance test can use the exact sampled angle.
        @scenario: ``sample_spec_pose`` locks yaw for static actors in the
            centred batch (lock_yaw_for_stable_static), which froze the box
            orientation at zero yaw; this family's yaw iteration (2026-08-23)
            deliberately overrides that lock with the manifest yaw range.
        """
        w = spec.workspace
        centre = np.array([
            float(np.random.uniform(w.x[0], w.x[1])),
            float(np.random.uniform(w.y[0], w.y[1])),
        ], dtype=float)
        yaw = float(np.random.uniform(w.yaw_rad[0], w.yaw_rad[1]))
        half = yaw / 2.0
        qz = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])
        base = np.asarray(spec.base_quat, dtype=float)
        # qz (x,y,z,w order-agnostic quaternion mult in (w,x,y,z) scalars).
        w1, x1, y1, z1 = qz
        w2, x2, y2, z2 = base
        quat = np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ], dtype=float)
        quat /= float(np.linalg.norm(quat))
        return sapien.Pose([centre[0], centre[1], 0.741], quat.tolist()), yaw

    @staticmethod
    def _clear_can_box(can_pose, box_pose, box_yaw, margin: float = 0.05) -> bool:
        """Derived can/box clearance against the yaw-rotated box rectangle.

        @input: a can pose, the plasticbox pose and its sampled yaw; the
            margin defaults to the 0.05 m safety threshold requested for this
            family.
        @output: True when the can's circular footprint stays at least
            ``margin`` outside the box's collision rectangle in the box frame.
        @scenario: The admissible can region is derived from the generated box
            pose and geometry (world-z rotation of the 0.078 x 0.097
            half-extents) instead of a fixed band, so cans can appear on any
            side of a randomly oriented box without ever overlapping it.
        """
        delta = np.asarray(can_pose.p[:2], dtype=float) - np.asarray(box_pose.p[:2], dtype=float)
        c, s = np.cos(box_yaw), np.sin(box_yaw)
        local = np.array([c * delta[0] + s * delta[1],
                          -s * delta[0] + c * delta[1]], dtype=float)
        clamped = np.clip(local, -np.asarray(_BOX_HALF), np.asarray(_BOX_HALF))
        dist = float(np.linalg.norm(local - clamped))
        return dist >= margin + _CAN_HALF[0]

    @staticmethod
    def _clear(pose_a, pose_b, half_a, half_b) -> bool:
        """Per-axis AABB footprint clearance (version 4 geometry).

        Two actors are clear when their world-frame collision footprints stay
        apart on at least one axis; a radial centre-distance test lets a can
        sit inside the rectangular box hull's corner region, which the settle
        phase then ejects as UnStableError (~20% of collection failures under
        the version 3 manifest).  The 0.015 m margin (square iteration,
        2026-08-22) is deliberately slightly stricter than the 0.01 m first
        cut, as requested for the can/can and can/box spacing.
        """
        dx = abs(float(pose_a.p[0]) - float(pose_b.p[0]))
        dy = abs(float(pose_a.p[1]) - float(pose_b.p[1]))
        margin = 0.015
        return (dx >= half_a[0] + half_b[0] + margin
                or dy >= half_a[1] + half_b[1] + margin)

    def load_actors(self) -> None:
        layout = self.sample_layout()
        if not self.validate_layout(layout):
            raise SceneInitRejectError(
                "place_cans_plasticbox layout rejected by validate_layout"
            )
        self.plasticbox_id = int(np.random.choice([3, 5]))
        self.plasticbox = create_actor(
            scene=self, pose=layout["plasticbox"], modelname="062_plasticbox",
            convex=True, model_id=self.plasticbox_id, is_static=True,
        )
        self.plasticbox.set_mass(0.05)
        # Measure the real collision rim so the release heights below track the
        # collision world, not the visual mesh (as in place_can_basket).
        _box_lo, self.box_rim_top = _actor_aabb_z(self.plasticbox)
        # ALL 071_can models (0,1,2,3,5,6) are used again per the square
        # iteration (2026-08-22): _rest_on_table seats each hull exactly on
        # the table plane, which removes the buried-hull pop that made models
        # 0/1/5 look unspawnable; any residual per-model settle instability
        # now costs only that seed (rejection sampling), not the pool.  The
        # earlier [2,3,6]-only restriction (first 2026-08-22 cut) is kept in
        # the VERSION 4 CHANGES history above in case stability regresses.
        self.object1_id = int(np.random.choice([0, 1, 2, 3, 5, 6]))
        self.can1 = create_actor(
            scene=self, pose=layout["can1"], modelname="071_can",
            convex=True, model_id=self.object1_id,
        )
        self.can1.set_mass(0.05)
        _rest_on_table(self.can1)
        self.object2_id = int(np.random.choice([0, 1, 2, 3, 5, 6]))
        self.can2 = create_actor(
            scene=self, pose=layout["can2"], modelname="071_can",
            convex=True, model_id=self.object2_id,
        )
        self.can2.set_mass(0.05)
        _rest_on_table(self.can2)
        self.add_prohibit_area(self.plasticbox, padding=0.1)
        self.add_prohibit_area(self.can1, padding=0.05)
        self.add_prohibit_area(self.can2, padding=0.05)
        self.record_layout(layout)
        fp0 = self.plasticbox.get_functional_point(0)[:3]
        fp1 = self.plasticbox.get_functional_point(1)[:3]
        self.route_bins = self.classify_routes([
            ("can1", np.asarray(layout["can1"].p), np.asarray(fp1)),
            ("can2", np.asarray(layout["can2"].p), np.asarray(fp0)),
        ])

    def _move_with_retries(self, arm_tag, targets) -> bool:
        """Try each candidate target pose until one plans.

        @input: an arm tag and candidate 7-value EE target poses.
        @output: whether any candidate planned.
        @scenario: Near the reachable ceiling a fixed approach/retreat height
            is sometimes infeasible; progressively smaller/larger offsets give
            the planner a second chance (the retry ladder proven by
            place_can_basket).
        """
        for index, target in enumerate(targets):
            if index > 0:
                if self.plan_success:
                    return True
                self.plan_success = True
            self.move(self.move_to_pose(arm_tag, target))
        return bool(self.plan_success)

    def place_can_left(self, can, point_id: int, drop_in: bool = False) -> bool:
        """Grasp, lift, transit, measured descend, gentle release, retreat.

        @input: the can actor and the target functional point index;
            ``drop_in`` seats the can just above the rim so it settles inside
            the box (used for the second can, whose floor-level seat would
            collide with the first seated can at the other functional point).
        @output: boolean plan success.
        @scenario: Serial left-arm transfer with home reset between cans.  The
            release is the place_can_basket recipe: descend to a measured
            seat, open to 0.7 twice (a full 1.0 open sweeps the fingers past
            the can and flings it out of the box), then 1.0 for the predicate.
        """
        if not self.plan_success:
            return False
        arm_tag = ArmTag("left")
        self.move(self.grasp_actor(can, arm_tag=arm_tag, pre_grasp_dis=0.1))
        if not self.plan_success:
            return False
        # Lifting z=0.20 right after the grasp pushes the left EEF past its
        # ~1.03 m ceiling; z=0.08 keeps the lift reachable.
        self.move_by_displacement(arm_tag=arm_tag, z=0.08)
        if not self.plan_success:
            return False
        target_pose = self.plasticbox.get_functional_point(point_id)
        # Diagonal transit to a pre pose above the target; pre_dis 0.1 pushes
        # the pre pose EE to z~1.03 (the ceiling), so approach at 0.05 with a
        # 0.03 fallback for near-horizontal grasp axes.
        pres = [self.get_place_pose(can, arm_tag, target_pose,
                                    constrain="free", pre_dis=d)
                for d in (0.05, 0.03)]
        if not self._move_with_retries(arm_tag, pres):
            return False
        # Descend so the can CENTRE reaches the seat height, measured from the
        # live can-to-EE offset so a can that slipped in the fingers is still
        # seated correctly (as in place_can_basket).
        ee = np.asarray(self.get_arm_pose(arm_tag))
        offset = np.asarray(can.get_pose().p[:3]) - ee[:3]
        fp = np.asarray(target_pose[:3], dtype=float)
        # Seat height: the can is released ~2.5 cm above the box bottom so it
        # slips out of the closing-axis grip and FALLS free — releasing a
        # still-gripped can lets the opening finger pads sweep it sideways
        # out of the box (3-6 cm, often over the near rim).  The drop_in
        # fallback releases just above the rim when the interior descend is
        # refused (only ~6 mm hull clearance to the first seated can at the
        # other functional point).
        seat_zs = ([fp[2] + _CAN_HALF_Z + 0.001] if not drop_in else
                   [fp[2] + _CAN_HALF_Z + 0.001,
                    self.box_rim_top + _CAN_HALF_Z + 0.01])
        seats = []
        for seat_z in seat_zs:
            seats.append(np.concatenate(
                [np.array([fp[0], fp[1], seat_z]) - offset, ee[3:]]))
        if not self._move_with_retries(arm_tag, seats):
            return False
        # Gentle release: 0.7 opens just past the can diameter; the middle
        # open(0.7) pairs are 200-step settle holds; the final 1.0 satisfies
        # is_left_gripper_open.
        self.move(self.open_gripper(arm_tag, pos=0.7))
        self.move(self.open_gripper(arm_tag, pos=0.7))
        self.move(self.open_gripper(arm_tag, pos=0.7))
        self.move(self.open_gripper(arm_tag, pos=1.0))
        if not self.plan_success:
            return False
        # Retreat: a fixed raise is sometimes infeasible at the ceiling, so
        # retry with progressively smaller heights; a +y pullback keeps the
        # home path clear of the box airspace but is best-effort — if it
        # cannot plan, reset and head home anyway (the planner routes the
        # empty hand around the box collision hull on its own).
        ee2 = np.asarray(self.get_arm_pose(arm_tag))
        raises = [np.array([ee2[0], ee2[1], ee2[2] + h, *ee2[3:]])
                  for h in (0.08, 0.05, 0.03)]
        self._move_with_retries(arm_tag, raises)
        if self.plan_success:
            self.move_by_displacement(arm_tag=arm_tag, y=0.10)
        self.plan_success = True
        return self.left_return_home()

    def play_once(self) -> dict:
        self.place_can_left(self.can1, 1)
        self.place_can_left(self.can2, 0)
        self.info["info"] = {
            "{A}": f"071_can/base{self.object1_id}",
            "{B}": f"062_plasticbox/base{self.plasticbox_id}",
            "{C}": f"071_can/base{self.object2_id}",
            "{a}": "left",
        }
        return self.info

    def check_success(self) -> bool:
        fp0 = self.plasticbox.get_functional_point(0)[0:2]
        fp1 = self.plasticbox.get_functional_point(1)[0:2]
        dis1 = min(
            float(np.linalg.norm(self.can1.get_pose().p[0:2] - fp0)),
            float(np.linalg.norm(self.can1.get_pose().p[0:2] - fp1)),
        )
        dis2 = min(
            float(np.linalg.norm(self.can2.get_pose().p[0:2] - fp0)),
            float(np.linalg.norm(self.can2.get_pose().p[0:2] - fp1)),
        )
        threshold = 0.04
        return dis1 < threshold and dis2 < threshold \
            and self.is_left_gripper_open() and self.right_arm_stationary()
