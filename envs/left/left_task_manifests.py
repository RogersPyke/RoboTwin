"""Immutable per-family task manifests for the eleven left-arm families.

Every ``central_cam``/``left_oppo_cam`` pair shares exactly one manifest.  A
pair may not have separate geometry tuning because camera is the only intended
variation.

Initial ranges are the source task ranges shifted into the left-arm workspace
(the active left Piper sits at x=-0.30 m) and kept inside the arm's reliable
reach.  Any range/yaw change must be accompanied by a pilot result, a version
increase, and a new hash.

``scan_object`` is explicitly excluded from this project and has no manifest.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Mapping

from .left_task_base import ActorSamplingSpec, TaskManifest, XYYawRange


def _spec(name: str, x: tuple[float, float], y: tuple[float, float],
          yaw_rad: tuple[float, float], min_clearance_m: float,
          static: bool, base_quat: tuple[float, float, float, float]
          = (1.0, 0.0, 0.0, 0.0)) -> ActorSamplingSpec:
    return ActorSamplingSpec(
        name=name,
        workspace=XYYawRange(x=x, y=y, yaw_rad=yaw_rad),
        minimum_clearance_m=min_clearance_m,
        static=static,
        base_quat=base_quat,
    )


def _manifest(semantic_task: str, specs: list[ActorSamplingSpec], *,
              route_bin_count: int = 4, pilot_seed_count: int = 100,
              min_success_rate: float = 0.20,
              min_crossing_pair_rate: float = 0.0) -> TaskManifest:
    return TaskManifest(
        semantic_task=semantic_task,
        actor_specs={spec.name: spec for spec in specs},
        route_bin_count=route_bin_count,
        pilot_seed_count=pilot_seed_count,
        min_success_rate=min_success_rate,
        min_crossing_pair_rate=min_crossing_pair_rate,
    )


# Shared left-workspace constants.
_LEFT_CENTER = (-0.30, 0.0)
# Arm-base exclusion band: actors should not spawn right under the shoulder.
ARM_BASE_X_BAND = (-0.35, -0.25)

MANIFESTS: dict[str, TaskManifest] = {}


def _register(manifest: TaskManifest) -> None:
    MANIFESTS[manifest.semantic_task] = manifest


_register(_manifest(
    "blocks_ranking_rgb",
    [
        # x upper bound -0.22 keeps every spawn inside the left-arm reachable
        # region (x <= -0.20) so a grasp can always plan.
        _spec("block1", (-0.45, -0.22), (-0.08, 0.05), (0.0, 0.60), 0.10, False,
              base_quat=(1.0, 0.0, 0.0, 0.0)),
        _spec("block2", (-0.45, -0.22), (-0.08, 0.05), (0.0, 0.60), 0.10, False,
              base_quat=(1.0, 0.0, 0.0, 0.0)),
        _spec("block3", (-0.45, -0.22), (-0.08, 0.05), (0.0, 0.60), 0.10, False,
              base_quat=(1.0, 0.0, 0.0, 0.0)),
    ],
))

_register(_manifest(
    "blocks_ranking_size",
    [
        # x upper bound -0.22 keeps every spawn inside the left-arm reachable
        # region (x <= -0.20) so a grasp can always plan.
        _spec("block1", (-0.45, -0.22), (-0.08, 0.05), (0.0, 0.60), 0.10, False,
              base_quat=(1.0, 0.0, 0.0, 0.0)),
        _spec("block2", (-0.45, -0.22), (-0.08, 0.05), (0.0, 0.60), 0.10, False,
              base_quat=(1.0, 0.0, 0.0, 0.0)),
        _spec("block3", (-0.45, -0.22), (-0.08, 0.05), (0.0, 0.60), 0.10, False,
              base_quat=(1.0, 0.0, 0.0, 0.0)),
    ],
))

# DEPRECATED (2026-08-09): the only reliable hanging geometry uses a rack
# tilted ~58 deg about x.  An upright, yaw-only rack (the only orientation
# permitted going forward) cannot physically hold the 039_mug, so the task is
# frozen and must not be collected.  Kept in the registry to preserve the
# manifest hash 28ee798eb3dfcde4 and the wrapper/test contract.
_register(_manifest(
    "hanging_mug",
    [
        # Single-arm pilot (iteration 4): the source x+90 mug orientation is
        # the only one that settles stably on the table, and the left EEF
        # ceiling (~1.03 m) plus the ~0.23 m grasp offset caps the mug
        # functional point near z=0.87.  To bring the hook into reach the rack
        # is rolled about x by 45 deg (fp0 world z ~0.85-0.87) and pinned to a
        # single left-reachable spot; the mug band narrows to the grasp-stable
        # strip that still leaves the "mug rides the tilted pillar" success
        # case above the 0.20 pilot threshold (30-seed rate 0.60).
        _spec("mug", (-0.44, -0.40), (-0.01, 0.01), (0.0, 0.0), 0.10, False,
              base_quat=(0.707, 0.707, 0.0, 0.0)),
        _spec("rack", (-0.29, -0.29), (-0.02, -0.02), (0.0, 0.0), 0.10, True,
              base_quat=(-0.11906, -0.28744, 0.3626, 0.8754)),
    ],
))

_register(_manifest(
    "place_bread_basket",
    [
        # The bread band matches the verified place_bread_skillet bread band
        # (x in [-0.42,-0.36], y in [-0.15,0.02], 30-seed SR 0.93): the left
        # arm's downward grasp of 075_bread is only reliable there.  The basket
        # sweeps x in [-0.30,-0.12], y in [-0.28,-0.02] so every basket sits at
        # least 0.06 m right of the bread band (clearance 0.05 m) and the
        # rejection sampler never locks, while the basket yaw band keeps both
        # camera variants seeing a varied placement.
        _spec("breadbasket", (-0.30, -0.12), (-0.28, -0.02), (0.0, 0.50), 0.05, True,
              base_quat=(0.5, 0.5, 0.5, 0.5)),
        _spec("bread0", (-0.42, -0.36), (-0.15, 0.02), (0.0, 0.50), 0.10, False,
              base_quat=(0.707, 0.707, 0.0, 0.0)),
        _spec("bread1", (-0.42, -0.36), (-0.15, 0.02), (0.0, 0.50), 0.10, False,
              base_quat=(0.707, 0.707, 0.0, 0.0)),
    ],
))

_register(_manifest(
    "place_bread_skillet",
    [
        _spec("bread", (-0.42, -0.36), (-0.15, 0.02), (0.0, 0.50), 0.10, False,
              base_quat=(0.707, 0.707, 0.0, 0.0)),
        # The skillet keeps x >= -0.26 so it always sits >= 0.10 m right of the
        # bread band (bread x-max -0.36) and the clearance test never locks.
        # y sweeps broadly in front of the workspace; the raised skillet keeps
        # its functional point above the 0.76 m table-height predicate.
        _spec("skillet", (-0.26, -0.12), (-0.24, 0.14), (0.0, 0.50), 0.10, True,
              base_quat=(0.0, 0.0, 0.707, 0.707)),
    ],
))

_register(_manifest(
    "place_burger_fries",
    [
        # The tray (0.31 x 0.21 m footprint at scale 2.0) sweeps the left
        # workspace; the two foods spawn in y-split bands above and below the
        # tray so neither settles into the tray collision hull and launches.
        # Any food spawned geometrically inside the tray hull interpenetrates
        # the tray rim (z in [0.739, 0.777]) and the settle phase ejects it,
        # launching the food off the table.  The tray centre therefore stays in
        # y in [-0.16, -0.04] so its footprint y range [-0.245, 0.065] keeps
        # the hamburg band (y >= 0.10, tray y-max plus bun radius and margin)
        # above it and the fries band (y <= -0.27, tray y-min minus carton
        # half-width and margin) below it, with >= 0.10 m y separation for the
        # clearance test.  x sweeps from the far left to just right of centre
        # within the left arm's reliable place reach.
        _spec("tray", (-0.44, -0.08), (-0.16, -0.04), (0.0, 0.00), 0.08, True,
              base_quat=(0.706527, 0.706483, -0.0291356, -0.0291767)),
        _spec("hamburg", (-0.44, -0.36), (0.10, 0.14), (0.0, 0.20), 0.08, False,
              base_quat=(0.5, 0.5, 0.5, 0.5)),
        _spec("frenchfries", (-0.44, -0.36), (-0.33, -0.27), (0.0, 0.20), 0.08, False,
              base_quat=(1.0, 0.0, 0.0, 0.0)),
    ],
))

_register(_manifest(
    "place_can_basket",
    [
        # The basket sweeps the left-front quadrant (x in [-0.36, -0.16],
        # y in [-0.28, -0.15]) after a leftward shift from the source x=0.02
        # centre-line position, keeping its solid plate cluster inside the left
        # arm's reliable reach.  The quat (0.5, 0.5, 0.5, 0.5) with zero yaw is
        # deterministic, which keeps the plate offset (+0.05, -0.04) used by
        # place_can_basket_left_impl valid across every episode.  The basket
        # stays >= 0.15 m below the can band (y >= 0.15 separation) so the
        # clearance test never locks, and the plate offset tracks the basket
        # centre, so a shifted basket still receives the can.
        _spec("basket", (-0.36, -0.16), (-0.28, -0.15), (0.0, 0.00), 0.15, True,
              base_quat=(0.5, 0.5, 0.5, 0.5)),
        # The can spawns on its side (identity quat) near the left arm base
        # (x in [-0.25, -0.20], y in [0, 0.08]), matching the source layout
        # that keeps every downward grasp IK-reachable.  y is capped at 0.08:
        # cans at y ~ 0.10 sit at the edge of the left grasp zone and fail
        # choose_best_pose.  The lying can is transported and lowered onto the
        # basket plate rather than dropped.
        _spec("can", (-0.25, -0.20), (0.0, 0.08), (0.0, 0.00), 0.15, False,
              base_quat=(1.0, 0.0, 0.0, 0.0)),
    ],
))

_register(_manifest(
    "place_cans_plasticbox",
    [
        # The plastic box sweeps x in [-0.36, -0.14] at a deep-y band in
        # [-0.28, -0.23], behind the two can bands (y in [-0.15, -0.07] on
        # both x flanks).  The box's y <= -0.23 keeps every can-to-box
        # separation >= 0.08 m (the clearance) so the rejection
        # sampler never locks, and keeping the box behind the cans (not in
        # front) avoids blocking the left arm's grasp approach over them.  The
        # depth matches the place_can_basket basket band (y in [-0.28,-0.15],
        # 30-seed SR 0.37) so it sits inside the verified reach.
        _spec("plasticbox", (-0.36, -0.14), (-0.28, -0.23), (0.0, 0.00), 0.08, True,
              base_quat=(0.5, 0.5, 0.5, 0.5)),
        _spec("can1", (-0.42, -0.36), (-0.15, -0.07), (0.0, 0.00), 0.08, False,
              base_quat=(0.5, 0.5, 0.5, 0.5)),
        _spec("can2", (-0.16, -0.10), (-0.15, -0.07), (0.0, 0.00), 0.08, False,
              base_quat=(0.5, 0.5, 0.5, 0.5)),
    ],
))

_register(_manifest(
    "place_dual_shoes",
    [
        # The shoe box (0.18 x 0.26 m footprint under the stand90 quat) is a
        # tall static hull; a shoe that spawns geometrically inside that
        # footprint interpenetrates the box rim and the settle phase ejects it,
        # raising UnStableError.  Both shoes therefore spawn in narrow bands
        # above the box (y >= 0.13) with an x split so the two shoes never
        # collide.  The box sweeps x in [-0.44, -0.16] and y in [-0.18, -0.02];
        # its y half-extent is 0.13, so the footprint top sits at most at
        # y=0.11, still below the shoe bands' y=0.13, and the y separation to
        # the shoes is >= 0.15 m, above the 0.10 m clearance.
        # Physical-reset grasp+lift maps over the left workspace showed that
        # only the y=0.13..0.14 row keeps a lift reachable for the left arm:
        # y below 0.13 (left shoe at 0.116-0.127) and the former right band x
        # in [-0.20,-0.14] at y above 0.14 both fail with plan=False after
        # grasp.  The bands below are the verified contiguous OK subregions:
        # left x in [-0.44,-0.40] (y=0.13 row OK in [-0.46,-0.40], y=0.14 row
        # OK in [-0.44,-0.30]) and right x in [-0.20,-0.16] (y=0.13 row OK in
        # [-0.20,-0.14], y=0.14 row OK in [-0.24,-0.16]).  The two bands keep a
        # 0.20 m x separation, well above the 0.10 m clearance.  The box quat
        # is stand90, matching the shoe base quat: the align place constraint
        # then keeps the shoe standing (a small wrist rotation) instead of
        # requiring a lateral 120-degree roll that the left arm cannot reach.
        _spec("shoe_box", (-0.44, -0.16), (-0.18, -0.02), (0.0, 0.00), 0.10, True,
              base_quat=(0.707, 0.707, 0.0, 0.0)),
        _spec("left_shoe", (-0.44, -0.40), (0.13, 0.14), (0.0, 0.50), 0.10, False,
              base_quat=(0.707, 0.707, 0.0, 0.0)),
        _spec("right_shoe", (-0.20, -0.16), (0.13, 0.14), (0.0, 0.50), 0.10, False,
              base_quat=(0.707, 0.707, 0.0, 0.0)),
    ],
))

_register(_manifest(
    "stack_blocks_two",
    [
        _spec("block1", (-0.45, -0.10), (-0.08, 0.05), (0.0, 0.60), 0.10, False,
              base_quat=(1.0, 0.0, 0.0, 0.0)),
        _spec("block2", (-0.45, -0.10), (-0.08, 0.05), (0.0, 0.60), 0.10, False,
              base_quat=(1.0, 0.0, 0.0, 0.0)),
    ],
))

_register(_manifest(
    "stack_bowls_two",
    [
        _spec("bowl1", (-0.50, -0.05), (-0.15, 0.15), (0.0, 0.00), 0.13, False,
              base_quat=(0.5, 0.5, 0.5, 0.5)),
        _spec("bowl2", (-0.50, -0.05), (-0.15, 0.15), (0.0, 0.00), 0.13, False,
              base_quat=(0.5, 0.5, 0.5, 0.5)),
    ],
))


def get_manifest(semantic_task: str) -> TaskManifest:
    """Return the immutable manifest for one semantic task name.

    @input: a source semantic task name such as ``place_burger_fries``.
    @output: the shared ``TaskManifest``.
    @scenario: Keep one source of geometry truth per family.
    """
    if semantic_task not in MANIFESTS:
        raise KeyError(f"Unknown left-arm semantic task: {semantic_task!r}")
    return MANIFESTS[semantic_task]


def _canonical_dict(manifest: TaskManifest) -> dict:
    payload = asdict(manifest)
    # Keep actor spec ordering deterministic.
    return {"actor_specs": [payload["actor_specs"][name] for name in sorted(payload["actor_specs"])],
            "semantic_task": payload["semantic_task"],
            "route_bin_count": payload["route_bin_count"],
            "pilot_seed_count": payload["pilot_seed_count"],
            "min_success_rate": payload["min_success_rate"],
            "min_crossing_pair_rate": payload["min_crossing_pair_rate"]}


def manifest_hash(manifest: TaskManifest) -> str:
    """Return a stable hex hash of one manifest.

    @input: any ``TaskManifest``.
    @output: 16-hex-character digest of the canonical serialization.
    @scenario: Prove that both camera variants share identical geometry.
    """
    canonical = json.dumps(_canonical_dict(manifest), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def all_manifests() -> Mapping[str, TaskManifest]:
    """Return the immutable registry of all semantic manifests."""
    return dict(MANIFESTS)
