import logging
import math
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import transforms3d as t3d


def _build_logger():
    logger = logging.getLogger(__name__)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    logs_dir = Path("./logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    tz = timezone(timedelta(hours=8))
    stamp = datetime.now(tz).strftime("%Y%m%d%H%M%S")
    file_path = logs_dir / f"cone_bezier_planner_{stamp}.log"
    fmt = logging.Formatter("[ConeBezierPlanner] [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(file_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)
    return logger


_LOGGER = _build_logger()


class ConeBezierPlanner:
    """
    Cubic-Bezier cone-constrained waypoint planner.
    """

    def __init__(self, cfg):
        """
        @input cfg: [dict, planner config, keys listed below]
            - half_angle_deg: [float, (0, 89), cone half-angle]
            - candidate_trials: [int, >=1, candidate generation attempts]
            - waypoint_count_min/max: [int, >=1, sampled waypoint count range]
            - ctrl_axial_range_1/2: [list[float], 2 elems in (0,1), control-point axial ranges]
            - twist_range_deg: [float, [0,180], control-point direction perturbation]
            - orientation_jitter_deg: [float, >=0, quaternion jitter scale]
            - cone_task_types: [dict, {"convergent":[...], "divergent":[...]}, explicit task mapping]
        @output: [None, -, planner initialized]
        @scenario: Configure deterministic geometry and stochastic candidate sampling.
        """
        cfg = cfg or {}
        self.half_angle_deg = float(cfg.get("half_angle_deg", 10.0))
        self.candidate_trials = max(1, int(cfg.get("candidate_trials", 6)))
        self.waypoint_count_min = max(1, int(cfg.get("waypoint_count_min", 2)))
        self.waypoint_count_max = max(self.waypoint_count_min, int(cfg.get("waypoint_count_max", 4)))
        self.ctrl_axial_range_1 = self._normalize_range(cfg.get("ctrl_axial_range_1", [0.15, 0.45]), 0.01, 0.49)
        self.ctrl_axial_range_2 = self._normalize_range(cfg.get("ctrl_axial_range_2", [0.55, 0.85]), 0.51, 0.99)
        self.twist_range_deg = max(0.0, float(cfg.get("twist_range_deg", 45.0)))
        self.orientation_jitter_deg = max(0.0, float(cfg.get("orientation_jitter_deg", 10.0)))

        cone_map = deepcopy(cfg.get("cone_task_types", {}))
        self.convergent_tasks = set(cone_map.get("convergent", []) or [])
        self.divergent_tasks = set(cone_map.get("divergent", []) or [])
        _LOGGER.debug(
            "init cfg half_angle_deg=%.3f trials=%d wp=[%d,%d] twist=%.3f",
            self.half_angle_deg,
            self.candidate_trials,
            self.waypoint_count_min,
            self.waypoint_count_max,
            self.twist_range_deg,
        )

    @staticmethod
    def _normalize_range(raw_range, hard_min, hard_max):
        if not isinstance(raw_range, (list, tuple)) or len(raw_range) != 2:
            return [hard_min, hard_max]
        low = max(hard_min, float(raw_range[0]))
        high = min(hard_max, float(raw_range[1]))
        if high < low:
            low, high = hard_min, hard_max
        return [low, high]

    def _determine_cone_type(self, task_name):
        """
        @input task_name: [str, class/task name, non-empty]
        @output: [str, "convergent"/"divergent", success indicator]
        @scenario: Select cone direction based on explicit task mapping.
        """
        if task_name in self.convergent_tasks:
            return "convergent"
        if task_name in self.divergent_tasks:
            return "divergent"
        raise KeyError(
            f"task_name={task_name} is not listed in cone_task_types. "
            "Please configure explicit convergent/divergent task names."
        )

    @staticmethod
    def _normalize_quat(quat_wxyz):
        quat = np.array(quat_wxyz, dtype=np.float64)
        n = float(np.linalg.norm(quat))
        if n < 1e-8:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return quat / n

    @staticmethod
    def _build_lateral_frame(axis_unit):
        """
        @input axis_unit: [ndarray(3,), unit vector, principal axis]
        @output: [tuple(ndarray(3,), ndarray(3,)), orthonormal lateral basis]
        @scenario: Build a stable frame perpendicular to motion direction.
        """
        world_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(float(np.dot(axis_unit, world_z))) > 0.9:
            seed = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            seed = world_z
        e1 = seed - float(np.dot(seed, axis_unit)) * axis_unit
        e1_norm = float(np.linalg.norm(e1))
        if e1_norm < 1e-8:
            e1 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            e1_norm = 1.0
        e1 = e1 / e1_norm
        e2 = np.cross(axis_unit, e1)
        e2_norm = float(np.linalg.norm(e2))
        if e2_norm < 1e-8:
            e2 = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            e2_norm = 1.0
        e2 = e2 / e2_norm
        return e1, e2

    @staticmethod
    def _cone_radius(t_ratio, r_max, cone_type):
        if cone_type == "convergent":
            return r_max * (1.0 - t_ratio)
        return r_max * t_ratio

    @staticmethod
    def _evaluate_bezier(s, c1, c2, t, ratio):
        omt = 1.0 - ratio
        return omt * omt * omt * s + 3.0 * omt * omt * ratio * c1 + 3.0 * omt * ratio * ratio * c2 + ratio * ratio * ratio * t

    def _sample_control_points(self, s, t, axis_vec, r_max, cone_type, e1, e2, rng):
        """
        @input s/t: [ndarray(3,), start/target, valid finite]
        @input axis_vec: [ndarray(3,), t-s, finite]
        @input r_max: [float, >=0, maximum cone radius]
        @input cone_type: [str, convergent/divergent, mapping result]
        @input e1/e2: [ndarray(3,), unit vectors, lateral basis]
        @input rng: [numpy random interface, must support uniform()]
        @output: [tuple(ndarray(3,), ndarray(3,)), sampled control points]
        @scenario: Sample cubic control points with coherent lateral direction.
        """
        t_c1 = float(rng.uniform(self.ctrl_axial_range_1[0], self.ctrl_axial_range_1[1]))
        t_c2 = float(rng.uniform(self.ctrl_axial_range_2[0], self.ctrl_axial_range_2[1]))

        theta_primary = float(rng.uniform(-math.pi, math.pi))
        twist_rad = self.twist_range_deg * math.pi / 180.0
        theta_1 = theta_primary + float(rng.uniform(-twist_rad, twist_rad))
        theta_2 = theta_primary + float(rng.uniform(-twist_rad, twist_rad))

        r1 = self._cone_radius(t_c1, r_max, cone_type)
        r2 = self._cone_radius(t_c2, r_max, cone_type)

        mag_1 = float(rng.uniform(0.0, max(0.0, r1)))
        mag_2 = float(rng.uniform(0.0, max(0.0, r2)))

        lateral_1 = mag_1 * (math.cos(theta_1) * e1 + math.sin(theta_1) * e2)
        lateral_2 = mag_2 * (math.cos(theta_2) * e1 + math.sin(theta_2) * e2)

        c1 = s + t_c1 * axis_vec + lateral_1
        c2 = s + t_c2 * axis_vec + lateral_2
        _LOGGER.debug(
            "sample controls t1=%.4f t2=%.4f mag1=%.6f mag2=%.6f",
            t_c1,
            t_c2,
            mag_1,
            mag_2,
        )
        return c1, c2

    def _clamp_to_cone(self, pos, ratio, s, axis_vec, axis_unit, r_max, cone_type):
        """
        @input pos: [ndarray(3,), sampled waypoint pos, finite]
        @input ratio: [float, [0,1], progress on axis]
        @input s/axis_vec/axis_unit: [ndarray(3,), geometry basis]
        @input r_max: [float, >=0, max cone radius]
        @input cone_type: [str, convergent/divergent, mapping result]
        @output: [ndarray(3,), clamped waypoint, success indicator]
        @scenario: Force every waypoint to stay in cone; prevent oscillatory off-cone motion.
        """
        on_axis = s + ratio * axis_vec
        offset = pos - on_axis
        axial_comp = float(np.dot(offset, axis_unit))
        lateral = offset - axial_comp * axis_unit
        lateral_dist = float(np.linalg.norm(lateral))
        allowed = self._cone_radius(ratio, r_max, cone_type)
        if lateral_dist > allowed and lateral_dist > 1e-8:
            pos = on_axis + (allowed / lateral_dist) * lateral
            _LOGGER.debug("clamp waypoint ratio=%.4f lateral=%.6f allowed=%.6f", ratio, lateral_dist, allowed)
        return pos

    @staticmethod
    def _slerp_wxyz(q0, q1, ratio):
        q0 = ConeBezierPlanner._normalize_quat(q0)
        q1 = ConeBezierPlanner._normalize_quat(q1)
        dot = float(np.dot(q0, q1))
        if dot < 0.0:
            q1 = -q1
            dot = -dot
        dot = min(1.0, max(-1.0, dot))
        if dot > 0.9995:
            q = q0 + ratio * (q1 - q0)
            return ConeBezierPlanner._normalize_quat(q)
        theta_0 = math.acos(dot)
        sin_theta_0 = math.sin(theta_0)
        theta = theta_0 * ratio
        s0 = math.sin(theta_0 - theta) / sin_theta_0
        s1 = math.sin(theta) / sin_theta_0
        return s0 * q0 + s1 * q1

    def _interpolate_orientation(self, q_start, q_target, ratio, noise_scale_deg, rng):
        """
        @input q_start/q_target: [array-like(4,), quaternion wxyz, normalized or normalizable]
        @input ratio: [float, [0,1], interpolation ratio]
        @input noise_scale_deg: [float, >=0, orientation noise bound]
        @input rng: [numpy random interface, supports uniform()]
        @output: [ndarray(4,), quaternion wxyz, normalized]
        @scenario: Build smooth orientation evolution along geometric waypoints.
        """
        q_base = self._slerp_wxyz(q_start, q_target, ratio)
        if noise_scale_deg <= 0.0:
            return self._normalize_quat(q_base)
        rpy = rng.uniform(-noise_scale_deg, noise_scale_deg, size=3) * math.pi / 180.0
        q_delta = t3d.euler.euler2quat(float(rpy[0]), float(rpy[1]), float(rpy[2]), axes="sxyz")
        q_out = t3d.quaternions.qmult(q_delta, q_base)
        return self._normalize_quat(q_out)

    def _generate_single_candidate(self, start_pose_7d, target_pose_7d, cone_type, rng):
        s = np.array(start_pose_7d[:3], dtype=np.float64)
        t = np.array(target_pose_7d[:3], dtype=np.float64)
        axis_vec = t - s
        length = float(np.linalg.norm(axis_vec))
        if length < 1e-8:
            return {"strategy": "cone_cubic_bezier", "waypoints": []}
        axis_unit = axis_vec / length
        e1, e2 = self._build_lateral_frame(axis_unit)
        r_max = max(0.0, length * math.tan(self.half_angle_deg * math.pi / 180.0))
        c1, c2 = self._sample_control_points(s, t, axis_vec, r_max, cone_type, e1, e2, rng)

        wp_count = int(rng.randint(self.waypoint_count_min, self.waypoint_count_max + 1))
        ratios = np.linspace(0.0, 1.0, num=wp_count + 2, dtype=np.float64)[1:-1]

        q_start = np.array(start_pose_7d[3:7], dtype=np.float64)
        q_target = np.array(target_pose_7d[3:7], dtype=np.float64)
        waypoints = []
        for ratio in ratios:
            pos = self._evaluate_bezier(s, c1, c2, t, float(ratio))
            pos = self._clamp_to_cone(pos, float(ratio), s, axis_vec, axis_unit, r_max, cone_type)
            allowed = self._cone_radius(float(ratio), r_max, cone_type)
            noise_scale = 0.0
            if r_max > 1e-8:
                noise_scale = self.orientation_jitter_deg * (allowed / r_max)
            quat = self._interpolate_orientation(q_start, q_target, float(ratio), noise_scale, rng)
            pose = target_pose_7d.copy()
            pose[:3] = pos.tolist()
            pose[3:7] = quat.tolist()
            waypoints.append(pose)
        return {"strategy": "cone_cubic_bezier", "waypoints": waypoints}

    def generate_candidates(self, start_pose_7d, target_pose_7d, task_name, rng=None):
        """
        @input start_pose_7d: [list[float], len=7 [x,y,z,qw,qx,qy,qz], finite]
        @input target_pose_7d: [list[float], len=7 [x,y,z,qw,qx,qy,qz], finite]
        @input task_name: [str, valid configured task name]
        @input rng: [numpy random interface or None, optional random source]
        @output: [list[dict], planner candidates with waypoint chains]
        @scenario: Provide multiple cone-constrained cubic candidates for segment planning.
        """
        if rng is None:
            rng = np.random
        cone_type = self._determine_cone_type(task_name)
        _LOGGER.debug("generate task=%s cone_type=%s", task_name, cone_type)
        candidates = []
        for _ in range(self.candidate_trials):
            candidate = self._generate_single_candidate(start_pose_7d, target_pose_7d, cone_type, rng)
            candidates.append(candidate)
        return candidates
