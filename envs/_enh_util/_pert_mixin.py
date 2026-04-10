import math
from copy import deepcopy

import numpy as np
import transforms3d as t3d

from ._cone_bezier_planner import ConeBezierPlanner


class PerturbationMixin:
    """
    Lightweight perturbation mixin for *_pert tasks.
    """

    def _load_pert_cfg(self, kwargs):
        cfg = kwargs.get("perturbation", {}) or {}
        conservative = bool(kwargs.get("conservative_mode", cfg.get("conservative_mode", False)))
        planner_cfg = cfg.get("planner_augmentation", {}) or kwargs.get("planner_augmentation", {}) or {}

        self._pert_cfg = {
            "enabled": bool(cfg.get("enabled", True)),
            "grasp_xy_jitter": float(cfg.get("grasp_xy_jitter", 0.008)),
            "grasp_yaw_jitter_deg": float(cfg.get("grasp_yaw_jitter_deg", 6.0)),
            "pre_place_xy_jitter": float(cfg.get("pre_place_xy_jitter", 0.02)),
            "conservative_xy_scale": float(cfg.get("conservative_xy_scale", 0.6)),
            "conservative_yaw_scale": float(cfg.get("conservative_yaw_scale", 0.5)),
            "conservative_mode": conservative,
        }

        self._plan_aug_cfg = {
            "enabled": bool(planner_cfg.get("enabled", False)),
            "candidate_trials": int(planner_cfg.get("candidate_trials", 6)),
            "waypoint_count_min": int(planner_cfg.get("waypoint_count_min", 2)),
            "waypoint_count_max": int(planner_cfg.get("waypoint_count_max", 4)),
            "half_angle_deg": float(planner_cfg.get("half_angle_deg", 10.0)),
            "twist_range_deg": float(planner_cfg.get("twist_range_deg", 45.0)),
            "orientation_jitter_deg": float(planner_cfg.get("orientation_jitter_deg", 10.0)),
            "ctrl_axial_range_1": deepcopy(planner_cfg.get("ctrl_axial_range_1", [0.15, 0.45])),
            "ctrl_axial_range_2": deepcopy(planner_cfg.get("ctrl_axial_range_2", [0.55, 0.85])),
            "cone_task_types": deepcopy(planner_cfg.get("cone_task_types", {})),
            "conservative_cone_scale": float(planner_cfg.get("conservative_cone_scale", 0.7)),
            "conservative_orientation_scale": float(planner_cfg.get("conservative_orientation_scale", 0.6)),
        }
        if self._plan_aug_cfg["waypoint_count_min"] < 1:
            self._plan_aug_cfg["waypoint_count_min"] = 1
        if self._plan_aug_cfg["waypoint_count_max"] < self._plan_aug_cfg["waypoint_count_min"]:
            self._plan_aug_cfg["waypoint_count_max"] = self._plan_aug_cfg["waypoint_count_min"]
        if self._plan_aug_cfg["candidate_trials"] < 1:
            self._plan_aug_cfg["candidate_trials"] = 1

        planner_runtime_cfg = deepcopy(self._plan_aug_cfg)
        if self._pert_cfg.get("conservative_mode", False):
            planner_runtime_cfg["half_angle_deg"] *= planner_runtime_cfg["conservative_cone_scale"]
            planner_runtime_cfg["orientation_jitter_deg"] *= planner_runtime_cfg["conservative_orientation_scale"]
        self._cone_planner = ConeBezierPlanner(planner_runtime_cfg)

        self._pert_meta = {
            "enabled": self._pert_cfg["enabled"],
            "conservative_mode": self._pert_cfg["conservative_mode"],
            "planner_augmentation_enabled": self._plan_aug_cfg["enabled"],
            "planner_augmentation_strategies": ["cone_cubic_bezier"],
        }

    def setup_demo(self, *args, **kwargs):
        self._load_pert_cfg(kwargs)
        return super().setup_demo(*args, **kwargs)

    def _jitter_xy(self, pose_7d, jitter):
        if pose_7d is None:
            return None
        pose = deepcopy(pose_7d)
        pose[0] += float(np.random.uniform(-jitter, jitter))
        pose[1] += float(np.random.uniform(-jitter, jitter))
        return pose

    def _jitter_yaw(self, pose_7d, jitter_deg):
        if pose_7d is None:
            return None
        pose = np.array(deepcopy(pose_7d), dtype=np.float64)
        yaw = float(np.random.uniform(-jitter_deg, jitter_deg)) * math.pi / 180.0
        q_orig = pose[3:7]
        q_yaw = t3d.euler.euler2quat(0.0, 0.0, yaw, axes="sxyz")
        q_new = t3d.quaternions.qmult(q_yaw, q_orig)
        q_new = q_new / np.linalg.norm(q_new)
        pose[3:7] = q_new
        return pose.tolist()

    def _is_plan_aug_active(self):
        return bool(self._plan_aug_cfg.get("enabled", False) and self.need_plan)

    def _normalize_pose_7d(self, pose_7d):
        if pose_7d is None:
            return None
        if hasattr(pose_7d, "p") and hasattr(pose_7d, "q"):
            return pose_7d.p.tolist() + pose_7d.q.tolist()
        return np.array(deepcopy(pose_7d), dtype=np.float64).tolist()

    def _apply_orientation_noise(self, pose_7d, jitter_deg):
        if pose_7d is None or jitter_deg <= 0.0:
            return pose_7d
        pose = np.array(deepcopy(pose_7d), dtype=np.float64)
        rpy = np.random.uniform(-jitter_deg, jitter_deg, size=3) * math.pi / 180.0
        q_delta = t3d.euler.euler2quat(float(rpy[0]), float(rpy[1]), float(rpy[2]), axes="sxyz")
        q_new = t3d.quaternions.qmult(q_delta, pose[3:7])
        q_new = q_new / np.linalg.norm(q_new)
        pose[3:7] = q_new
        return pose.tolist()

    def _plan_single_segment(self, arm_tag, pose_7d, constraint_pose, last_full_qpos=None, last_arm_qpos=None):
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

    def _merge_segment_results(self, segment_results):
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

    def _try_plan_waypoint_chain(self, arm_tag, target_pose, waypoints, constraint_pose):
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

    def _plan_augmented_result(self, arm_tag, target_pose, constraint_pose):
        start_pose = self._normalize_pose_7d(self.get_arm_pose(arm_tag))
        target_pose = self._normalize_pose_7d(target_pose)
        if start_pose is None or target_pose is None:
            return None
        task_name = self.__class__.__name__
        try:
            candidates = self._cone_planner.generate_candidates(
                start_pose_7d=start_pose,
                target_pose_7d=target_pose,
                task_name=task_name,
                rng=np.random,
            )
        except Exception:
            self._pert_meta["planner_augmentation_last"] = {
                "strategy": "cone_candidate_generation_failed",
                "waypoint_count": 0,
            }
            return None

        for candidate in candidates:
            merged = self._try_plan_waypoint_chain(
                arm_tag=arm_tag,
                target_pose=target_pose,
                waypoints=candidate["waypoints"],
                constraint_pose=constraint_pose,
            )
            if merged is None:
                continue
            self._pert_meta["planner_augmentation_last"] = {
                "strategy": candidate["strategy"],
                "waypoint_count": int(len(candidate["waypoints"])),
            }
            return merged

        self._pert_meta["planner_augmentation_last"] = {
            "strategy": "cone_candidates_failed",
            "waypoint_count": 0,
        }
        return None

    def left_move_to_pose(
        self,
        pose,
        constraint_pose=None,
        use_point_cloud=False,
        use_attach=False,
        save_freq=-1,
    ):
        if not self._is_plan_aug_active():
            return super().left_move_to_pose(
                pose=pose,
                constraint_pose=constraint_pose,
                use_point_cloud=use_point_cloud,
                use_attach=use_attach,
                save_freq=save_freq,
            )
        if not self.plan_success:
            return
        target_pose = self._normalize_pose_7d(pose)
        if target_pose is None:
            self.plan_success = False
            return
        result = self._plan_augmented_result(
            arm_tag="left",
            target_pose=target_pose,
            constraint_pose=constraint_pose,
        )
        if result is None or result.get("status") != "Success":
            self.plan_success = False
            return
        self.left_joint_path.append(deepcopy(result))
        return result

    def right_move_to_pose(
        self,
        pose,
        constraint_pose=None,
        use_point_cloud=False,
        use_attach=False,
        save_freq=-1,
    ):
        if not self._is_plan_aug_active():
            return super().right_move_to_pose(
                pose=pose,
                constraint_pose=constraint_pose,
                use_point_cloud=use_point_cloud,
                use_attach=use_attach,
                save_freq=save_freq,
            )
        if not self.plan_success:
            return
        target_pose = self._normalize_pose_7d(pose)
        if target_pose is None:
            self.plan_success = False
            return
        result = self._plan_augmented_result(
            arm_tag="right",
            target_pose=target_pose,
            constraint_pose=constraint_pose,
        )
        if result is None or result.get("status") != "Success":
            self.plan_success = False
            return
        self.right_joint_path.append(deepcopy(result))
        return result

    def choose_grasp_pose(self, actor, arm_tag, pre_dis=0.1, target_dis=0, contact_point_id=None):
        result = super().choose_grasp_pose(
            actor,
            arm_tag=arm_tag,
            pre_dis=pre_dis,
            target_dis=target_dis,
            contact_point_id=contact_point_id,
        )

        if result is None or not self._pert_cfg.get("enabled", True) or not self.need_plan:
            return result

        pre_pose, grasp_pose = result
        xy_jitter = self._pert_cfg["grasp_xy_jitter"]
        yaw_jitter_deg = self._pert_cfg["grasp_yaw_jitter_deg"]

        if self._pert_cfg.get("conservative_mode", False):
            xy_jitter *= self._pert_cfg["conservative_xy_scale"]
            yaw_jitter_deg *= self._pert_cfg["conservative_yaw_scale"]

        pre_pose = self._jitter_xy(pre_pose, xy_jitter)
        grasp_pose = self._jitter_xy(grasp_pose, xy_jitter)
        grasp_pose = self._jitter_yaw(grasp_pose, yaw_jitter_deg)

        self._pert_meta["grasp_xy_jitter"] = float(xy_jitter)
        self._pert_meta["grasp_yaw_jitter_deg"] = float(yaw_jitter_deg)

        return pre_pose, grasp_pose

    def place_actor(
        self,
        actor,
        arm_tag,
        target_pose,
        functional_point_id=None,
        pre_dis=0.1,
        dis=0.02,
        is_open=True,
        **args,
    ):
        arm, actions = super().place_actor(
            actor,
            arm_tag,
            target_pose,
            functional_point_id=functional_point_id,
            pre_dis=pre_dis,
            dis=dis,
            is_open=is_open,
            **args,
        )

        if not self._pert_cfg.get("enabled", True) or not self.need_plan:
            return arm, actions

        if not actions:
            return arm, actions

        xy_jitter = self._pert_cfg["pre_place_xy_jitter"]
        if self._pert_cfg.get("conservative_mode", False):
            xy_jitter *= self._pert_cfg["conservative_xy_scale"]

        if getattr(actions[0], "target_pose", None) is not None:
            actions[0].target_pose = self._jitter_xy(actions[0].target_pose, xy_jitter)

        self._pert_meta["pre_place_xy_jitter"] = float(xy_jitter)
        return arm, actions
