import math
from copy import deepcopy

import numpy as np
import transforms3d as t3d


class PerturbationMixin:
    """
    Lightweight perturbation mixin for *_pert tasks.
    """

    def _load_pert_cfg(self, kwargs):
        cfg = kwargs.get("perturbation", {})
        conservative = bool(kwargs.get("conservative_mode", False))

        self._pert_cfg = {
            "enabled": bool(cfg.get("enabled", True)),
            "grasp_xy_jitter": float(cfg.get("grasp_xy_jitter", 0.008)),
            "grasp_yaw_jitter_deg": float(cfg.get("grasp_yaw_jitter_deg", 6.0)),
            "pre_place_xy_jitter": float(cfg.get("pre_place_xy_jitter", 0.02)),
            "conservative_xy_scale": float(cfg.get("conservative_xy_scale", 0.6)),
            "conservative_yaw_scale": float(cfg.get("conservative_yaw_scale", 0.5)),
            "conservative_mode": conservative,
        }

        self._pert_meta = {
            "enabled": self._pert_cfg["enabled"],
            "conservative_mode": self._pert_cfg["conservative_mode"],
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
