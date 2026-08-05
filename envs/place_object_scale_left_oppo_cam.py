"""Place-object-scale task with the head camera on the opposite table side."""

from copy import deepcopy

from .place_object_scale_left import place_object_scale_left


class place_object_scale_left_oppo_cam(place_object_scale_left):
    """Reuse the left-arm scale task with only a front-to-back camera flip."""

    def setup_demo(self, **kwargs):
        """Initialize the inherited task with an opposite-side head camera.

        The default Piper head camera is ``[-0.032, -0.45, 1.35]`` and looks
        along ``[0, 0.6, -0.8]``.  This task negates only its table-depth
        position and look direction.  The camera-left axis is also negated so
        that ``cross(forward, left)`` remains an upward-facing axis.
        """
        task_kwargs = dict(kwargs)
        embodiment_config = deepcopy(kwargs["left_embodiment_config"])

        for camera_info in embodiment_config["static_camera_list"]:
            if camera_info["name"] == "head_camera":
                camera_info.update(
                    {
                        "position": [-0.032, 0.45, 1.35],
                        "forward": [0, -0.6, -0.8],
                        "left": [1, 0, 0],
                    }
                )
                break
        else:
            raise ValueError("Piper configuration has no head_camera entry.")

        task_kwargs["left_embodiment_config"] = embodiment_config
        super().setup_demo(**task_kwargs)
