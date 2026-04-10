from ._enh_util._pert_mixin import PerturbationMixin
from .unhanging_mug import unhanging_mug


class unhanging_mug_pert(PerturbationMixin, unhanging_mug):
    """
    Diverse data collection for unhanging_mug with conservative perturbation.
    Forces conservative_mode=True so the mixin uses tighter cone angle and
    smaller pre-place XY offset to keep success rate as high as possible.

    @input:  same args as unhanging_mug (passed via collect_data.py)
    @output: same data format as unhanging_mug (hdf5 episodes, diverse trajectories)
    @scenario: Augmented expert data for the constrained off-rack task.
    """

    def setup_demo(self, *args, **kwargs):
        """
        Force conservative_mode before delegating to PerturbationMixin.setup_demo.

        @input:  same as unhanging_mug.setup_demo(*args, **kwargs)
        @output: delegates to super().setup_demo after injecting conservative_mode
        @scenario: Tighter perturbation bounds for off-rack un-task.
        """
        pert = kwargs.get("perturbation") or {}
        kwargs["perturbation"] = pert
        pert.setdefault("conservative_mode", True)
        planner = pert.get("planner_augmentation") or {}
        pert["planner_augmentation"] = planner
        planner.setdefault(
            "cone_task_types",
            {
                "convergent": [
                    "stack_bowls_three_pert",
                    "stack_blocks_three_pert",
                    "move_pillbottle_pad_pert",
                    "hanging_mug_pert",
                ],
                "divergent": [
                    "unstack_bowls_three_pert",
                    "unstack_blocks_three_pert",
                    "unmove_pillbottle_pad_pert",
                    "unhanging_mug_pert",
                ],
            },
        )
        return super().setup_demo(*args, **kwargs)
