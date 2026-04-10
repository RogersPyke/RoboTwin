from ._enh_util._pert_mixin import PerturbationMixin
from .stack_blocks_three import stack_blocks_three


class stack_blocks_three_pert(PerturbationMixin, stack_blocks_three):
    """
    Diverse data collection for stack_blocks_three.
    Grasp approaches sampled from a cone; pre-place poses offset randomly in XY.

    @input:  same args as stack_blocks_three (passed via collect_data.py)
    @output: same data format as stack_blocks_three (diverse hdf5 episodes)
    @scenario: Augmented expert data for the three-block stacking task.
    """

    def setup_demo(self, *args, **kwargs):
        pert = kwargs.get("perturbation") or {}
        kwargs["perturbation"] = pert
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
