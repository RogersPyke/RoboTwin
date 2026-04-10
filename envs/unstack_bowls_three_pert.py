from ._enh_util._pert_mixin import PerturbationMixin
from .unstack_bowls_three import unstack_bowls_three


class unstack_bowls_three_pert(PerturbationMixin, unstack_bowls_three):
    """
    Diverse data collection for unstack_bowls_three with conservative perturbation.
    Conservative mode uses tighter bounds to preserve success rate on the
    stack-aware grasp constraint (shallow grasp to avoid lifting two bowls).

    @input:  same args as unstack_bowls_three (passed via collect_data.py)
    @output: same data format as unstack_bowls_three (diverse hdf5 episodes)
    @scenario: Augmented expert data for the three-bowl unstack task.
    """

    def setup_demo(self, *args, **kwargs):
        """
        Force conservative_mode before delegating to PerturbationMixin.setup_demo.

        @input:  same as unstack_bowls_three.setup_demo(*args, **kwargs)
        @output: delegates to super().setup_demo after injecting conservative_mode
        @scenario: Tighter perturbation bounds for bowl unstack un-task.
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
