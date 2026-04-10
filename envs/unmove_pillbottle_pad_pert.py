from ._enh_util._pert_mixin import PerturbationMixin
from .unmove_pillbottle_pad import unmove_pillbottle_pad


class unmove_pillbottle_pad_pert(PerturbationMixin, unmove_pillbottle_pad):
    def setup_demo(self, *args, **kwargs):
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
