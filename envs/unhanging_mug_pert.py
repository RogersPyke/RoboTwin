from ._pert_mixin import PerturbationMixin
from .unhanging_mug import unhanging_mug


class unhanging_mug_pert(PerturbationMixin, unhanging_mug):

    def setup_demo(self, *args, **kwargs):
        kwargs["conservative_mode"] = True
        return super().setup_demo(*args, **kwargs)
# Purpose: Perturbation variant of unhanging_mug (conservative mode).
# Conservative mode uses tighter bounds (smaller cone angle, smaller XY offset)
# to preserve the high success rate required by the more constrained un-task.
# Dependencies: _pert_mixin.PerturbationMixin, unhanging_mug
# Usage: ./collect_data.sh unhanging_mug_pert demo_clean_pert <gpu_id>
#   Same-seed: SOURCE_SEED_PATH=./data/unhanging_mug/demo_clean/seed.txt \
#              ./collect_data.sh unhanging_mug_pert demo_clean_pert <gpu_id>

from ._pert_mixin import PerturbationMixin
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
        return super().setup_demo(*args, **kwargs)
